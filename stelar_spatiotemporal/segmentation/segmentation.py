import geopandas as gpd
import os
import json
import numpy as np
import glob
from pyproj.crs.crs import CRS
from shapely import unary_union
from shapely.geometry.base import BaseGeometry
import shutil
import datetime as dt
import math

from .bands_data_package import BandsDataPackage
from ..preprocessing.preprocessing import split_patch_into_patchlets, combine_npys_into_eopatches
from ..eolearn.core import EOPatch, OverwritePermission, FeatureType
from ..eoflow.models.segmentation_unets import ResUnetA
from ..lib import *
from ..io import S3FileSystem


def unpack_contours(df_filename: str, threshold: float = 0.6) -> gpd.GeoDataFrame:
    """ Convert multipolygon contour row above given threshold into multiple Polygon rows. """
    df = gpd.read_file(df_filename)
    if len(df) <= 2:
        query = df[df.amax > threshold] 
        if len(query):
            return gpd.GeoDataFrame(geometry=list(query.iloc[0].geometry.geoms), crs=df.crs)
        else:
            return gpd.GeoDataFrame(geometry=[], crs=df.crs)
    raise ValueError(
        f"gdal_contour dataframe {df_filename} has {len(df)} contours, "
        f"but should have maximal 2 entries (one below and/or one above threshold)!")


# Split it into smaller patchlets (in parallel)
def patchify_segmentation_data(eop_path: str, outdir: str, patchlet_size:tuple=(1128,1128), buffer:int=100, n_jobs:int=2):
    eop_paths = [eop_path]
    multiprocess_map(func=split_patch_into_patchlets, object_list=eop_paths, n_jobs=n_jobs, 
                 patchlet_size=patchlet_size, buffer=buffer, output_dir=outdir)


def load_model(model_folder:str, tile_shape:tuple = (None,1128,1128,4)):
    """
    Load a segmentation model from a folder.
    """
    filesystem = get_filesystem(model_folder)

    # Load model config
    with filesystem.open(os.path.join(model_folder, "model_cfg.json"), "r") as f:
        model_cfg = json.load(f)

    input_shape = dict(features=tile_shape)

    # Load, build and compile model
    model = ResUnetA(model_cfg)
    model.build(input_shape)
    model.net.compile()

    checkpoint_dir = os.path.join(model_folder, "checkpoints")

    # If filesystem of model folder is s3, download weights into temporary folder
    if type(filesystem) == S3FileSystem:
        print("Downloading model weights from S3")
        filesystem.download(checkpoint_dir, "/tmp", recursive=True)
        checkpoint_dir = "/tmp/checkpoints"

    model.net.load_weights(os.path.join(checkpoint_dir, "model.ckpt"))

    return model


def get_patchlet_shape(patchlet_path: str):
    eop = EOPatch.load(patchlet_path, lazy_loading=True)
    return eop.data['BANDS'].shape


def upscale_prediction(boundary: np.ndarray, scale_factor:int=2, disk_size:int=2):
    boundary = upscale_and_rescale(boundary, scale_factor=scale_factor)
    boundary = smooth(boundary, disk_size=disk_size)
    boundary = upscale_and_rescale(boundary, scale_factor=scale_factor)
    array = smooth(boundary, disk_size=disk_size * scale_factor)
    return np.expand_dims(array, axis=-1)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def segment_eopatch(eop_path: str, model: ResUnetA):
    # Load tile data
    eop = EOPatch.load(eop_path, lazy_loading=True)

    data = eop.data["BANDS"]
    timestamps = np.array(eop.timestamp)

    # Skip already segmented patches
    if "SEGMENTATION" in eop.data_timeless.keys():
        print(f"Skipping {eop_path} as it is already segmented")
        return

    # Iterate over inference timestamps
    for timestamp, bands in zip(timestamps, data):
        mean_stats = np.mean(bands, axis=(0,1))
        std_stats = np.std(bands, axis=(0,1))
        
        # Normalize bands
        norm_bands = (bands - mean_stats) / std_stats

        # Segment image
        extent, boundary, distance = model.net.predict(norm_bands[np.newaxis, ...], batch_size=1)

        # Crop to original size
        extent = crop_array(extent, 12)[..., :1]
        boundary = crop_array(boundary, 12)[..., :1]
        distance = crop_array(distance, 12)[..., :1]

        # Add to eopatch (initialize features first if necessary)
        if "BOUNDARY" not in eop.data.keys():
            eop.data['BOUNDARY'] = boundary
        else:
            eop.data['BOUNDARY'] = np.concatenate([eop.data['BOUNDARY'], boundary], axis=0)

    # Combine all boundary predictions into one segmentation map
    boundary = eop.data['BOUNDARY']
    boundary_combined = combine_temporal(boundary).squeeze()

    # Upscale and smooth segmentation map
    boundary_combined = upscale_prediction(boundary_combined, scale_factor=2, disk_size=2)

    # Set as segmentation map
    eop.data_timeless['SEGMENTATION'] = boundary_combined

    # Remove boundary predictions
    eop.remove_feature(FeatureType.DATA, 'BOUNDARY')

    # Save eopatch
    eop.save(eop_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)


def segment_patchlets(modeldir:str, patchlet_dir:str):
    # Load model
    print("Loading model")
    model = load_model(modeldir)

    # Segment each patchlet
    patchlet_paths = glob.glob(os.path.join(patchlet_dir, "*"))
    for i,patchlet_path in enumerate(patchlet_paths):
        print(f"{(i+1)}/{len(patchlet_paths)}: Segmenting {patchlet_path}")
        segment_eopatch(patchlet_path, model)


def prediction_to_tiff(eop_path:str, outdir:str):
    path_elements = eop_path.split("/")
    patchlet_name = path_elements[-1]
    tile_name = path_elements[-2]

    eop = EOPatch.load(eop_path, lazy_loading=True)

    tiff_dir = os.path.join(outdir, "tiffs")
    os.makedirs(tiff_dir, exist_ok=True)

    tiff_path = os.path.join(tiff_dir, patchlet_name + ".tiff")

    # Save to tiff
    export_eopatch_to_tiff(eop_path, out_path=tiff_path,feature=(FeatureType.DATA_TIMELESS, 'SEGMENTATION'))

    return tiff_path


def vectorize(eop_path:str, outdir:str, threshold:float=0.51):
    """Contour single eopatch"""

    eop_name = os.path.basename(eop_path)
    vec_path = os.path.join(outdir, eop_name + '.gpkg')

    # Skip if already vectorized
    if os.path.exists(vec_path):
        print(f"Skipping {eop_name} as it is already vectorized")
        return
    

    # Temporarily save as tiff
    tiff_path = prediction_to_tiff(eop_path, outdir)

    # Vectorize tiff file using gdal
    gdal_str = f"gdal_contour -of gpkg {tiff_path} {vec_path} -i {threshold} -amin amin -amax amax -p > /dev/null"
    os.system(gdal_str)

    # Unpack contours from tiff file
    df = unpack_contours(vec_path, threshold)

    # Remove temporary file
    os.remove(vec_path)

    # Save as geopackage
    df.to_file(vec_path, driver='GPKG', mode='w')

    # Remove tiff file
    os.remove(tiff_path)

    return df


def vectorize_patchlets(patchlet_dir:str, outdir:str, n_jobs:int=8):
    patchlet_paths = glob.glob(os.path.join(patchlet_dir, "*"))

    multiprocess_map(func=vectorize, object_list=patchlet_paths, n_jobs=n_jobs, outdir=outdir)

    # Delete empty tiff directory
    tiff_dir = os.path.join(outdir, "tiffs")
    if os.path.exists(tiff_dir):
        os.rmdir(tiff_dir)


def combine_shapes(s1:BaseGeometry,s2:BaseGeometry):
    """ Combine two lists of shapes"""
    combined_list = s1 + s2

    if len(combined_list) <= 1:
        return combined_list
    else:
        return list(unary_union(combined_list).geoms)


def combine_shapes_recursive(shapes:list, left:list, right:list, crs:CRS=CRS('32630')):
    """Recursively combine shapes (divide and conquer) and save intermediate results"""

    print("Combining shapes progress: {:.2f}%".format(left / len(shapes) * 100), end="\r")

    out = []
    if left == right:
        out = shapes[left]
    elif left + 1 == right:
        out = combine_shapes(shapes[left], shapes[right])
    else:
        mid = (left + right) // 2
        shapes1 = combine_shapes_recursive(shapes, left, mid, crs)
        shapes2 = combine_shapes_recursive(shapes, mid + 1, right, crs)
        out = combine_shapes(shapes1, shapes2)

    return out


def combine_patchlet_shapes(contours_dir:str, outpath:str, crs:CRS=CRS('32630'), min_area:int = 0, max_area:int = 500000):
    # Get all patchlet vector files
    vec_paths = glob.glob(os.path.join(contours_dir, "*.gpkg"))

    # Read all shapes
    shapes = [gpd.read_file(vec_path).geometry.tolist() for vec_path in vec_paths]

    if len(shapes) == 0:
        raise ValueError("No shapes found")

    # Combine shapes recursively
    combined_shapes = combine_shapes_recursive(shapes, 0, len(shapes) - 1, crs=crs)

    # Filter by area
    print("Filtering final shapefile by area", end="\r")
    df = gpd.GeoDataFrame(geometry=combined_shapes, crs=crs)
    df = df[df.area > min_area]
    df = df[df.area < max_area]

    # Save df in temporary location
    print("Saving final result", end="\r")
    temp_path = os.path.join("/tmp", os.path.basename(outpath))
    df.to_file(temp_path, mode='w')

    # Move to final location
    print("Moving to final location", end="\r")
    filesystem = get_filesystem(outpath)
    filesystem.move(temp_path, outpath, overwrite=True)


# def combine_npys(datadir: str, 
#                      eopatch_name:str, 
#                      dates: list = None, 
#                      feature_name:str = 'RGB', 
#                      bands:list = ['B2', 'B3', 'B4', 'B8A'], 
#                      derive_mask:bool = False, 
#                      delete_after:bool = False,
#                      partition_size:int = 10):
#     dateformat = "%Y_%m_%d"

#     # Get all the dates that have info for all bands
#     if dates is None:
#         dates = []
#         for band in bands:
#             band_dates = set()
#             banddir = os.path.join(datadir, band)
#             files = glob.glob(os.path.join(banddir, "*.npy"))
#             # Add dates to list for files with correct format
#             for file in files:
#                 try:
#                     band_dates.add(dt.datetime.strptime(os.path.basename(file).replace(".npy",""), dateformat))
#                 except ValueError:
#                     pass
#             dates = set.intersection(dates, band_dates) if len(dates) > 0 else band_dates
    
#     dates = sorted(list(dates))
    
#     if len(dates) == 0:
#         raise ValueError("No dates found with data for all bands")

#     # Check if we have all the band values for these dates
#     for date in dates:
#         for band in bands:
#             if not os.path.exists(os.path.join(datadir, band, date.strftime(dateformat) + ".npy")):
#                 raise FileNotFoundError(f"Band {band} for date {date} missing in {datadir}")
            
#     # Partition dates into groups of partition_size
#     date_partitions = [dates[i:i + partition_size] for i in range(0, len(dates), partition_size)]
#     print(f"Processing {len(date_partitions)} partitions of {partition_size} dates each")

#     # Create parent directory
#     parent_dir = os.path.join(datadir, eopatch_name)
#     os.makedirs(parent_dir, exist_ok=True)

#     # Process each partition
#     for i, partition in enumerate(date_partitions):
#         print(f"Processing partition {i+1}/{len(date_partitions)}", end="\r")

#         # Combine the bands into one array
#         DATA = None # SHAPE: (n_dates, img_h, img_w, n_bands)
#         MASK = None # SHAPE: (n_dates, img_h, img_w)

#         for date in partition:
#             # Create a local array
#             band_array = [np.load(os.path.join(datadir, band, date.strftime(dateformat) + ".npy")) for band in bands]

#             # Stack the bands
#             data = np.stack(band_array, axis=-1)[np.newaxis, ...]

#             # Add to global array
#             if DATA is None:
#                 DATA = data
#             else:
#                 DATA = np.concatenate([DATA, data], axis=0)

#             # Derive mask
#             if derive_mask:
#                 mask = data[..., -1] > 0
#                 mask = mask.astype(np.uint8)[..., np.newaxis]
#                 if MASK is None:
#                     MASK = mask
#                 else:
#                     MASK = np.concatenate([MASK, mask], axis=0)

#         # Get bboxes from band folders
#         bboxes = []
#         for band in bands:
#             bbox_path = os.path.join(datadir, band, 'bbox.txt')
#             bboxes.append(bbox_from_file(bbox_path))

#         # Check if all bboxes are the same
#         bbox = bboxes[0]
#         for b in bboxes:
#             if b != bbox:
#                 raise ValueError("Bounding boxes for each band are not the same")
        
#         # Create eopatch
#         eopatch = EOPatch()
#         eopatch.data[feature_name] = DATA
#         eopatch.bbox = bbox
#         eopatch.timestamp = partition

#         # Add mask if necessary
#         if derive_mask:
#             eopatch.mask['IS_DATA'] = MASK

#         # Save eopatch
#         print(f"Saving eopatch {i+1}/{len(date_partitions)}", end="\r")
#         begin_date = partition[0].strftime(dateformat)
#         eopatch.save(os.path.join(parent_dir, f"{eopatch_name}_{begin_date}"), overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

#     # (Optional) Delete all the individual files
#     if delete_after:
#         for date in dates:
#             for band in bands:
#                 os.remove(os.path.join(datadir, band, date.strftime(dateformat) + ".npy"))
#             os.remove(os.path.join(datadir, "bbox_" + date.strftime(dateformat) + ".txt"))

#     return eopatch



def combine_rgb_npys_into_eopatch(bands_data_package:BandsDataPackage, outdir:str, dates:list = None) -> str:
    DATE_FORMAT = "%Y_%m_%d"

    if dates is None:
        dates = [dt.datetime.strptime(os.path.basename(pair[0]).replace(".RAS",""), DATE_FORMAT) for pair in bands_data_package.B2_package.path_pairs]

    if len(dates) > 1000:
        raise ValueError("Too many dates to segment")

    bands = bands_data_package.tolist()

    # Check if all bands have the necessary segment dates
    for band_data_package in bands:
        for segment_date in dates:
            if not os.path.exists(os.path.join(band_data_package.BAND_PATH, segment_date.strftime(f"{DATE_FORMAT}.npy"))):
                raise ValueError(f"Date {segment_date} not found in {band_data_package.BAND_PATH}")

    # Combine the segment dates into one eopatch per band
    print("Combining segment dates into one eopatch per band")
    eop_paths = []
    for band_data_package in bands:
        bbox = load_bbox(os.path.join(band_data_package.BAND_PATH, "bbox.pkl"))

        outpath = os.path.join(outdir, band_data_package.BAND_NAME + "_eopatch")
        eop_paths += [outpath]

        npy_paths = [os.path.join(band_data_package.BAND_PATH, date.strftime(f"{DATE_FORMAT}.npy")) for date in dates]

        combine_npys_into_eopatches(
            npy_paths=npy_paths,
            outpath=outpath,
            feature_name='BANDS',
            bbox=bbox,
            dates=dates,
            delete_after=False,
            partition_size=1000
        )

    # Now combine the eopatches into one eopatch
    print("Combining eopatches into one eopatch")
    base_eopatch = None
    for eop_path in eop_paths:
        eop = EOPatch.load(eop_path)
        if base_eopatch is None:
            base_eopatch = eop
        else:
            base_eopatch.data['BANDS'] = np.concatenate((base_eopatch.data['BANDS'], eop.data['BANDS']), axis=-1)
        
    # Save the eopatch
    print("Saving eopatch")
    outpath = os.path.join(outdir, "segment_eopatch")
    base_eopatch.save(outpath, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

    return outpath