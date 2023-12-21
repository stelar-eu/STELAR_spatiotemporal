from ..lib import fetch_image, check_types
import datetime as dt
import pystac
from pystac_client import Client
from shapely import Geometry
from shapely.geometry import shape
import json
from typing import List, Tuple, Union
import cv2
import numpy as np
import os
import shutil
from sentinelhub import CRS, BBox
from ..eolearn.core import EOPatch, OverwritePermission, FeatureType
from ..eopatch_functions import data_to_eopatch


def fetch_s2_images(startdate: dt.datetime, enddate: dt.datetime, 
                    aoi: Geometry, max_cloud_cover:int = 100, limit: int = 100) -> pystac.ItemCollection:
    """
    Fetch Sentinel-2 images from the AWS Open Data Registry
    startdate: start date of the search
    enddate: end date of the search
    aoi: area of interest
    max_cloud_cover: maximum cloud cover
    limit: maximum number of images to return
    """

    # Format query parameters
    datefmt = "%Y-%m-%d"

    startdate = startdate.strftime(datefmt)
    enddate = enddate.strftime(datefmt)

    daterange = startdate + "/" + enddate
    query = f"eo:cloud_cover<={max_cloud_cover}"

    api_url = "https://earth-search.aws.element84.com/v0"
    collection = "sentinel-s2-l2a-cogs"  # Sentinel-2, Level 2A, COGs

    #     Open client
    client = Client.open(api_url)

    #     Search images
    search = client.search(
        collections=[collection],
        intersects = aoi,
        query=[query],
        datetime=daterange,
        max_items=limit
    )

    items = search.item_collection()

    return items


def json_to_shape(json_path: str) -> Geometry:
    """
    Transform a GeoJSON file to a Shapely geometry
    """

    with open(json_path) as f:
        geojson: dict = json.load(f)

    return shape(geojson["features"][0]["geometry"])


def download_bands(item: pystac.Item, bands: List[str]):
    """
    Download the bands of a Sentinel-2 image
    item: pystac.Item object containing all the available bands in the image
    bands: list of bands to download
    """
    # Get the bands
    assets = item.assets

    # Check if all bands are available
    data = []
    for band in bands:
        if band not in assets:
            raise ValueError(f"Band {band} is not available for this image")
        print("Downloading band", band)
        data.append(fetch_image(assets[band].href))
        
    return data

def upsample(arr: np.ndarray, ratio:int=2) -> np.ndarray:
    """
    Upsample an array by a given ratio
    """
    w,h = arr.shape
    return cv2.resize(arr, (ratio*w,ratio*h), interpolation=cv2.INTER_NEAREST)


def prepare_image(item: pystac.Item, eop_name: str, bands: list, outdir: str = "") -> None:
    """
    Prepare a single image (i.e. one timestamp) for use in STELAR.
    This image is then optionally combined with images from other timestamps.

    item: pystac item
    eop_name: name of the eopatch to save
    outdir: directory to save the eopatch to
    bands: list of bands to download
    """
    print("Processing item", item.id)

    # Download bands
    b02,b03,b04,b08,scl = download_bands(item, bands)

    # SCL to mask
    mask = (scl == 4).astype(np.uint8)

    # Upsample mask
    mask = upsample(mask, ratio=2)

    # Combine bands
    data = np.dstack([b02,b03,b04,b08,mask])

    # Transform to eopatch and save temporarily
    data_to_eopatch(eop_name, data, item, outdir)

def prepare_tile(tile_id: str, items: List[pystac.Item], aoi_name: str, bands: list = ["B02", "B03", "B04", "B08", "SCL"], outdir: str = "") -> None:
    """
    Prepare a single tile for use in STELAR.

    tile_id: id of the tile
    items: list of pystac items for this tile, one per timestamp (all with the same bbox)
    aoi_name: name of the area of interest, used for naming the eopatches
    bands: list of bands to download for each timestamp
    outdir: directory to save the eopatches to
    """

    print("Processing tile", tile_id)
    eop_name = "_".join([aoi_name, tile_id])

    assert "SCL" in bands, "SCL band is required"

    # Prepare each image and combine (both done in prepare_image)
    bbox1 = items[0].bbox
    for item in items:
        # assert item.bbox == bbox1, "All items should have the same bbox"
        prepare_image(item, eop_name, bands, outdir)


def split_patch_into_patchlets(eop_path: str, output_dir: str, patchlet_size: tuple = (1128,1128), buffer: int = 100):
    """
    Splits a large tile into patchlets of fixed width and height with a fixed overlap between them on all sides, 
    and saves each patchlet as an eopatch

    Parameters:
        eop_path (str): The path to the (full) eopatch to split.
        output_dir (str): The directory to save the patchlets to.
        patchlet_size (tuple): A tuple (width, height) specifying the size of each patchlet.
        buffer (int): The number of pixels by which each patchlet should overlap on all sides.
    """
    
    full_eop = EOPatch.load(eop_path, lazy_loading=True)

    # Temporarily combine all data and masks, save the structure of the image to reconstruct it later
    feature_structure: List[Tuple[int, Tuple(FeatureType, str)]] = []
    data_features = full_eop.data.keys()
    image = None
    for feature in data_features:
        if image is None:
            image = full_eop.data[feature]
        else:
            image = np.concatenate([image, full_eop.data[feature]], axis=-1)
        feature_structure.append((image.shape[-1], (FeatureType.DATA, feature)))

    mask_features = full_eop.mask.keys()
    for feature in mask_features:
        if image is None:
            image = full_eop.mask[feature]
        else:
            image = np.concatenate([image, full_eop.mask[feature]], axis=-1)
        feature_structure.append((image.shape[-1], (FeatureType.MASK, feature)))

    # Get patchlet dimensions
    ntimes, img_height, img_width, nbands = image.shape
    plet_width, plet_height = patchlet_size

    # Get coordinate ratios
    xmin, ymin, xmax, ymax = full_eop.bbox
    img_cwidth = xmax - xmin
    img_cheight = ymax - ymin
    px_width = img_cwidth / img_width
    px_height = img_cheight / img_height

    # Compute the number of patchlets in each dimension
    nplets_x = int(np.ceil(img_width / (patchlet_size[0] - 2*buffer)))
    nplets_y = int(np.ceil(img_height / (patchlet_size[1] - 2*buffer)))

    # Add padding to the image to make it evenly divisible into patchlets
    padded_width = plet_width * nplets_x
    padded_height = plet_height * nplets_y
    padded_image = np.zeros((ntimes, padded_height, padded_width, nbands), dtype=image.dtype)
    padded_image[:, buffer:buffer+img_height, buffer:buffer+img_width, :] = image

    # Extract each patchlet
    ystart = 0
    yend = buffer
    for y in range(nplets_y):
        ystart = yend - buffer
        yend = ystart + plet_height
        
        xstart = 0
        xend = buffer
        for x in range(nplets_x):
            xstart = xend - buffer
            xend = xstart + plet_width

            patchlet = padded_image[:, ystart:yend, xstart:xend, :]

            # Get bbox of patchlet
            plet_xmin = xmin + (xstart - buffer) * px_width
            plet_xmax = plet_xmin + plet_width * px_width
            plet_ymax = ymax - (ystart - buffer) * px_height
            plet_ymin = plet_ymax - plet_height * px_height
            plet_bbox = BBox([plet_xmin, plet_ymin, plet_xmax, plet_ymax], crs=full_eop.bbox.crs)

            # Reconstruct the eopatch structure
            new_eop = EOPatch()
            c = 0
            for n, (ftype, fname) in feature_structure:
                if ftype == FeatureType.DATA:
                    new_eop.data[fname] = patchlet[:, :, :, c:c+n]
                elif ftype == FeatureType.MASK:
                    new_eop.mask[fname] = patchlet[:, :, :, c:c+n]
                c += n

            new_eop.bbox = plet_bbox
            new_eop.timestamp = full_eop.timestamp

            # Save patchlet as eopatch
            new_eop.save(os.path.join(output_dir, f"patchlet_{x}_{y}"), overwrite_permission=OverwritePermission.OVERWRITE_PATCH)


def split_array_into_patchlets(npy_path: str, output_dir: str = None, patchlet_size: tuple = (1128,1128), buffer: int = 100,
                               feature_name: str = 'BANDS', bbox: BBox = None, timestamp: dt.datetime = None) -> List[EOPatch]:
    """
    Splits an array into patchlets of fixed width and height with a fixed overlap between them on all sides, 
    and saves each patchlet as an eopatch

    Parameters:
        npy_path (str): The path to the numpy array to split.
        output_dir (str): The directory to save the patchlets to.
        patchlet_size (tuple): A tuple (width, height) specifying the size of each patchlet.
        buffer (int): The number of pixels by which each patchlet should overlap on all sides.
        feature_name (str): The name of the feature to save the data to.
        bbox (BBox): The bounding box of the array.
        timestamp (datetime): The timestamp of the array.
    """
    
    # Extend the output directory with the name of the eopatch
    if output_dir is not None:
        output_dir = os.path.join(output_dir, os.path.basename(npy_path))
        os.makedirs(output_dir, exist_ok=True)

    arr = np.load(npy_path)

    # Expand image dimensions if necessary
    if len(arr.shape) == 2:
        arr = arr[np.newaxis, ..., np.newaxis]
    elif len(arr.shape) == 3:
        arr = arr[np.newaxis, ...]

    # Get image dimensions
    ntimes, img_height, img_width, nbands = arr.shape
    plet_width, plet_height = patchlet_size

    # Get coordinate ratios for bbox
    if bbox is None:
        bbox = BBox([0,0,img_width,img_height], crs=CRS.WGS84)
    
    xmin, ymin, xmax, ymax = bbox
    img_cwidth = xmax - xmin
    img_cheight = ymax - ymin
    px_width = img_cwidth / img_width
    px_height = img_cheight / img_height

    # Compute the number of patchlets in each dimension
    nplets_x = int(np.ceil(img_width / (patchlet_size[0] - 2*buffer)))
    nplets_y = int(np.ceil(img_height / (patchlet_size[1] - 2*buffer)))

    # Add padding to the image to make it evenly divisible into patchlets
    padded_width = plet_width * nplets_x
    padded_height = plet_height * nplets_y
    padded_image = np.zeros((ntimes, padded_height, padded_width, nbands), dtype=arr.dtype)
    padded_image[:, buffer:buffer+img_height, buffer:buffer+img_width, :] = arr

    # Extract each patchlet
    eopatches = []

    ystart = 0
    yend = buffer
    for y in range(nplets_y):
        ystart = yend - buffer
        yend = ystart + plet_height
        
        xstart = 0
        xend = buffer
        for x in range(nplets_x):
            xstart = xend - buffer
            xend = xstart + plet_width

            patchlet = padded_image[:, ystart:yend, xstart:xend, :]

            # Get bbox of patchlet
            plet_xmin = xmin + (xstart - buffer) * px_width
            plet_xmax = plet_xmin + plet_width * px_width
            plet_ymax = ymax - (ystart - buffer) * px_height
            plet_ymin = plet_ymax - plet_height * px_height
            plet_bbox = BBox([plet_xmin, plet_ymin, plet_xmax, plet_ymax], crs=CRS.WGS84)

            # Create the eopatch
            new_eop = EOPatch()
            new_eop.data[feature_name] = patchlet
            new_eop.bbox = plet_bbox

            if timestamp is not None:
                new_eop.timestamp = [timestamp]

            # Save patchlet as eopatch
            if output_dir is None:
                eopatches.append(new_eop)
            else:
                new_eop.save(os.path.join(output_dir, f"patchlet_{x}_{y}"), overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    return eopatches




def bbox_from_file(bbox_path: str):
    with open(bbox_path, 'r') as bboxfile:
        bbox = [float(i) for i in bboxfile.readline().split(" ") if i != ""]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        bbox = BBox([xmin,ymin,xmax,ymax], crs=CRS(32630))
    return bbox
            

def combine_npys_into_eopatches(npy_paths: list, 
                 outpath: str,
                 feature_name:str,
                 bbox: BBox,
                 dates:list = None,
                 delete_after:bool = False,
                 partition_size:int = 10):
    """
    Combine multiple numpy arrays into one eopatch by stacking them along the time axis.
    If dates are not given, infer the dates from the filenames.
    """
    dateformat = "%Y_%m_%d"

    # Get all the dates that have info for all bands
    if dates is None:
        dates = [dt.datetime.strptime(os.path.basename(file).replace(".npy",""), dateformat) for file in npy_paths]
    else: # Check if dates are valid
        if len(npy_paths) != len(dates):
            raise ValueError("Number of dates does not match number of files")
        
    # Process each partition
    if partition_size > len(dates): partition_size = len(dates)
    partitions = np.arange(0, len(dates), partition_size)
    print(f"Processing {len(partitions)} partitions of {partition_size} dates each")

    for i,start in enumerate(partitions):
        print(f"Processing partition {i+1}/{len(partitions)}", end="\r")

        end = min(start+partition_size, len(npy_paths))

        # Stack data
        arrays = [np.load(npy_paths[i]) for i in range(start, end)]
        part_data = np.stack(arrays, axis=0)
        part_dates = dates[start:end]
        
        # Create eopatch
        eopatch = EOPatch()
        eopatch.data[feature_name] = part_data[..., np.newaxis]
        eopatch.bbox = bbox
        eopatch.timestamp = part_dates

        # Save eopatch
        print(f"Saving eopatch {i+1}/{len(partitions)}", end="\r")
        part_outpath = outpath if len(partitions) == 1 else os.path.join(outpath, f"partition_{i+1}")
        eopatch.save(part_outpath, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    # (Optional) Delete all the individual files
    if delete_after:
        for file in npy_paths:
            os.remove(file)

# Compute the size of the partition, based on the size of one full image, the number of images, and the available RAM

def max_partition_size(example_path:str, MAX_RAM:int = 4 * 1e9):
    N_BYTES_PER_IMAGE = os.path.getsize(example_path)
    return int(MAX_RAM // N_BYTES_PER_IMAGE)


def combine_dates_for_eopatch(eop_name: str, eop_paths:list, outdir:str = None, delete_after:bool=False):
    """
    This function combines the multiple eopatches with different dates into one eopatch.
    eop_name: name of the eopatch to save
    eop_paths: list of paths to the eopatches to combine
    outdir: directory to save the eopatch to, defaults to the directory of the first eopatch
    delete_after: whether to delete the indicidual eopatches after combining them
    """
    if outdir is None:
        outdir = os.path.dirname(eop_paths[0])

    eopatch = None
    for eop_path in eop_paths:
        if not os.path.exists(eop_path):
            raise FileNotFoundError(f"EOPatch {eop_path} not found")

        # Load the patchlet and combine it with the previous patchlets
        patchlet = EOPatch.load(eop_path)
        if eopatch is None:
            eopatch = patchlet
        else:
            eopatch += patchlet

        # Delete the temporary patchlet if necessary
        if delete_after:
            shutil.rmtree(eop_path)
        
    
    # Save the combined patchlet
    eopatch.save(os.path.join(outdir, eop_name), overwrite_permission=OverwritePermission.OVERWRITE_PATCH)



