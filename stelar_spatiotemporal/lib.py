from decorator import decorator
import inspect
import typing
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from typing import Callable
import skimage.filters.rank as rank
from skimage.morphology import disk
import rasterio
import os
import pandas as pd
from sentinelhub import BBox, CRS
import pickle
from rasterio.io import MemoryFile
import re
import datetime as dt

from .io import LocalFileSystem, S3FileSystem
from .eolearn.core import EOPatch

@decorator
def check_types(func, *args, **kwargs):
    """
    Decorator that automatically checks the types of the arguments passed to a function
    """
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    for name, value in bound.arguments.items():
        expected_type = sig.parameters[name].annotation
        # Ignore if value is empty
        if value is None:
            continue
        if issubclass(expected_type.__class__, typing._SpecialGenericAlias):
            continue
        if not isinstance(value, expected_type):
            raise TypeError(f"Expected type {expected_type.__name__} for argument '{name}', but got type {type(value).__name__} instead")
    return func(*args, **kwargs)


def df_to_csv_manual(df: pd.DataFrame, outpath:str, index:bool=True, header:bool=True, mode:str='w'):
    """Write a pandas DataFrame to a CSV file manually"""
    filesystem = get_filesystem(outpath)

    with filesystem.open(outpath, mode) as f:
        if header:
            if index:
                index_name = df.index.name if df.index.name is not None else 'index'
                f.write(index_name + ',')
            f.write(','.join(df.columns.astype(str)) + '\n') # Make sure columns are strings
        for i in range(len(df)):
            if index:
                f.write(str(df.index[i]) + ',')
            f.write(','.join([str(x) for x in df.iloc[i]]) + '\n')


def fetch_image(url: str) -> np.ndarray:
    """Fetch image through an API request"""

    # Download the TIF file
    response = requests.get(url)
    
    # Open the TIF file as a PIL image
    img = Image.open(BytesIO(response.content))

    # Convert the PIL image to a NumPy array
    arr = np.array(img)
    
    return arr

def open_rasterio(path:str):
    """Open a rasterio dataset"""
    if path.startswith("s3://"):
        endpoint = os.environ.get("MINIO_ENDPOINT_URL").replace("http://", "").replace("https://", "")
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='YES', AWS_VIRTUAL_HOSTING=False, AWS_S3_ENDPOINT=endpoint):
            with rasterio.open(path) as src:
                return src.read(), src.profile
    else:
        with rasterio.open(path) as src:
            return src.read(), src.profile
    
def get_rasterio_bbox(profile:dict):
    bbox = rasterio.transform.array_bounds(profile['height'], profile["width"], profile["transform"])
    crs = CRS(profile["crs"].to_string())
    return BBox(bbox, crs)

def get_rasterio_timestamps(profile:dict, path:str):
    # See if there are timestamps in the metadata in format YYYYMMDD
    timestamps = None
    if 'tags' in profile.keys():
        if "TIFFTAG_DATETIME" in profile['tags']:
            timestamps = [profile['tags']["TIFFTAG_DATETIME"]]

    if timestamps is None and profile["count"] == 1:
        # Check if there is a timestamp in the filename in the format YYYYMMDD
        filename = os.path.basename(path)
        timestamps = re.search(r"\d{8}", filename)
        if timestamps:
            timestamps = [timestamps.group()]

    if not timestamps:
        raise ValueError("No timestamps found in the metadata or the filename.")

    # Parse the timestamps
    return [dt.datetime.strptime(timestamp, "%Y%m%d") for timestamp in timestamps]

def multiprocess_map(func: Callable, object_list: list, n_jobs:int = 4, **kwargs: dict):
    """
    Do a parallel map over a list of objects using multiprocessing
    func: function to apply to each object
    object_list: list of objects to apply the function to
    kwargs: keyword arguments to pass to the function
    n_jobs: number of processes to use
    """
    partial_func = partial(func, **kwargs)
    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap_unordered(partial_func, object_list), total=len(object_list)))
    return results


def crop_array(array: np.ndarray, buffer:int):
    """ Crop height and width of a 4D array given a buffer size. Array has shape B x H x W x C """
    assert array.ndim == 4, 'Input array of wrong dimension, needs to be 4D B x H x W x C'

    return array[:, buffer:-buffer:, buffer:-buffer:, :]


def pad_array(array: np.ndarray, buffer: int) -> np.ndarray:
    """ Pad height and width dimensions of a 4D array with a given buffer. Height and with are in 2nd and 3rd dim """
    assert array.ndim == 4, 'Input array of wrong dimension, needs to be 4D B x H x W x C'

    return np.pad(array, [(0, 0), (buffer, buffer), (buffer, buffer), (0, 0)], mode='edge')


def combine_temporal(array: np.ndarray) -> np.ndarray:
    """ Temporally merge predictions within a given time interval using the median"""
    return np.nan_to_num(np.nanmedian(array, axis=0))


def smooth(array: np.ndarray, disk_size: int = 2) -> np.ndarray:
    """ Blur input array using a disk element of a given disk size """
    assert array.ndim == 2

    smoothed = rank.mean(array, disk(disk_size)).astype(np.float32)
    smoothed = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))

    # assert np.sum(~np.isfinite(smoothed)) == 0

    return smoothed


def upscale_and_rescale(array: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    """ Upscale a given array by a given scale factor using bicubic interpolation """
    assert array.ndim == 2

    height, width = array.shape

    rescaled = np.array(Image.fromarray(array).resize((width * scale_factor, height * scale_factor), Image.BICUBIC))
    rescaled = (rescaled - np.min(rescaled)) / (np.max(rescaled) - np.min(rescaled))

    # assert np.sum(~np.isfinite(rescaled)) == 0

    return rescaled


def export_eopatch_to_tiff(eop_path:str, out_path:str, feature:tuple, nodata:int=0, channel_pos:int=0):
    eopatch = EOPatch.load(eop_path, lazy_loading=True)

    image_array = eopatch[feature].squeeze()

    if len(image_array.shape) > 3:
        raise ValueError(f"Feature {feature} has more than 3 dimensions, make sure to select either a single timestamp or a single band")
    
    if len(image_array.shape) == 2:
        image_array = image_array[np.newaxis, ...]
        channel_pos = 0

    # Get image dimensions
    img_shape = list(image_array.shape)
    channel_count = img_shape[channel_pos]

    # Width and height are the other two dimensions
    img_shape.pop(channel_pos)
    width, height = img_shape

    # Make sure channels are first
    image_array = np.moveaxis(image_array, channel_pos, 0)

    dst_crs = eopatch.bbox.crs.ogc_string()
    dst_transform = rasterio.transform.from_bounds(*eopatch.bbox, width=width, height=height)

    with rasterio.open(out_path, 'w', driver='GTiff',
                        width=width, height=height,
                        count=channel_count,
                        dtype=image_array.dtype, nodata=nodata,
                        transform=dst_transform, crs=dst_crs,
                        compress='DEFLATE') as dst:
        dst.write(image_array)


def export_array_to_tiff(arr:np.ndarray, bbox:BBox, out_path:str, nodata:int=0, channel_pos:int=-1):
    if len(arr.shape) > 3:
        raise ValueError(f"Array has more than 3 dimensions, make sure to select either a single timestamp or a single band")
    if len(arr.shape) == 2:
        arr = arr[..., np.newaxis]

    # Get image dimensions
    img_shape = list(arr.shape)
    channel_count = img_shape[channel_pos]

    # Width and height are the other two dimensions
    img_shape.pop(channel_pos)
    width, height = img_shape

    # Make sure channels are first
    arr = np.moveaxis(arr, channel_pos, 0)

    dst_crs = bbox.crs.ogc_string()
    dst_transform = rasterio.transform.from_bounds(*bbox, width=width, height=height)

    with rasterio.open(out_path, 'w', driver='GTiff',
                        width=width, height=height,
                        count=channel_count,
                        dtype=arr.dtype, nodata=nodata,
                        transform=dst_transform, crs=dst_crs,
                        compress='DEFLATE') as dst:
        dst.write(arr)


def load_bbox(bbox_path:str):
    """Load a sentinelhub BBox from a pickle file"""
    if not bbox_path.endswith('.pkl'): raise ValueError('bbox_path must be a pickle file')
    if not os.path.exists(bbox_path): raise ValueError('bbox_path does not exist')

    try:
        with open(bbox_path, 'rb') as f:
            bbox = pickle.load(f)
    except:
        raise ValueError('Could not load bbox from file')
    
    return bbox


def save_bbox(bbox:BBox, path:str):
    with open(path, 'wb') as f:
        pickle.dump(bbox, f)


def get_filesystem(path:str):
    if path.startswith('s3://'):
        return get_s3_filesystem(path)
    else:
        return get_local_filesystem(path)
    

def get_s3_filesystem(path:str):
    if not path.startswith('s3://'):
        raise ValueError(f"AWS path has to start with s3:// but found '{path}'")

    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
    if not MINIO_ACCESS_KEY:
        raise ValueError("MINIO_ACCESS_KEY not set")
    
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
    if not MINIO_SECRET_KEY:
        raise ValueError("MINIO_SECRET_KEY not set")
    
    MINIO_ENDPOINT_URL = os.environ.get("MINIO_ENDPOINT_URL")
    if not MINIO_ENDPOINT_URL:
        raise ValueError("MINIO_ENDPOINT_URL not set")

    return S3FileSystem(key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY, endpoint_url=MINIO_ENDPOINT_URL)
        

def get_local_filesystem(path:str):
    # If path is a file, return the directory
    if os.path.isfile(path) or '.' in os.path.basename(path):
        path = os.path.dirname(path)
    return LocalFileSystem(path)