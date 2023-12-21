from .lib import check_types
import datetime as dt
import numpy as np
from sentinelhub import CRS, BBox
import os
import pystac
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .eolearn.core import EOPatch, OverwritePermission, FeatureType


def init_eopatch(data: np.ndarray, timestamp: dt.datetime, bbox:BBox):
    """
    Initialize an EOPatch with data, timestamp and bbox
    data: data array (height, width, channels)
    """
    # Add time axis
    data = data[np.newaxis, ...]

    # Check that data has all necessary channels
    try:
        assert data.shape[-1] == 5
    except AssertionError:
        raise ValueError("Data must have 5 channels (4 bands + mask)")

    eop = EOPatch()
    eop.data["BANDS"] = data[..., :4]
    eop.mask["IS_DATA"] = data[..., 4:]
    eop['timestamp'] = [timestamp]
    eop['bbox'] = bbox
    return eop


def create_eopatch(data: np.ndarray, timestamps: list, bbox:BBox):
    """
    Initialize an EOPatch with data, timestamp and bbox
    data: data array (time, height, width, channels)
    timestamps: list of timestamps
    bbox: BBox object, corresponding to the bounding box of the image (i.e. the coordinates)
    """
    try:
        assert data.shape[-1] == 5
    except AssertionError:
        raise ValueError("Data must have 5 channels (4 bands + mask)")

    eop = EOPatch()
    eop.data["BANDS"] = data[..., :4]
    eop.mask["IS_DATA"] = data[..., 4:]
    eop['timestamp'] = timestamps
    eop['bbox'] = bbox
    return eop


def combine_eopatches(base_eop:EOPatch, add_eop: EOPatch):
    """
    Combine two EOPatches
    """
    # Check that bboxes are the same
    # assert base_eop.bbox == add_eop.bbox

    # Check that shapes are the same
    assert base_eop.data["BANDS"].shape[1:] == add_eop.data["BANDS"].shape[1:]

    # Combine data
    base_eop.data["BANDS"] = np.concatenate([base_eop.data["BANDS"], add_eop.data["BANDS"]], axis=0)
    base_eop.mask["IS_DATA"] = np.concatenate([base_eop.mask["IS_DATA"], add_eop.mask["IS_DATA"]], axis=0)
    base_eop.timestamp = base_eop.timestamp + add_eop.timestamp

    return base_eop




def data_to_eopatch(eop_name: str, data: np.ndarray, item: pystac.Item, outdir: str = ""):
    """
    Save image to eopatch (either create a new one or add to existing)
    eop_name: name of the eopatch to save
    data: data to save (should include bands and mask)
    item: pystac.Item object containing the metadata of the image (timestamp, bbox)
    """
    eop_dir = os.path.join(outdir, "eopatches")
    if not os.path.exists(eop_dir): os.makedirs(eop_dir)

    eop_path = os.path.join(eop_dir, eop_name)

    # Initialize eopatch
    eopatch = init_eopatch(data, item.datetime, BBox(item.bbox, crs=CRS.WGS84))

    # Add to existing if necessary
    if os.path.exists(eop_path):
        base_eop = EOPatch.load(eop_path, lazy_loading=True)
        eopatch = combine_eopatches(base_eop, eopatch)

    # Save eopatch
    print("Saving...")
    eopatch.save(eop_path, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

def plot_rgb(eopatch: EOPatch, timestamp_idx: int = 0, ax: Axes = None):
    """
    Plot the RGB bands of an eopatch
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(eopatch.data["BANDS"][timestamp_idx][..., [2, 1, 0]]/10000 * 3.5)


def compute_lai(eop_path:str) -> None:
    """
    Compute the Leaf Area Index (LAI) for an EOPatch,
    and store it in a new EOPatch feature named 'LAI'
    """
    # Load the EOPatch
    eopatch = EOPatch.load(eop_path, lazy_loading=True)

    # Load the SCL mask
    B,G,R,NIR = np.moveaxis(eopatch.data["BANDS"], -1, 0)
    
    # Compute the LAI
    EVI = 2.5 * ((NIR - R) / (NIR + 6 * R - 7.5 * B + 1))
    LAI = 3.618 * EVI - 0.118

    # Set the LAI to 0 for invalid pixels
    LAI[LAI == -np.inf] = 0
    LAI[LAI == np.inf] = 0
    LAI[LAI > 18] = 0
    LAI[LAI < 0] = 0
    
    # Add the LAI to the EOPatch
    eopatch.data["LAI"] = LAI[..., np.newaxis].astype(np.float16)

    # Save the EOPatch
    try:
        eopatch.save(eop_path, features=[(FeatureType.DATA, "LAI")])
    except ValueError:
        return