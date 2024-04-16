"""
    Main function for a baseline data fusion of LAI values.
    Data Fusion for aligning ENMAP hyperspectral data and S2 multispectral data:
        (1) co-registration via gdal
        (2) spectral + spatial fusion to obtain S2 geometry with hyperspectral information

    @author: C. Jörges, VISTA GmbH
    Date: 11/23

"""

# load packages
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.io import MemoryFile
from ..lib import *

# Define function
def reproj2base(inpath, basepath, outpath, bands=[], resampling_method='nearest'):
    """
    Transform raster file to match the shape and projection of existing raster (co-registration).

    Inputs
    ----------
    inpath : (string) filepath of input raster file
    basepath : (string) path to raster with reference shape and projection
    outpath : (string) path to output raster file (tif)
    bands : (list) specified # band to reproject with default None=All bands
    resampling_method : (string) 'nearest', 'bilinear', 'cubic', 'average'
    plot : (bool) plot the data?
    """
    # TODO: Add reading RAS Files instead of TIF Files (use existing code)

    # open input file
    with open_rasterio(inpath) as src:
        src_nodata = src.nodata
        with open_rasterio(basepath) as base:
            dst_crs = base.crs
            dst_count = len(bands) if bands else src.count
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,  # input CRS
                dst_crs,  # output CRS
                base.width,  # base width
                base.height,  # base height
                *base.bounds)  # base outer boundaries (left, bottom, right, top)
                # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                        "transform": dst_transform,
                        "width": dst_width,
                        "height": dst_height,
                        "count": dst_count,
                        "nodata": src_nodata})
        
        # Temporarily save the reprojected data in memory
        TMPPATH = "/tmp/tmp.tif"

        with rasterio.open(TMPPATH, 'w', **dst_kwargs) as dst:
            #dst=rasterio.open(outpath, 'w', **dst_kwargs)
            # define resampling method
            if resampling_method == 'nearest':
                resampling = Resampling.nearest
            elif resampling_method == 'bilinear':
                resampling = Resampling.bilinear
            elif resampling_method == 'average':
                resampling = Resampling.average
            elif resampling_method == 'cubic':
                resampling = Resampling.cubic
            else: print("No Resampling method specified!")

            # reproject the data and write to the temporary file
            out = np.zeros((dst_count, dst_height, dst_width), dtype=src.read(1).dtype)
            out, dst_transform = reproject(
                source=src.read(),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling)
            
            if bands:
                for i, band in enumerate(bands):
                    dst.write(out[band-1], i+1)
            else:
                dst.write(out)
                    
    # copy the temporary file to the output path
    outfs = get_filesystem(outpath)
    outfs.move(TMPPATH, outpath, overwrite=True)
