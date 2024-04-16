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

# Define function
def reproj2base(inpath, basepath, outpath, bands=None, resampling_method='nearest', plot=False):
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
    # ToDo: Add reading RAS Files instead of TIF Files (use existing code)



    # open input file
    with rasterio.open(inpath, 'r') as src:
        #src = rasterio.open(inpath, 'r')
        src_transform = src.transform
        src_nodata = src.nodata
        src_height = src.height
        src_width = src.width
        with rasterio.open(basepath, 'r') as base:
            dst_crs = base.crs
            dst_count = len(bands)
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
        #print("Original shape:", src_height, src_width, '\n Affine', src_transform)
        #print("Coregistered to shape:", dst_height, dst_width, '\n Affine', dst_transform)
        # open output
        with rasterio.open(outpath, 'w', **dst_kwargs) as dst:
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
            # specify amount of bands
            if bands:
                destination = np.zeros((dst_height,dst_width))
                for i, iband in enumerate(bands):
                    reproject(
                        source=rasterio.band(src, iband),
                        destination=destination, #rasterio.band(dst, i+1), # destination
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling)
                    dst.write(destination, 1)
            else:
                # iterate through all bands and write using reproject function
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling)
    # open input file to plot data
    if plot:
        with rasterio.open(outpath, 'r') as out:
            data = out.read()
        # plot resampled data
        # plot destination
        plt.figure(figsize=(8, 8))
        cmap = matplotlib.colors.ListedColormap(['red', 'green'])
        plt.imshow(data[0, :, :], cmap=cmap)
        plt.colorbar()
        plt.title("DST DATA")
        plt.show()
