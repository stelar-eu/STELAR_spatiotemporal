from stelar_spatiotemporal.preprocessing.preprocessing import unpack_tif
from stelar_spatiotemporal.lib import *
from stelar_spatiotemporal.data_fusion.coregistration import reproj2base
import os
import glob
import numpy as np


if __name__ == "__main__":
    
    os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
    os.environ["MINIO_SECRET_KEY"] = "minioadmin"
    os.environ["MINIO_ENDPOINT_URL"] = 'http://localhost:9000'

    tiffdir = "s3://stelar-spatiotemporal/LAI_fused"
    # ref_path = "s3://stelar-spatiotemporal/S2_30TYQMTI_200106_IC_DEMO.tif"
    outdir = "/tmp/lai"
    # os.makedirs(outdir, exist_ok=True)

    # fs = get_filesystem(tiffdir)

    # tiff_files = fs.glob(tiffdir + "/*.TIF")

    # for tiff_file in tiff_files:
    #     outpath = os.path.join(outdir, os.path.basename(tiff_file))
    #     reproj2base(tiff_file, ref_path, outpath)
        
    # print("Done")

    unpack_tif(tiffdir, outdir, extension="TIF")