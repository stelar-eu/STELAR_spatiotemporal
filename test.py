from stelar_spatiotemporal.lib import get_filesystem
from stelar_spatiotemporal.preprocessing.preprocessing import split_array_into_patchlets
from stelar_spatiotemporal.preprocessing.vista_preprocessing import unpack_ras, get_rhd_info
import os
import glob
import numpy as np

if __name__ == "__main__":
    ras_path = "s3://stelar-spatiotemporal/RGB_small/B2/30TYQ_B2_2020.RAS"
    rhd_path = "s3://stelar-spatiotemporal/RGB_small/B2/30TYQ_B2_2020.RHD"

    os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
    os.environ["MINIO_SECRET_KEY"] = "minioadmin"
    os.environ["MINIO_ENDPOINT_URL"] = 'http://localhost:9000'

    img_h, img_w, timestamps, bbox = get_rhd_info(rhd_path)

    # Unpack
    # unpack_ras(ras_path, "/tmp", timestamps, img_w, img_h)

    files = glob.glob("/tmp/*.npy")

    # Split into patchlets
    patchlets = split_array_into_patchlets(files[0], patchlet_size=(1128,1128), buffer=0)



    

