from stelar_spatiotemporal.preprocessing.timeseries import lai_to_csv_field
import os
import glob
import numpy as np

if __name__ == "__main__":
    eop_dir = "/tmp/lai_eopatch"
    field_path = "s3://stelar-spatiotemporal/fields_2020_07_27.gpkg"

    os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
    os.environ["MINIO_SECRET_KEY"] = "minioadmin"
    os.environ["MINIO_ENDPOINT_URL"] = 'http://localhost:9000'

    lai_to_csv_field([eop_dir], fields_path=field_path, outdir="/tmp", n_jobs=1)