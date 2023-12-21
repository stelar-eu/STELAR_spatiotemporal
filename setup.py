from setuptools import setup, find_packages
import codecs
import os

DESCRIPTION = 'Spatiotemporal data processing for the STELAR project'
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    LONG_DESCRIPTION = "\\n" + fh.read()

setup(
    name='stelar_spatiotemporal',
    version='{{VERSION_PLACEHOLDER}}',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Jens d'Hondt",
    author_email="j.e.d.hondt@tue.nl",
    packages=find_packages(),
    install_requires=[
        "pystac",
        "pystac_client",
        "shapely",
        "opencv-python",
        "numpy",
        "sentinelhub",
        "pandas",
        "rasterio",
        "geopandas>=0.8.1",
        "decorator",
        "scikit-image",
        "fs",
        "fs-s3fs",
        "s3fs",
        "tqdm",
        "boto3",
        "matplotlib",
        "fiona"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
