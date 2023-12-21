from typing import List, Tuple
import glob
import os
from ..lib import get_filesystem

class BandDataPackage:
    BAND_NAME: str = None
    BAND_PATH: str = None
    path_pairs: List[Tuple[str, str]] = None
    file_extension:str = None

    def __init__(self, band_name, band_path, file_extension:str = "RAS"):
        self.BAND_NAME = band_name
        self.BAND_PATH = band_path
        self.file_extension = file_extension

        self.check_file_extension()

        if file_extension == "RAS":
            self.check_ras_rhd_paths()

    """
    Check if each of the bands actually contains at least one file with the given file extension
    """
    def check_file_extension(self):
        filesystem = get_filesystem(self.BAND_PATH)
        if len(filesystem.glob(os.path.join(self.BAND_PATH, "**", f'*.{self.file_extension}'))) == 0:
            raise ValueError(f"No files with extension .{self.file_extension} found in {self.BAND_PATH}")

    """
    Check if for every RAS file there is a corresponding RHD file and 
    create a list of paths pairs
    """
    def check_ras_rhd_paths(self):
        filesystem = get_filesystem(self.BAND_PATH)

        self.path_pairs = []
        ras_paths = filesystem.glob(os.path.join(self.BAND_PATH, "**", '*.RAS'))
        ras_bases = [os.path.basename(path).replace('.RAS', '') for path in ras_paths]

        rhd_paths = filesystem.glob(os.path.join(self.BAND_PATH, "**", '*.RHD'))
        rhd_bases = [os.path.basename(path).replace('.RHD', '') for path in rhd_paths]
        
        for ras_id, ras_base in enumerate(ras_bases):
            rhd_base = ras_base.replace('.RAS', '.RHD')
            try:
                rhd_id = rhd_bases.index(rhd_base)
                self.path_pairs.append((ras_paths[ras_id], rhd_paths[rhd_id]))
            except ValueError:
                raise ValueError(f"RHD file for {ras_base} not found")

class BandsDataPackage:
    B2_package: BandDataPackage = None
    B3_package: BandDataPackage = None
    B4_package: BandDataPackage = None
    B8_package: BandDataPackage = None

    def __init__(self, b2_path, b3_path, b4_path, b8_path, file_extension:str = "RAS"):
        self.B2_package = BandDataPackage("B2", b2_path, file_extension)
        self.B3_package = BandDataPackage("B3", b3_path, file_extension)
        self.B4_package = BandDataPackage("B4", b4_path, file_extension)
        self.B8_package = BandDataPackage("B8A", b8_path, file_extension)

    def tolist(self):
        return [self.B2_package, self.B3_package, self.B4_package, self.B8_package]