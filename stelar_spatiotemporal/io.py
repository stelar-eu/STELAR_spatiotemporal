from typing import SupportsInt, Text, List
from fs.osfs import OSFS
from s3fs.core import S3FileSystem
import os

class LocalFileSystem(OSFS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def glob(self, path: Text, **kwargs) -> List[str]:
        # Remove root path from path
        path = path.replace(self.root_path, '')

        # Glob the path and return full paths as strings
        return [self.root_path + x.path for x in super().glob(path, **kwargs)]
    
    def open(self, path: Text, mode: Text = 'r', **kwargs):
        # Remove root path from path
        path = path.replace(self.root_path, '')

        # Open the file
        return super().open(path, mode, **kwargs)
    
    # Function that moves files from one path to another
    def move(self, src: Text, dst: Text, overwrite: bool = False):
        if src.startswith("s3://") or dst.startswith("s3://"):
            raise ValueError("Cannot move files between local and S3 filesystems using a LocalFileSystem")

        # Remove root path from paths
        src = src.replace(self.root_path, '')
        dst = dst.replace(self.root_path, '')

        # Move the file
        super().move(src, dst, overwrite)

class S3FileSystem(S3FileSystem):
    def __init__(self, key=None, secret=None, endpoint_url=None, *args, **kwargs):
        client_kwargs = kwargs.get("client_kwargs", {})
        client_kwargs.update({"endpoint_url": endpoint_url})
        super().__init__(key=key, secret=secret, client_kwargs=client_kwargs, *args, **kwargs)

    def glob(self, path: Text, **kwargs) -> List[str]:
        return ["s3://" + file for file in super().glob(path, **kwargs)]
    
    # Function that moves, downloads or uploads files from one path to another
    def move(self, src: Text, dst: Text, overwrite: bool = False):
        if src.startswith("s3://") and dst.startswith("s3://"):
            # Move from S3 to S3
            super().mv(src, dst, recursive=True)
        elif src.startswith("s3://") and not dst.startswith("s3://"):
            # Move from S3 to local
            super().get(src, dst)
            super().rm(src, recursive=True)
        elif not src.startswith("s3://") and dst.startswith("s3://"):
            # Move from local to S3
            super().put(src, dst)
            os.remove(src)
        else:
            # Move from local to local
            os.rename(src, dst)