
import glob
import os
import re

import numpy as np


# Rotates across version numbers
# saves the version after the filename but before the file extension
class FilenameVersioner:
    """
    Container for saving versioned files
    """

    def __init__(self, filename_ext_pair, base_dir='.', max_versions=16):
        self.base_name, self.ext_name = filename_ext_pair
        self.max_versions = max_versions
        self.base_dir = base_dir

    @staticmethod
    def filename_regex(base_name, ext_name):
        return fr'({base_name})\.(.+)\.({ext_name})'

    @staticmethod
    # hacky and a bit unsafe, but easy
    def parse_version(filename):
        return filename.split(".")[-2]

    def filename_format(self, ):
        return f'{self.base_name}.*.{self.ext_name}'

    def fn_regex(self):
        return self.filename_regex(self.base_name, self.ext_name)

    def get_base_and_version(self, search=None):
        search_str = search or f'{self.base_dir}/{self.filename_format()}'
        versions = glob.glob(search_str)

        # early return
        if len(versions) < 1:
            return None, None

        last_version = max(versions, key=os.path.getctime)
        m = re.search(self.fn_regex(), last_version)
        # return filename and version part
        return m.group(0), m.group(2)

    def get_latest_path(self, search=None):
        f, _ = self.get_base_and_version(search=search)
        return f'{self.base_dir}/{f}' if f else None

    def increment_version_number(self, old_version):
        old_version_int = int(old_version, self.max_versions)
        new_version_mod = (old_version_int + 1) % self.max_versions
        return np.base_repr(new_version_mod, self.max_versions)

    def increment_version(self, version=None):
        # set version
        start_version = '0'
        if version is None:
            old_fname, old_version = self.get_base_and_version()
            if old_version:
                version = self.increment_version_number(old_version)
            else:
                version = start_version
        return f'{self.base_dir}/{self.base_name}.{version}.{self.ext_name}'
