import re
import os
from src.log import Log

VERSION_MATCHER = r'\.(.*)\.'

# Need to rework the API
# Client actions
# 1. v = get_current_version (Foo.get_current_version -> current Versioner
# 2. v.bump_version()
# 3. v.name() (path?)


class Versioner:

    @classmethod
    def unversioned_base(cls, base):
        parts = base.split('.')
        if len(parts) > 1 and parts[-1].isdigit():
            parts = parts[0:-1]
        return '.'.join(parts)

    def __init__(self, base, ext, version=0):
        self.log = Log.set(self.__class__.__name__)
        if type(version) != int:
            raise ValueError('version must be an integer')
        self.base = base
        self.ext = ext
        self.version = version
        self.finder = VersionFinder(base, ext)

    def name(self):
        return f'{self.base}.{self.version}.{self.ext}'

    def bump_version(self):
        self.version += 1
        self.log.info(f'Bumped version to {self.version}')
        return self


class FileVersioner(Versioner):
    """
    A Versioner for versioning file names within a particular directory
    """

    def __init__(self, dir, base, ext, version=0):
        self.dir = dir
        super().__init__(base, ext, version)

    def find_all(self):
        entries = os.listdir(self.dir)
        return self.finder.find_all(entries)

    # TODO: Is this method actually used?
    def paths_finder(self):
        versioned_entries = self.find_all()
        return [os.path.join(self.dir, entry) for entry in versioned_entries]

    def get_current_path(self):
        names = self.find_all()
        entry, _ = self.finder.get_latest_name_and_version(names)
        return os.path.join(self.dir, entry)




class VersionFinder:
    DEFAULT = 'default'

    def __init__(self, base, ext):
        self.base = base
        self.ext = ext
        self.matcher = self._matcher()
        self.custom_sorters = {
            # Default sorter sorts by version
            self.DEFAULT: self.get_version
        }

    def register_sorter(self, sorter_name, sorter_fn):
        if sorter_name in self.custom_sorters.keys():
            raise ValueError(f'sorter: {sorter_name} already exists. No overwriting is allowed')
        self.custom_sorters[sorter_name] = sorter_fn

    def _get_sort_key(self, sorter=None):
        if sorter is None:
            sorter = self.custom_sorters[self.DEFAULT]
        elif isinstance(sorter, str):
            sorter = self.custom_sorters[sorter]
        elif callable(sorter):
            pass  # sorter is already callable
        else:
            raise ValueError(f'invalid sorter type: {sorter}')
        return sorter

    def _matcher(self):
        escaped_base = re.escape(self.base)
        escaped_ext = re.escape(self.ext)
        pattern = rf'^{escaped_base}{VERSION_MATCHER}{escaped_ext}$'
        return re.compile(pattern)

    def find_all(self, in_, sorter=None):
        filtered_iter = filter(self.matcher.match, in_)
        entries = list(filtered_iter)
        sort_key = self._get_sort_key(sorter)
        return sorted(entries, key=sort_key)

    def get_version_part(self, name):
        match = re.search(self.matcher, name)
        # reminder that '' in python also evaluates as falsy
        return match and match.group(1)

    def get_version(self, name):
        version_part = self.get_version_part(name)
        if not version_part:
            return None
        return version_part and int(version_part.split('.')[-1])

    def get_latest_name_and_version(self, names, sorter=None):
        sort_key = self._get_sort_key(sorter)
        versioned_names = self.find_all(names)
        if len(versioned_names) == 0:
            return f'{self.base}.0.{self.ext}', 0
        last_version_name = sorted(versioned_names, key=sort_key)[-1]
        return last_version_name, int(self.get_version(last_version_name))
