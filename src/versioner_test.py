import pytest
import os
import random
import re
from collections.abc import Iterable

from src.versioner import Versioner, FileVersioner, VersionFinder

#TODO: Move BASE and EXT from global test scope
BASE = 'basename'
EXT = 'extension'
V0_ARGS = [
    # ((BASE, EXT), {}, 0),
    ((BASE, EXT, 0), {}, 0),
    ((BASE, EXT), {'version': 0}, 0)
]

V1_ARGS = [
    ((BASE, EXT, 1), {}, 1),
    ((BASE, EXT), {'version': 1}, 1)
]


def version_matcher(prefix, ext):
    return [
        (f'{prefix}.0.{ext}', prefix, ext, 0),
        (f'{prefix}.0.42.{ext}', f'{prefix}.0', ext, 42),
        (f'{prefix}.42.0.{ext}', f'{prefix}.42', ext, 0)
    ]


BASE_VERSION_MATCHER = version_matcher('a', 'b')


def idFn(val):
    if isinstance(val, (Iterable,)):
        return '-'.join([str(el) for el in val])


class TestVersioner:

    def test_requires_integer_version(self):
        base = 'a'
        ext = 'b'
        with pytest.raises(ValueError):
            Versioner(base, ext, '0')

    @pytest.mark.parametrize('args,kwargs,version', V0_ARGS+V1_ARGS, ids=idFn)
    def test_throwaway(self, args, kwargs, version):
        assert True

    @pytest.mark.parametrize('args,kwargs,version', V0_ARGS+V1_ARGS, ids=idFn)
    def test_instance_attributes(self, args, kwargs, version):
        base = args[0]
        ext = args[1]
        versioner = Versioner(*args, **kwargs)

        assert versioner.base == base
        assert versioner.ext == ext
        assert versioner.version == version

    @pytest.mark.parametrize('args,kwargs,version', V0_ARGS+V1_ARGS, ids=idFn)
    def test_name(self, args, kwargs, version):
        base = args[0]
        ext = args[1]
        expected_name = f'{base}.{version}.{ext}'
        versioner = Versioner(*args, **kwargs)
        assert versioner.name() == expected_name

    @pytest.mark.parametrize('args,kwargs,version', V0_ARGS+V1_ARGS, ids=idFn)
    def test_bump_version(self, args, kwargs, version):
        versioner = Versioner(*args, **kwargs)
        assert versioner.bump_version().version == version + 1

    def test_multi_version_name(self):
        base = 'foo.0.0'
        ext = 'ext'
        version = 42
        expected_name = f'{base}.{version}.{ext}'
        assert Versioner(base, ext, version).name() == expected_name


class TestFileVersioner:
    BASE = 'base'
    EXT = 'ext'
    DIR = os.path.join('some', 'dir')
    VERSIONED_ENTRIES = [
        f'{BASE}.0.{EXT}', f'{BASE}.1.{EXT}',
        f'{BASE}.0.0.{EXT}', f'{BASE}.0.42.{EXT}', f'{BASE}.42.0.{EXT}'
    ]
    DIR_ENTRIES = ['.', '..', 'not-a-versioned-file'] + VERSIONED_ENTRIES

    @pytest.mark.parametrize('args,kwargs,version', V0_ARGS+V1_ARGS, ids=idFn)
    def test_instance_attributes(self, args, kwargs, version):
        base = args[0]
        ext = args[1]
        # versioner = Versioner(*args, **kwargs)
        dir = '/some/path/'
        # file_versioner = FileVersioner(dir, versioner)
        file_versioner = FileVersioner(dir, *args, **kwargs)
        assert file_versioner.dir == dir
        assert file_versioner.base == base
        assert file_versioner.ext == ext
        assert file_versioner.version == version
        # assert file_versioner.versioner == versioner

    def test_find_all(self, mocker):
        os_listdir_mock = mocker.patch('os.listdir')
        os_listdir_mock.return_value = self.DIR_ENTRIES

        file_versioner = FileVersioner(self.DIR, self.BASE, self.EXT)
        assert file_versioner.find_all() == self.VERSIONED_ENTRIES

    def test_paths_finder(self, mocker):
        os_listdir_mock = mocker.patch('os.listdir')
        os_listdir_mock.return_value = self.DIR_ENTRIES
        versioned_paths = [os.path.join(self.DIR, entry) for entry in self.VERSIONED_ENTRIES]

        file_versioner = FileVersioner(self.DIR, self.BASE, self.EXT)
        assert file_versioner.paths_finder() == versioned_paths


def _version_generator_fn(base, ext):
    def version_generator(vsn_tuple):
        vsn, vsn_prefix = vsn_tuple
        return vsn, f'{vsn_prefix}{vsn}', f'{base}.{vsn_prefix}{vsn}.{ext}', base, ext

    return version_generator


TVF_BASE = 'a'
TVF_EXT = 'b'


@pytest.fixture()
def tvf_base_ext():
    yield (TVF_BASE, TVF_EXT)


class TestVersionFinder:
    V0 = (0, '')
    V1 = (1, '')
    V42 = (42, '')
    V003 = (3, '0.0.')
    VSN_TUPLES = [V1, V0, V42, V003]
    # VERSION_DATA looks like [(1, '1', 'a.1.b', 'a', 'b'), ... (3, '0.0.3', 'a.0.0.3.b', 'a', 'b')]
    VERSION_DATA = [_version_generator_fn(TVF_BASE, TVF_EXT)(vsn_tuple) for vsn_tuple in VSN_TUPLES]
    LATEST_VERSION_TUPLE = _version_generator_fn(TVF_BASE, TVF_EXT)(V42)
    NON_VERSION_DATA = [
        (None, None, 'a', 'a', None),
        (None, None, 'a.b', 'a', 'b'),
        (None, None, 'b.a', 'b', 'a'),
        (2, '1.2', 'a.1.2.3', 'a', '3'),
        (3, '1.2.3', 'b.1.2.3.b', 'b', 'b'),
        (3, '2.3', '1.2.3.b', '1', 'b'),
        (3, '1.2.3', '.1.2.3.b', '', 'b')
    ]
    ALL_DATA = VERSION_DATA+NON_VERSION_DATA
    random.Random(0).shuffle(ALL_DATA)

    @pytest.mark.parametrize('version,version_str,name,base,ext', VERSION_DATA)
    def test_matcher(self, version, version_str, name, base, ext):
        pattern = VersionFinder(base, ext).matcher()
        match = re.search(pattern, name)
        assert match
        assert match.group(1) == version_str
        assert not re.search(pattern, f'foo{name}foo')
        assert not re.search(pattern, f'foo{name}.{ext}')

    def test_find_all(self, tvf_base_ext):
        base, ext = tvf_base_ext
        full_list = [el[2] for el in self.__class__.ALL_DATA]
        expected_list = [el[2] for el in self.__class__.VERSION_DATA]
        assert sorted(VersionFinder(base, ext).find_all(full_list)) == sorted(expected_list)

    @pytest.mark.parametrize('version,version_str,name,base,ext', VERSION_DATA)
    def test_get_version_part(self, version, version_str, name, base, ext):
        version_part = VersionFinder(base, ext).get_version_part(name)
        assert version_part == version_str

    @pytest.mark.parametrize('version,version_str,name,base,ext', VERSION_DATA)
    def test_get_version(self, version, version_str, name, base, ext):
        version = VersionFinder(base, ext).get_version(name)
        assert version == version

    def test_get_latest_name_and_version_default(self, tvf_base_ext):
        base, ext = tvf_base_ext
        full_list = [el[2] for el in self.__class__.ALL_DATA]
        expected_version, _, expected_name, _, _ = self.__class__.LATEST_VERSION_TUPLE
        latest_name, latest_version = VersionFinder(base, ext).get_latest_name_and_version(full_list)
        assert latest_name == expected_name
        assert latest_version == expected_version

    def test_get_latest_name_and_version_registered_sorter(self):
        vf = VersionFinder('a', 'b')
        sorter_name = 'foo'
        sorter_fn = lambda x: re.search(r'\d+', x).group()
        vf.register_sorter(sorter_name, sorter_fn)
        assert vf.get_latest_name_and_version(['a.1.b', 'a.3.b', 'a.2.b'], sorter_name) == ('a.3.b', 3)

    def test_get_latest_name_and_version_inline_sorter(self):
        vf = VersionFinder('a', 'b')
        sorter_fn = lambda x: re.search(r'\d+', x).group()
        assert vf.get_latest_name_and_version(['a.1.b', 'a.3.b', 'a.2.b'], sorter_fn) == ('a.3.b', 3)

    def  test_get_latest_name_and_version_invalid_sorter(self):
        vf = VersionFinder('a', 'b')
        invalid_sorter_fn = object()
        with pytest.raises(ValueError):
            vf.get_latest_name_and_version(['a.1.b', 'a.3.b', 'a.2.b'], invalid_sorter_fn)



