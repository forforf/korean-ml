import joblib
import keras
from src.log import Log
from src.versioner import FileVersioner
from abc import ABC
from collections.abc import Iterable
import numpy as np

# TODO: The sklearn/keras logic doesn't belong here, or if it does, the class/module names should make it clear
SKLEARN = 'sklearn-model'
KERAS = 'keras-model'
TRANSFORMER = 'transformer'
WRAPPER = 'wrapper'
JOBLIB = 'joblib'


def _joblib_loader(obj_path):
    return joblib.load(obj_path)


def _joblib_saver(obj, obj_path):
    joblib.dump(obj, obj_path)


def _keras_loader(keras_path):
    return keras.models.load_model(keras_path)


def _keras_saver(keras_obj, keras_path):
    return keras_obj.save(keras_path)


def might_have_np_inside(a, b):
    try:
        if not len(a) == len(b):
            return False

        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, dict):
            return all([generic_eq(v, b[k]) for k, v in a.items()])
        if isinstance(a, Iterable):
            for i, el in enumerate(a):
                if not generic_eq(el, b[i]):
                    return False
            # l = [maybe_np_eq(aa, bb) for aa, bb in zip(a, b)]
            # return all(l)
        else:  # some other type that doesn't support a == b
            raise NotImplementedError('FOO')

    except (TypeError, KeyError):
        return False


# can probably be improved
def generic_eq(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    if isinstance(a, dict):
        return all([generic_eq(v, b[k]) for k, v in a.items()])

    if isinstance(a, str):
        return a == b

    if isinstance(a, bytes):
        return a == b

    if isinstance(a, Iterable) and not isinstance(a, (dict, str, bytes)):
        if not len(a) == len(b):
            return False

        for i, el in enumerate(a):
            if not generic_eq(el, b[i]):
                return False

        return True

    else:
        return a == b


class FileStorage:
    LOADERS = {
        KERAS: _keras_loader,
        SKLEARN: _joblib_loader,
        TRANSFORMER: _joblib_loader,  # TODO: Deprecate
        WRAPPER: _joblib_loader,
        JOBLIB: _joblib_loader
    }

    SAVERS = {
        KERAS: _keras_saver,
        SKLEARN: _joblib_saver,
        WRAPPER: _joblib_saver,
        JOBLIB: _joblib_saver
    }

    @classmethod
    def get_ext(cls, filename):
        return filename.split('.')[-1]

    def load(self, full_path):
        """
        the filename extension must match one of the loader types.
        """
        loader_type = self.get_ext(full_path)
        loader_fn = self.LOADERS[loader_type]
        return loader_fn(full_path)

    def load_if_valid(self, fname):
        if fname:
            return self.load(fname)
        else:
            return None

    def save(self, obj, full_path):
        """
        the filename extension must match one of the loader types.
        """
        saver_type = self.get_ext(full_path)
        saver_fn = self.SAVERS[saver_type]
        return saver_fn(obj, full_path)


class FileVersionedStorage(ABC):

    def __init__(self, dir='.', data=None, base=None, ext=None):
        self.log = Log.set(self.__class__.__name__)
        self.dir = dir
        self.data = data
        self.base = base
        self.ext = ext
        self.fv = FileVersioner(self.dir, self.base, self.ext)

    def load(self, update=True):
        path = self.fv.get_saved_path()
        # FileStorage picks correct loader based on extension
        loaded_data = FileStorage().load(path)
        if update:
            self.data = loaded_data
        return loaded_data

    def save(self):
        path = self.fv.build_path()
        FileStorage().save(self.data, path)
        self.log.info(f'Saved data to {path}')

    def bump_version_and_save(self):
        self.fv.bump_version()
        self.save()

    def bump_and_save_if_needed(self):
        if self.is_saved():
            self.log.info(f'Did not bump version or save as No changes to the data for {self.base} ({self.ext})')
        else:
            self.bump_version_and_save()

    def is_saved(self):
        orig_data = self.data
        try:
            # self.data is updated here as well
            saved_data = self.load(update=False)
        except FileNotFoundError:
            return False

        return generic_eq(orig_data, saved_data)
