from abc import ABC
from typing import Any, Callable, Protocol

import joblib
import keras.models
import numpy as np
from numpy.typing import ArrayLike

from src.log import Log
from src.storage import FileStorage, KERAS as KERAS_EXT, SKLEARN as SKLEARN_EXT
import os
import shutil
from src.versioner import FileVersioner, VersionFinder

# Reqs
# √ save model with versioning
# √ common transform (SlidingWindow), regardless of underlying model
# √ it/predict use transformed x,y

# Nice to have
# _ Unchanged models do not bump version
# _ model parameters are exposed and can be used for things like GridSearch

# ModelWrapper Functional Reqs
# * Incorporate feature transformers
#   * Add transformers to the fit/predict flow
#   * Save/Load transformers
# * Expose underlying model
# * Save/Load Model
# * Work for both Keras and SKLearn models (maybe more in the future)

# API
# Factory with context (i.e. project) settings
#   * Load/save path
#
# Model Wrapper
#   * model id
#   * model
#   * transformer (Up to the user to wrap multiple transforms)
#      * document transformer contract (maybe type hints is sufficient?)
#      * transformer needs wrapping too, as it's slightly different between Keras and SKLearn
#
# Persistence
#   * save model
#   * Save transformer
#   * load model
#   * load transformer
#   * save model/transformer
#   * load model/transformer

# Open Question
# For different model family (Keras vs SKLearn), what is best way to differentiate?
#   * When saving, maybe use model.__class__.__module__.split('.')[0] -> 'sklearn' or 'keras' to determine how to save
#   * When loading, extension should indicate if model is keras or sklearn.


class WrapperStore:

    def __init__(self, wrapper_id, model_path, transformer_path):
        self.id = wrapper_id
        self.model_path = model_path
        self.transformer_path = transformer_path


class ModelWrapper(ABC):
    KERAS_MODULE_ROOT = 'keras'
    SKLEARN_MODULE_ROOT = 'sklearn'
    TRANSFORMER_MODULE_ROOT = 'src'
    WRAPPER_EXT = 'wrapper'

    EXT_MAP = {
        KERAS_MODULE_ROOT: KERAS_EXT,
        SKLEARN_MODULE_ROOT: SKLEARN_EXT,
        TRANSFORMER_MODULE_ROOT: SKLEARN_EXT  # TODO This is a bad hack, as it maps any transformer to the sklearn ext
    }

    @classmethod
    def _get_ext_map_key_from_module(cls, module):
        return module.split('.')[0]

    @classmethod
    def load(cls, fullpath):
        wrapper_store = joblib.load(fullpath)
        assert isinstance(wrapper_store, WrapperStore)
        fs = FileStorage()
        wrapper_id = wrapper_store.id
        model = fs.load(wrapper_store.model_path)
        transformer = fs.load(wrapper_store.transformer_path)
        if cls._get_ext_map_key_from_module(model.__module__) == cls.KERAS_MODULE_ROOT:
            return KerasModelWrapper(wrapper_id, model, transformer, os.path.dirname(fullpath))
        else:
            return cls(wrapper_id, model, transformer, os.path.dirname(fullpath))

    def __init__(self, model_id, model, transformer=None, save_dir='.'):
        self.log = Log.set(self.__class__.__name__)
        self.save_dir = save_dir
        self.id = model_id
        self.model = model  # or model_fetcher if model is None (to allow for saving/loading non-serializable models)
        self.ext = self._get_ext(model)
        self.transformer_id = f'{model_id}-transformer'
        self.transformer = transformer
        self.transformer_ext = self._get_ext(transformer)

    def _get_ext(self, model):
        ext_key = self._get_ext_map_key_from_module(model.__module__)
        return self.EXT_MAP[ext_key]

    def _get_latest_version_from_model(self, base_id, model):
        ext = self._get_ext(model)
        return self._find_latest_version(base_id, ext)

    def _save_path(self, fname):
        return os.path.join(self.save_dir, fname)

    def find_entries(self, id_, ext):
        vf = VersionFinder(id_, ext)
        entries = os.listdir(self.save_dir)
        return vf.find_all(entries)

    def _find_latest_version(self, id_, ext):
        vf = VersionFinder(id_, ext)
        entries = os.listdir(self.save_dir)
        fnames = vf.find_all(entries)
        return vf.get_latest_name_and_version(fnames)

    def find_latest_wrapper(self):
        return self._find_latest_version(self.id, self.WRAPPER_EXT)

    def get_current_wrapper_path(self):
        fname, vsn = self.find_latest_wrapper()
        return os.path.join(self.save_dir, fname)

    def _get_and_bump_version(self, id_, ext):
        # ext = self._get_ext(model)
        fname, vsn = self._find_latest_version(id_, ext)
        full_base, _ = os.path.splitext(fname)
        base = FileVersioner.unversioned_base(full_base)
        fv = FileVersioner(self.save_dir, base, ext, vsn)
        fv.bump_version()
        new_fname = fv.name()
        return new_fname

    def transform(self, X, y):
        """
        Applies transformer and converts X into appropriate shape (3D matrix) for Keras models.
        :param X: features
        :param y: [optional] predictions
        :return: transformed X,y
        """
        if y is None:
            y = np.zeros(len(X))
        x_txfm = self.transformer.fit_transform(X, y)
        self.log.info(f'transformed X, y shapes: {self.transformer.X.shape}, {self.transformer.y.shape}')
        return x_txfm, y

    def fit(self, X, y, **kwargs):
        self.log.info(f'fitting using model: {self.model}')
        X_t, y_t = self.transform(X, y)
        return self.model.fit(X_t, y_t, **kwargs)

    def predict(self, X):
        X_t, _ = self.transform(X, None)
        return self.model.predict(X_t)

    def save(self):
        model_fname = self._get_and_bump_version(self.id, self.ext)
        transformer_fname = self._get_and_bump_version(self.transformer_id, self.transformer_ext)
        wrapper_fname = self._get_and_bump_version(self.id, self.WRAPPER_EXT)
        wrapper_path = self._save_path(wrapper_fname)
        model_path = self._save_path(model_fname)
        transformer_path = self._save_path(transformer_fname)
        ws = WrapperStore(self.id, model_path, transformer_path)
        FileStorage().save(self.model, model_path)
        FileStorage().save(self.transformer, transformer_path)
        FileStorage().save(ws, wrapper_path)

    def _delete_path(self, path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    def remove_old_versions(self, keep=3):
        assert keep >= 0
        id_exts = [(self.id, self.ext), (self.transformer_id, self.transformer_ext), (self.id, self.WRAPPER_EXT)]
        for id_, ext in id_exts:
            entries_to_remove = self.find_entries(id_, ext)[0:-keep]
            paths = [os.path.join(self.save_dir, entry) for entry in entries_to_remove]
            [self._delete_path(path) for path in paths]


class KerasModelWrapper(ModelWrapper):

    def transform(self, X, y):
        """
        Applies transformer and converts X into appropriate shape (3D matrix) for Keras models.
        :param X: features
        :param y: [optional] predictions
        :return: transformed X,y
        """
        if y is None:
            y = np.zeros(len(X))
        self.transformer.fit_transform(X, y)
        y_t = self.transformer.y.astype('float32')
        X_t = self.transformer.X.reshape(self.transformer.X.shape[0], self.transformer.X.shape[1], 1)
        self.log.info(f'transformed X, y shapes: {self.transformer.X.shape}, {self.transformer.y.shape}')
        return X_t, y_t
