from abc import ABC
from typing import Any, Callable, Protocol

import joblib
import keras.models
import numpy as np
from numpy.typing import ArrayLike

from src.filename_versioner import FilenameVersioner
from src.log import Log

# Reqs
# √ save model with versioning
# √ common transform (SlidingWindow), regardless of underlying model
# √ it/predict use transformed x,y

# Nice to have
# _ Unchanged models do not bump version
# _ model parameters are exposed and can be used for things like GridSearch

SaverFn: Callable


class TransformerProtocol(Protocol):
    X: ArrayLike
    y: ArrayLike

    def fit_transform(self, X: ArrayLike, y: ArrayLike) -> (ArrayLike, ArrayLike):
        ...


class SaverProtocol(Protocol):

    def save(self) -> Any:
        ...


class NoopTransformer:

    def __init__(self):
        """
        NoopTransform constructor. Does not do any transforms
        """
        self.X = None
        self.y = None

    def fit_transform(self, X: ArrayLike, y: ArrayLike):
        self.X = X
        self.y = y
        return X, y


class BaseTransform:
    def __init__(self, transformer: TransformerProtocol = NoopTransformer()):
        """
        KerasTransform constructor
        :param transformer: When fitted, the transformer.X and transformer.y should be the model input X and y
        """
        self.transformer = transformer
        self.log = Log.set(self.__class__.__name__)

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
        self.log.info(f'transformed X, y shapes: {self.transformer.X.shape}, {self.transformer.y.shape}')
        return self.transformer.X, self.transformer.y


class KerasTransform:
    """
    Used to wrap SciKitLearn-like transformers to be compatible with the model wrapper
    """

    def __init__(self, transformer: TransformerProtocol = NoopTransformer()):
        """
        KerasTransform constructor
        :param transformer: When fitted, the transformer.X and transformer.y should be the model input X and y
        """
        self.transformer = transformer
        self.log = Log.set(self.__class__.__name__)

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


class ModelSaver(ABC):

    def __init__(self, model, path):
        self.model = model
        self.path = path

    def _set_path(self, path=None):
        if path is not None:
            self.path = path

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class KerasSaver(ModelSaver):

    def __init__(self, model, path):
        super().__init__(model, path)

    def save(self, path=None):
        self._set_path(path)
        self.model.save(self.path)

    def load(self, path=None):
        self._set_path(path)
        return keras.models.load_model(self.path)


class SKLearnSaver(ModelSaver):

    def __init__(self, model, path):
        super().__init__(model, path)

    def save(self, path=None):
        self._set_path(path)
        joblib.dump(self.model, self.path)

    def load(self, path=None):
        self._set_path(path)
        return joblib.load(self.path)


class VersionedSaver:

    def __init__(self, saver: ModelSaver, file_versioner: FilenameVersioner):
        self.saver = saver
        self.fv = file_versioner
        self.log = Log.set(self.__class__.__name__)

    def bump_version(self):
        return self.fv.increment_version()
    
    def get_path(self):
        return self.fv.get_latest_path()

    def save(self):
        saved_model_path = self.bump_version()
        self.log.info(f'Saving model to: {saved_model_path}')
        return self.saver.save(saved_model_path)

    def load(self):
        saved_model_path = self.get_path()
        return self.saver.load(saved_model_path)


class ModelWrapper:

    def __init__(self, model, transformer, versioned_saver):
        self.model = model
        self.transformer = transformer
        self.versioned_saver = versioned_saver
        self.log = Log.set(self.__class__.__name__)

    # Although we could have set this up in constructor, using a defined method is more explicit
    def transform(self, X, y):
        return self.transformer.transform(X, y)

    def fit(self, X, y, **kwargs):
        self.log.info(f'fitting using model: {self.model}')
        X_t, y_t = self.transform(X, y)
        return self.model.fit(X_t, y_t, **kwargs)

    def predict(self, X):
        X_t, _ = self.transform(X, None)
        return self.model.predict(X_t)

    def save(self):
        return self.versioned_saver.save()

    def load(self):
        return self.versioned_saver.load()


def get_KerasWrapper(model_name,
                    model,
                    model_dir,
                    ext='model',
                    training_version='0',
                    max_versions=3,
                    transformer=NoopTransformer):
    model_file = f'{model_name}.{training_version}'
    fv_tuple = (model_file, ext)
    fv = FilenameVersioner(fv_tuple, model_dir, max_versions=max_versions)
    kt = KerasTransform(transformer)
    ks = KerasSaver(model, fv.get_latest_path())
    versioned_saver = VersionedSaver(ks, fv)
    return ModelWrapper(model, kt, versioned_saver)


def get_SKLearnWrapper(model_name,
                          model,
                          model_dir,
                          ext='model',
                          training_version='0',
                          max_versions=3,
                          transformer=NoopTransformer):
    model_file = f'{model_name}.{training_version}'
    fv_tuple = (model_file, ext)
    fv = FilenameVersioner(fv_tuple, model_dir, max_versions=max_versions)
    t = BaseTransform(transformer)
    s = SKLearnSaver(model, fv.get_latest_path())
    versioned_saver = VersionedSaver(s, fv)
    return ModelWrapper(model, t, versioned_saver)
