import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Final
from pipewrap import PipeWrap
from transformers import SlidingWindow
from scorers import conv_var
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from filename_versioner import FilenameVersioner
import joblib
import keras.models

# Reqs
# save model with versioning
# common transform (SlidingWindow), regardless of underlying model
# fit/predict use transformed x,y

# Nice to have
# Unchanged models do not bump version
# model parameters are exposed and can be used for things like GridSearch

class Wrap(ABC):
    MODEL_DIR: Final[str] = './data/model'
    SW_WINDOW = 128
    SW_OFFSET_PERCENT = 0.75

    def __init__(self, model_name, model, training_version='0'):
        self.model = model
        self.model_name = model_name
        self.model_file = f'{model_name}.{training_version}'
        # TODO: Deprecate .sw, and pass the transformer into init
        self.sw = SlidingWindow(window=128, offset_percent=0.75)
        # self.transformer = transformer

    def transform(self, X, y):
        if y is None:
            y = np.zeros(len(X))
        self.sw.fit_transform(X, y)
        print('swX, swy shapes', self.sw.X.shape, self.sw.y.shape)
        y_sw = self.sw.y.astype('float32')
        X_sw = self.sw.X.reshape(self.sw.X.shape[0],self.sw.X.shape[1],1)
        return (X_sw, y_sw)

    def fit(self, X, y, **kwargs):
        print(f'fitting using model: {self.model}')
        X_sw, y_sw = self.transform(X, y)
        return self.model.fit(X_sw, y_sw, **kwargs)

    def predict(self, X):
        X_sw, _ = self.transform(X, None)
        y_pred_cont = self.model.predict(X_sw).squeeze()
        #TODO: Figure out better way to convert to boolean (and pass the threshold as a parameter)
        threshold = 0.2
        return (y_pred_cont > threshold).astype(float)

    def save(self):
        raise NotImplementedError




class KerasWrap(Wrap):
    FILE_EXT_NAME = 'keras'

    def __init__(self, model_name, model, training_version='0'):
        super(KerasWrap, self).__init__(model_name, model, training_version)
        model_fv_tuple = (self.model_file, self.FILE_EXT_NAME)
        self.fv = FilenameVersioner(model_fv_tuple, base_dir=self.MODEL_DIR, max_versions=3)

    def save(self):
        saved_model_path = self.fv.increment_version()
        print(saved_model_path)
        self.model.save(saved_model_path)

        # return saved model
        return keras.models.load_model(saved_model_path)

class TrainingLoader:
    TRAINING_DATA_DIR = './data/model'
    FILE_EXT_NAME = 'joblib'

    def __init__(self, training_filename):
        self.training_filename = training_filename

        file_ext_tuple = (self.training_filename, TrainingLoader.FILE_EXT_NAME)
        self.fv_xy = FilenameVersioner(file_ext_tuple, base_dir = TrainingLoader.TRAINING_DATA_DIR)

        xy_file_base, xy_vsn = self.fv_xy.get_base_and_version()
        self.training_version = xy_vsn
        self.xy_file = f'{self.fv_xy.base_dir}/{xy_file_base}'

    def load(self):
        return joblib.load(self.xy_file)


class SharedParams:
    SHARED_PARAMS_DIR = './data/model'
    FILENAME = 'shared_params'
    FILE_EXT_NAME = 'joblib'

    def __init__(self):
        fv_tuple = (SharedParams.FILENAME, SharedParams.FILE_EXT_NAME)
        self.fv = FilenameVersioner(fv_tuple, base_dir=SharedParams.SHARED_PARAMS_DIR)

    def load(self):
        return joblib.load(self.fv.get_latest_path())


class ModelWrap:
    SW_WINDOW = 128
    SW_OFFSET_PERCENT = 0.75
    MODEL_DIR = './data/model'
    FILE_EXT_NAME = 'joblib'
    SHARED_PARAMS_NAME = 'shared_params'

    def __init__(self, model_name, model, model_params, training_version='0'):
        self.model = model
        model_file = f'{model_name}.{training_version}'
        model_fv_tuple = (model_file, ModelWrap.FILE_EXT_NAME)
        self.fv = FilenameVersioner(model_fv_tuple, base_dir=ModelWrap.MODEL_DIR, max_versions=3)

        self.sliding_window = SlidingWindow()


        # from transformers import SlidingWindow
        steps = [('sw', self.sliding_window), (model_name, model)]
        self.pw = PipeWrap(steps)
        sw_params = {
            'sw__window': [ModelWrap.SW_WINDOW],
            'sw__offset_percent': [ModelWrap.SW_OFFSET_PERCENT]
        }
        pipe_params = (sw_params | model_params)
        tcsv = TimeSeriesSplit(n_splits=5)
        conv_score = make_scorer(conv_var)
        # self.search = GridSearchCV(self.pw.pipe, pipe_params, n_jobs=-1, cv=tcsv, scoring='f1', verbose=4)
        self.search = GridSearchCV(self.pw.pipe, pipe_params, n_jobs=-1, cv=tcsv, scoring=conv_score, verbose=4)

    # Need a better approach. Keras models don't pickle (but can be saved as json and their weights reloaded)
    # One possible approach is to have a common interface for saving loading
    # and separate implementations for sklearn pipelines and keras models.
    # TODO: Better model wrappping.
    # FIXME: model_fit might be breaking save versioning
    def model_fit(self, X, y):
        sw = SlidingWindow(window=128, offset_percent=0.75)

        sw.fit_transform(X, y)

        print('swX, swy shapes', sw.X.shape, sw.y.shape)
        y_sw = sw.y.astype('float32')
        X_sw = sw.X.reshape(sw.X.shape[0],sw.X.shape[1],1)
        print(f'fitting using model: {self.model}')
        self.model.fit(X_sw, y_sw)
        print(f'setting pipeline model to use fitted model')
        self.pw.pipe = self.model

    # TODO: Rename to search_fit, and then rename 'model_fit' to 'fit'
    def fit(self, X, y):
        print(f'fitting using grid search: {self.search}')
        self.search.fit(X, y)
        print(f'setting pipeline model to use best estimator: {self.search.best_estimator_}')
        self.pw.pipe = self.search.best_estimator_

    def save(self):
        saved_model_path = self.fv.get_latest_path()
        print(saved_model_path)

        saved_wrap = joblib.load(saved_model_path) if saved_model_path else None

        # Save the pipeline so we can modify it later without changing this code
        if saved_wrap is None or self.pw.params != saved_wrap.params:
            print('Detected new trained model')
            versioned_model_file = self.fv.increment_version()
            print(f'saving model as {versioned_model_file}')
            joblib.dump(self.pw, versioned_model_file)
        else:
            print('No changes detected')

        return saved_wrap


class ModelPlot:

    @staticmethod
    def pred_threshold(pred_vals, thresh=0.5):
        return np.where(pred_vals > thresh, True, False)

    @staticmethod
    def align(y, offset):
        return np.pad(y, (offset, 0), 'minimum')

    @staticmethod
    def delta(y1, y2, y1_offset, y2_offset):
        min_len = min(len(y1), len(y2))
        return ModelPlot.align(y1, y1_offset)[0:min_len] - ModelPlot.align(y2, y2_offset)[0:min_len]

    @staticmethod
    def speech_base_delta(y_pred, y, offset):
        return ModelPlot.delta(y_pred, y, offset, 0)

    @staticmethod
    def plot(X_train, y_train, y_train_pred, sw_offset, x_rms, x_sw, sw_mean_offset):
        fig, axs  = plt.subplots(6, 1, figsize=(18,20))
        axs[0].set_title(f'Mean of Sliding Window')
        axs[0].plot(X_train, color='lightblue', alpha=0.4, label='X')
        axs[0].plot(X_train - x_rms, color='yellow', alpha=0.4, label='X-X_rms')
        axs[0].plot(ModelPlot.align(np.mean(x_sw, axis=1), sw_mean_offset), color='cyan', alpha=0.6, label='sliding window mean (txfm X)')
        axs[0].legend(loc='center right')

        axs[1].set_title(f'Speech/No Speech (manually identified)')
        axs[1].plot(x_rms, color='slategray', alpha=0.4, label='rms of audio')
        axs[1].plot(y_train, color='lightblue', alpha=0.8, label='1: speech, 0: no speech')
        axs[1].legend(loc='center right')

        axs[2].set_title(f'Predicted vs Input')
        axs[2].plot(y_train, color='lightblue', alpha=0.8, label='1: speech, 0: no speech')
        axs[2].plot(ModelPlot.align(y_train_pred, sw_offset), color='cyan', alpha=0.6, label='prediction')
        axs[2].legend(loc='center right')

        axs[3].set_title(f'Difference to base Speech/No Speech Input')
        axs[3].plot(x_rms, color='slategray', alpha=0.3, label='rms of audio')
        axs[3].plot(y_train.astype(int)-y_train.astype(int), color='yellow', alpha=0.8, label='base')
        axs[3].plot(ModelPlot.speech_base_delta(y_train.astype(int), y_train, sw_offset), color='lightblue', alpha=0.8, label="padded to align")
        axs[3].plot(ModelPlot.speech_base_delta(y_train_pred, y_train, sw_offset), color='thistle', alpha=0.5, label='pred delta')
        axs[3].legend(loc='lower right')

        axs[4].set_title(f'Boolean Prediction')
        axs[4].plot(x_rms, color='slategray', alpha=0.3, label='rms of audio')
        axs[4].plot(1*ModelPlot.pred_threshold(y_train, 0.5), color='lime', alpha=0.4, label='manual')
        axs[4].legend(loc='center right')

        axs[5].set_title(f'Prediction Deltas to Base')
        axs[5].plot(ModelPlot.speech_base_delta(y_train_pred, y_train, sw_offset), color='thistle', alpha=0.5, label='pred vals - base')
        axs[5].axhspan(-0.5, 0.5, color='lightblue', alpha=0.3, label='0.5 thresh'),
        axs[5].axhspan(-0.75, 0.75, color='slategray', alpha=0.3, label='0.75 thresh')
        axs[5].legend(loc='upper right')

        plt.tight_layout()
        plt.show()
