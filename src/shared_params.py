import joblib
import os
from src.filename_versioner import FilenameVersioner


class SharedParams:
    SHARED_PARAMS_DIR = './data/model'
    FILENAME = 'shared_params'
    FILE_EXT_NAME = 'joblib'

    def __init__(self):
        fv_tuple = (self.FILENAME, self.FILE_EXT_NAME)
        self.fv = FilenameVersioner(fv_tuple, base_dir=self.SHARED_PARAMS_DIR)

    def load(self):
        return joblib.load(self.fv.get_latest_path())


def _validate_path(path):
    assert os.path.exists(path), f'{path} does not exist'


def make_shared_params_cls(params_dir, filename='shared_params', ext='joblib'):
    _validate_path(params_dir)

    class SharedParamsSub(SharedParams):
        SHARED_PARAMS_DIR = params_dir
        FILENAME = filename
        FILE_EXT_NAME = ext

    return SharedParamsSub
