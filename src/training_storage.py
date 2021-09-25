import numpy as np
from src.storage import FileVersionedStorage


# TODO: Class actuall loads all training data, not just RMS, fix.
class RmsTrainingStorage(FileVersionedStorage):
    FILENAME = 'xy_trn_rms'
    FILE_EXT_NAME = 'joblib'

    def __init__(self, dir='.', data={}):
        super().__init__(dir, data, self.FILENAME, self.FILE_EXT_NAME)

