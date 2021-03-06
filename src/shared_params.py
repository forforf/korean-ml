
from src.storage import FileVersionedStorage


class SharedParams(FileVersionedStorage):
    FILENAME = 'shared_params'
    FILE_EXT_NAME = 'joblib'

    def __init__(self, dir='.', data={}):
        super().__init__(dir, data, self.FILENAME, self.FILE_EXT_NAME)
