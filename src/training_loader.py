import os

import joblib

from src.filename_versioner import FilenameVersioner


class TrainingLoader:

    def __init__(self, filename_versioner):
        xy_file_base, xy_vsn = filename_versioner.get_base_and_version()
        self.training_version = xy_vsn
        self.xy_file = f'{filename_versioner.base_dir}/{xy_file_base}'

    def load(self):
        return joblib.load(self.xy_file)


def make_training_versioner(training_filename, training_dir, ext_name='joblib'):
    assert os.path.exists(training_dir), f'{training_dir} does not exist'
    tng_file_tuple = (training_filename, ext_name)
    return FilenameVersioner(tng_file_tuple, base_dir=training_dir)


def make_training_loader(training_filename, training_dir, ext_name='joblib'):
    return TrainingLoader(make_training_versioner(training_filename, training_dir, ext_name))
