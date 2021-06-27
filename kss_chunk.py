import glob
import pandas as pd
from enum import Enum
from log import Log


class KssChunkType(Enum):
    CHAR = 'char'
    SYL = 'syl'


class KssChunk:
    DIR = '.data/korean-single-speaker/kss_chunks'

    @staticmethod
    def get_df_from_kss_files(dir, type):
        assert isinstance(type, KssChunkType), f'type: {type} must an instance of KssChunkType (Enum)'
        filenames = sorted(glob.glob(f'{dir}/?_????_{type.value}.csv'))
        return pd.concat([pd.read_csv(f) for f in filenames])

    @classmethod
    def load_all(cls, base_dir=None):
        base_dir = base_dir or KssChunk.DIR
        char_df = cls.get_df_from_kss_files(base_dir, KssChunkType.CHAR)
        syl_df = cls.get_df_from_kss_files(base_dir, KssChunkType.SYL)
        return char_df, syl_df

    def __init__(self, df, kss_id, type, base_dir=None):
        self.log = Log.set(self.__class__.__name__)
        assert isinstance(type, KssChunkType), f'type: {type} must an instance of KssChunkType (Enum)'
        self.base_dir = base_dir or KssChunk.DIR
        self.type = type
        self.kss_id = kss_id
        self.df = df
        self.csv = f'{self.base_dir}/{self.kss_id}_{self.type.value}.csv'

    def save(self):
        self.log.info(f'Saving: {self.csv}')
        self.df.to_csv(self.csv, mode='w', header=True, index=False)
