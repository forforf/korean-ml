import re
import textgrid
import pandas as pd

import glob
import pandas as pd
from enum import Enum
from log import Log


class KssDfType(Enum):
    CHAR = 'char'
    SYL = 'syl'


class KssDf:
    """
    Loads KSS Dataframe from CSV by type or Creates KSS Dataframes from TextGrid (all types)
    """

    TG_DIR = './data/korean-single-speaker/kss'
    CSV_DIR = './data/korean-single-speaker/kss-csv'

    def __init__(self, kss_id):
        self.log = Log.set(self.__class__.__name__)
        subdir_id = re.search(r'(\d)_\d{4}', kss_id).group(1)
        self.id = kss_id
        self.audio_filename = f'{self.id}.wav'
        self.tg_filename = f'{self.id}.TextGrid'
        self.audio_path = f'{KssDf.DIR}/{subdir_id}/{self.audio_filename}'
        self.tg_path = f'{KssDf.DIR}/{subdir_id}/{self.tg_filename}'
        self.csv_filenames = {}
        self.csv_paths = {}
        for kss_type in KssDfType:
            self.csv_filenames[kss_type.value] = self.kss_filename(self.id, kss_type)
            self.csv_paths[kss_type.value] = f'{KssDf.CSV_DIR}/{self.csv_filenames[kss_type]}'

    def kss_filename(self, kss_id, kss_enum):
        return f'{kss_id}_{kss_enum.value}.csv'

    def load_csv(self, type=KssDfType.SYL):
        """
        Load compatible data into a DF of the right type
        :param type: KssDfType
        :return: Dataframe
        """
        return pd.read_csv(self.csv_paths[type])

    def load_tg(self, save=False, types=None):
        tg = textgrid.TextGrid.fromFile(self.tg_path)
        df_dict = {}
        for tier in tg.tiers:
            df_list = [[iv.mark, iv.minTime, iv.maxTime, self.audio_path] for iv in tier.intervals]
            columns = [tier.name, 'start', 'stop', 'audio']
            df = pd.DataFrame(df_list, columns=columns)
            df_dict[tier.name] = df

            if save:
                # Persist the dataframe as a csv file
                self.log.info(f'Saving: {self.csv}')
                self.df.to_csv(self.csv_paths[tier.name], mode='w', header=True, index=False)
        return df_dict


class KssTextGrid:
    DIR = './data/korean-single-speaker/kss'
    CSV_DIR = './data/korean-single-speaker/kss-csv'

    # kss id example: 1_0003
    # will refer to textgrid of 1_1003.TextGrid and 1_1003.wav
    def __init__(self, kss_id, path=None):
        subdir_id = re.search(r'(\d)_\d{4}', kss_id).group(1)
        self.id = kss_id
        self.file = f'{self.id}.TextGrid'
        self.audio_file = f'{self.id}.wav'
        self.path = path or f'{KssTextGrid.DIR}/{subdir_id}/{self.file}'
        self.audio_path = f'{KssTextGrid.DIR}/{subdir_id}/{self.audio_file}'
        self.tg = textgrid.TextGrid.fromFile(self.path)

    def chars_df(self):
        char_list = [[iv.mark, iv.minTime, iv.maxTime, self.audio_path] for iv in self.tg[0].intervals]
        return pd.DataFrame(char_list, columns=['char', 'start', 'stop', 'audio'])

    def syls_df(self):
        syl_list = [[ivl.mark, ivl.minTime, ivl.maxTime, self.audio_path] for ivl in self.tg[1].intervals]
        return pd.DataFrame(syl_list, columns=['syl', 'start', 'stop', 'audio'])

    def type_to_csv(self, type='syl'):
        pass

    def to_csv(self):
        pass