import re
import textgrid
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
        self.subdir_id = re.search(r'(\d)_\d{4}', kss_id).group(1)
        self.id = kss_id
        self.audio_filename = f'{self.id}.wav'
        self.tg_filename = f'{self.id}.TextGrid'
        self.audio_path = f'{KssDf.TG_DIR}/{self.subdir_id}/{self.audio_filename}'
        self.tg_path = f'{KssDf.TG_DIR}/{self.subdir_id}/{self.tg_filename}'
        self.csv_filenames = {}
        self.csv_paths = {}
        for kss_type in KssDfType:
            self.csv_filenames[kss_type.value] = self.kss_filename(self.id, kss_type.value)
            self.csv_paths[kss_type.value] = f'{KssDf.CSV_DIR}/{self.csv_filenames[kss_type.value]}'

    def kss_filename(self, kss_id, kss_type):
        return f'{kss_id}_{kss_type}.csv'

    def load_csv(self, type=KssDfType.SYL):
        """
        Load compatible data into a DF of the right type
        :param type: KssDfType
        :return: Dataframe
        """
        return pd.read_csv(self.csv_paths[type.value])

    def load_tg(self, save=False):
        tg = textgrid.TextGrid.fromFile(self.tg_path)
        df_dict = {}
        for tier in tg.tiers:
            df_list = [[iv.mark, iv.minTime, iv.maxTime, self.id, self.audio_path] for iv in tier.intervals]
            columns = [tier.name, 'start', 'stop', 'kssid', 'audio']
            df = pd.DataFrame(df_list, columns=columns)
            df_dict[tier.name] = df

            if save:
                # Persist the dataframe as a csv file
                save_path = self.csv_paths[tier.name]
                self.log.info(f'Saving: {save_path}')
                df.to_csv(save_path, mode='w', header=True, index=False)

        return df_dict

    def tg_pred_path(self, data_version, tng_version):
        return f'{KssDf.TG_DIR}/{self.subdir_id}/{self.id}.pred.{data_version}.{tng_version}.TextGrid'
