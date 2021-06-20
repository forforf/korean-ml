import textgrid
import pandas as pd


class KssTextGrid:

    # kss id example: 1_0003
    # will refer to textgrid of 1_1003.TextGrid and 1_1003.wav
    def __init__(self, kss_id, base_dir, path=None):
        self.id = kss_id
        self.base_dir = base_dir
        self.tg_file = f'{self.id}.TextGrid'
        self.audio_file = f'{self.id}.wav'
        self.tg_path = path or f'{self.base_dir}/{self.tg_file}'
        self.audio_path = f'{self.base_dir}/{self.audio_file}'
        self.tg = textgrid.TextGrid.fromFile(self.tg_path)

    def chars_df(self):
        char_list = [[iv.mark, iv.minTime, iv.maxTime, self.audio_path] for iv in self.tg[0].intervals]
        return pd.DataFrame(char_list, columns=['char', 'start', 'stop', 'audio'])

    def syls_df(self):
        syl_list = [ [ivl.mark, ivl.minTime, ivl.maxTime, self.audio_path] for ivl in self.tg[1].intervals]
        return pd.DataFrame(syl_list, columns=['syl', 'start', 'stop', 'audio'])