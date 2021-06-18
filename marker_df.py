import pandas as pd


class MarkerDf:

    @classmethod
    def read_csv(cls, csv_f):
        return cls(pd.read_csv(csv_f))

    def __init__(self, df):
        self.df = df

    def intervals(self):
        return self.df['stop'] - self.df['start']

    def audio_files(self):
        return self.df['audio'].unique()

    def markers_by_file(self, cols=None):
        if cols == None:
            cols = self.df.columns
        dict = {}
        for f in self.audio_files():
            col_dict = {}
            for col in cols:
                col_dict[col] = self.df[self.df['audio'] == f][col].values
            dict[f] = col_dict
        return dict