import math
import wave
import librosa
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from audio import Audio
from log import Log


# Handles time to samples conversion so it is consistent for all instances
class AudioPlexer:

    # Will verify the wav_files all have the specified sample rate
    # if sample rate is not provided, use the first wav file's sample rate.

    # PERFORMANCE NOTE: We're storing the raw wav file
    def __init__(self, audio_paths, sr=None, n_fft=256, n_hops=4):
        self.log = Log.set(self.__class__.__name__)
        self.audio_paths = audio_paths
        self.sr = sr
        self.n_fft = n_fft
        self.n_hops = n_hops
        self.hop_length = math.ceil(self.n_fft/self.n_hops)

        # validate all files have the same sample rate
        for fn in self.audio_paths:
            with wave.open(fn, 'rb') as f:
                sr_ = f.getframerate()
                if self.sr is None:
                    self.sr = sr_
                assert self.sr == sr_

        self.audios = [Audio(f, sr=self.sr, n_fft=self.n_fft, n_hops=n_hops) for f in self.audio_paths]
        self.size = sum([a.size for a in self.audios])

    def wav(self):
        return np.concatenate([a.wav for a in self.audios])

    def mfcc(self):
        return np.concatenate([a.mfcc for a in self.audios], axis=1)

    def rms(self):
        return np.concatenate([a.rms for a in self.audios], axis=1)

    # Return numpy array matching audio size
    def val_from_interval(self, df, ivl_cols=None, val_col='value', missing_val=None):
        if ivl_cols is None:
            ivl_cols = ['start', 'stop']
        a_dfs = [self.df_per_audio(df, a.path, val_col='syl') for a in self.audios]
        kw = {'ivl_cols':ivl_cols, 'val_col':val_col, 'missing_val':missing_val}
        return np.concatenate([a.val_from_interval(a_df , **kw) for (a_df, a) in zip(a_dfs, self.audios)])

    def df_per_audio(self, full_a_df, audio_path, val_col='syl'):
        a_df = full_a_df.loc[full_a_df['audio'] == audio_path,['start', 'stop', val_col, 'audio']]
        a_df['value'] = a_df[val_col] != '0'
        return a_df

    def to_rms(self, x):
        rms = librosa.feature.rms(y=x, frame_length=self.n_fft, hop_length=self.hop_length)
        return librosa.util.normalize(rms, axis=1)

    # given an X and y, will take n samples of X (ordered series)
    # Example, for len(X) = len(y) = 1000, and a frame_length=5 and y_offset=4
    #  y3, [x0, x1, x2, x3, x4]
    #  y4, [x1, x2, x3, x4, x5]
    #  y5, [x2, x3, x4, x5, x6]
    #  ...
    #  y999, [x996, x997, x998, x999, x1000]
    def sliding_window_features(self, X, y, window, offset=0):
        axis = len(X.shape) - 1
        # we will need to reduce (by slicing) X to match y when we are done
        slc_a = [slice(None)] * (axis + 1)
        x_ = sliding_window_view(X, window, axis=axis)

        # match y with the correct x
        y_ = y.flatten()[offset:x_.shape[axis]]
        slc_a[axis] = slice(0, y_.shape[0])
        slc = tuple(slc_a)

        # return same shape
        return x_[slc], y_
