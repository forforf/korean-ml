import logging

import math
import librosa
import numpy as np
from src.log import Log


class Audio:
    def __init__(self, path, sr=None, n_fft=256, n_hops=4, n_mfcc=8):
        self.log = Log.set(self.__class__.__name__)
        self.log.setLevel(logging.WARNING)
        self.n_fft = n_fft
        self.n_hops = n_hops
        self.hop_length = math.ceil(self.n_fft/self.n_hops)
        self.n_mfcc = n_mfcc
        self.path = path
        wav_, sr_ = librosa.load(self.path, sr=sr)
        nice_size = math.ceil(len(wav_) / self.hop_length) * self.hop_length
        delta = 0
        if nice_size != len(wav_):
            delta = nice_size - len(wav_)
            self.log.warning(f'Modifying input wav from size {len(wav_)} to {nice_size}')
            self.log.info(f'change in number of samples: {delta} [should be less than hop length: {self.hop_length}]')
            assert delta <= self.hop_length
        wav_ = np.pad(wav_, (0, delta), 'constant')

        # if memory performance becomes a constraint we can skip setting the wav in the constructor
        # and move it into a method instead
        self.wav = librosa.util.normalize(wav_)
        self.mfcc = librosa.feature.mfcc(y=self.wav, sr=sr_, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=self.hop_length)
        self.rms = self.to_rms(self.wav)
        self.rms_frame_rate = self.n_fft
        self.sr = sr_
        self.size = len(self.wav)

    def __eq__(self, o):
        if not isinstance(o, Audio):
            return False

        are_same = [
            self.n_fft == o.n_fft,
            self.n_hops == o.n_hops,
            self.path == o.path,
            self.sr == o.sr,
            np.array_equal(self.wav, o.wav),
            np.array_equal(self.rms, o.rms)
        ]
        return all(are_same)

    def __ne__(self, o):
        return not self == o

    def to_idx(self, t):
        return math.ceil(self.sr * t)

    def to_rms(self, x):
        rms = librosa.feature.rms(y=x, frame_length=self.n_fft, hop_length=self.hop_length)
        self.log.info(f'a.to_rms size: {rms.shape[1]}, reduction factor {(len(x)/self.n_fft)*self.n_hops}')
        return librosa.util.normalize(rms, axis=1)

    # Return numpy array matching audio size
    # val_col here must point to a valid type (i.e., boolean)
    def val_from_interval(self, df, ivl_cols=None, val_col='value', missing_val=None):
        if ivl_cols is None:
            ivl_cols = ['start', 'stop']
        vals = np.full(self.size, missing_val)
        ivl_df = (df.loc[:, ivl_cols] * self.sr).apply(np.ceil).astype(int)
        ivl_df[val_col] = df.loc[:, val_col]
        for (start, stop, val) in ivl_df.values:
            vals[start:stop] = val
        return vals

    # TODO: Figure out a way to make this more flexible and not tightly coupled to 'syl' column
    def speech_from_interval(self, df, **kwargs):
        assert 'syl' in df.columns, 'Dataframe must have column named: \'syl\''
        df['value'] = np.where(df['syl'] == '0', False, True)
        return self.val_from_interval(df, **kwargs)
