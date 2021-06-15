import math
import librosa
import numpy as np
from log import Log


class Audio:

    def __init__(self, path, sr=None, n_fft=256, n_hops=4):
        self.log = Log.set(self.__class__.__name__)
        self.n_fft = n_fft
        self.n_hops = n_hops
        self.hop_length = math.ceil(self.n_fft/self.n_hops)
        self.path = path
        wav_, sr_ = librosa.load(self.path, sr=sr)
        nice_size = math.floor(len(wav_) / self.hop_length) * self.hop_length
        if nice_size != len(wav_):
            delta = nice_size - len(wav_)
            self.log.warning(f'Modifying input wav from size {len(wav_)} to {nice_size}')
            self.log.info(f'change in number of samples: {delta} [should be less than hop length: {self.hop_length}]')
            assert(delta <= self.hop_length)
        wav_ = wav_[0:nice_size-1]

        # if memory performance becomes a constraint we can skip setting the wav in the constructor
        # and move it into a method instead
        self.wav = librosa.util.normalize(wav_)
        self.mfcc = librosa.feature.mfcc(y=self.wav, sr=sr_, n_mfcc=12, n_fft=n_fft, hop_length=self.hop_length)
        self.rms = self.to_rms(self.wav)
        self.sr = sr_
        self.size = len(self.wav)

    def to_idx(self, t):
        return math.ceil(self.sr * t)

    def to_rms(self, x):
        rms = librosa.feature.rms(y=x, frame_length=self.n_fft, hop_length=self.hop_length)
        print(f'a.to_rms size: {rms.shape[1]}, reduced {(len(x)/self.n_fft)*self.n_hops}')
        return librosa.util.normalize(rms, axis=1)

    # Return numpy array matching audio size
    def val_from_interval(self, df, ivl_cols=None, val_col='value', missing_val=None):
        if ivl_cols is None:
            ivl_cols = ['start', 'stop']
        vals = np.full(self.size, missing_val)
        ivl_df = (df.loc[:, ivl_cols] * self.sr).apply(np.ceil).astype(int)
        ivl_df[val_col] = df.loc[:, val_col]
        for (start, stop, val) in ivl_df.values:
            vals[start:stop] = val
        return vals
