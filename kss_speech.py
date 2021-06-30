import pandas as pd
import numpy as np
from numpy import ndarray

from audio import Audio
# from transformers import SlidingWindow
from kss_df import KssDfType


# TODO: This does not check that the start/stop columns are matching to the right audio file
#       It assumes there is only 1 audio file for the entire df.
# TODO: We've mixed up fitting and tranforming in the init, so as is this is problematic for predictions
class KssSpeech:
    #TODO: Figure out whether we use kss_id and derive df and audio, or pass them in.
    #TODO: Figure out if size and frame rate should be in constructor or not.
    # TODO: Deprecate is_speech (or speech_bools as they are the same)
    def __init__(self, df: pd.DataFrame, audio: Audio):
        self.df = df
        self.audio = audio
        # self.sw = sw
        audio_rms = self.audio.rms
        frame_rate = self.audio.sr / self.audio.hop_length
        speech_bools = self.speech_wav(len(audio_rms.squeeze()), frame_rate)
        self._speech_bools = speech_bools
        # self.sw.fit_transform(audio.rms, speech_bools)
        # self.rms = audio_rms.squeeze()
        # self.is_speech = self.sw.y
        self.is_speech = speech_bools

    def __eq__(self, o):
        if not isinstance(o, KssSpeech):
            return False

        are_same = [
            self.df.equals(o.df),
            self.audio == o.audio,
            # self.sw == o.sw
            ]
        return all(are_same)

    def __ne__(self, o):
        return not self == o

    # TODO: Figure out a way to make this more flexible and not tightly coupled to 'syl' column
    def speech_wav(self, size, rate, **kwargs):
        assert KssDfType.SYL.value in self.df.columns, f'Dataframe must have column based on KssDfType.SYL: \'{KssDfType.SYL.value}\''
        self.df['value'] = np.where(self.df[KssDfType.SYL.value] == '0', False, True)
        return self._val_from_interval(size=size, frame_rate=rate, missing_val=False, **kwargs)

    # Return numpy array matching audio size
    # val_col here must point to a valid type (i.e., boolean)
    def _val_from_interval(self, size=None, frame_rate=None, ivl_cols=None, val_col='value', missing_val=None):
        fr = frame_rate if frame_rate else self.audio.sr
        sz = size if size else len(self.audio.wav)
        if ivl_cols is None:
            ivl_cols = ['start', 'stop']
        vals: ndarray = np.full(sz, missing_val)
        ivl_df = (self.df.loc[:, ivl_cols] * fr).apply(np.ceil).astype(int)
        ivl_df[val_col] = self.df.loc[:, val_col]
        for (start, stop, val) in ivl_df.values:
            vals[start:stop] = val
        return vals
