# import math
# import wave
# import librosa
# import numpy as np
# from numpy.lib.stride_tricks import sliding_window_view
# from audio import Audio
# from log import Log
#
#
# # Handles time to samples conversion so it is consistent for all instances
# class AudioPlexer:
#
#     # Will verify the wav_files all have the specified sample rate
#     # if sample rate is not provided, use the first wav file's sample rate.
#
#     # PERFORMANCE NOTE: We're storing the raw wav file
#     def __init__(self, audio_paths, sr=None, n_fft=256, n_hops=4, sliding_window_size=64, sliding_offset=0):
#         self.log = Log.set(self.__class__.__name__)
#         self.audio_paths = audio_paths
#         self.sr = sr
#         self.n_fft = n_fft
#         self.n_hops = n_hops
#         self.hop_length = math.ceil(self.n_fft/self.n_hops)
#         self.sliding_window_size = sliding_window_size
#         self.sliding_offset = sliding_offset
#
#         # validate all files have the same sample rate
#         for fn in self.audio_paths:
#             with wave.open(fn, 'rb') as f:
#                 sr_ = f.getframerate()
#                 if self.sr is None:
#                     self.sr = sr_
#                 assert self.sr == sr_
#
#         self.audios = [Audio(f, sr=self.sr, n_fft=self.n_fft, n_hops=n_hops) for f in self.audio_paths]
#         self.size = sum([a.size for a in self.audios])
#
#     # So brittle it hurts
#     def __eq__(self, other):
#         if isinstance(other, self.__class__):
#             audio_paths_eq = np.array_equal(self.audio_paths, other.audio_paths) if isinstance(self.audio_paths, np.ndarray) else self.audio_paths == other.audio_paths
#             return (audio_paths_eq and
#                     self.sr == other.sr and
#                     self.n_fft == other.n_fft and
#                     self.n_hops == other.n_hops and
#                     self.sliding_window_size == other.sliding_window_size and
#                     self.sliding_offset == other.sliding_offset)
#         else:
#             return False
#
#     def __ne__(self, other):
#         return not self.__eq__(other)
#
#     def wav(self):
#         return np.concatenate([a.wav for a in self.audios])
#
#     def mfcc(self):
#         return np.concatenate([a.mfcc for a in self.audios], axis=1)
#
#     def rms(self):
#         return np.concatenate([a.rms for a in self.audios], axis=1)
#
#     # Return numpy array matching audio size
#     def val_from_interval(self, df, ivl_cols=None, val_col='value', missing_val=None):
#         if ivl_cols is None:
#             ivl_cols = ['start', 'stop']
#         a_dfs = [self.df_per_audio(df, a.path, val_col=val_col) for a in self.audios]
#         kw = {'ivl_cols': ivl_cols, 'missing_val': missing_val}
#         return np.concatenate([a.val_from_interval(a_df, **kw) for (a_df, a) in zip(a_dfs, self.audios)])
#
#     # TODO: Align with audio class, on fix for speech boolean test.
#     def speech_from_interval(self, df, ivl_cols=None, missing_val=None):
#         if ivl_cols is None:
#             ivl_cols = ['start', 'stop']
#         a_dfs = [df.loc[df['audio'] == a.path] for a in self.audios]
#         kw = {'ivl_cols': ivl_cols, 'missing_val': missing_val}
#         return np.concatenate([a.speech_from_interval(a_df, **kw) for (a_df, a) in zip(a_dfs, self.audios)])
#
#
#     def df_per_audio(self, full_a_df, audio_path, val_col='syl'):
#         df_for_audio_path = full_a_df.loc[full_a_df['audio'] == audio_path]
#         df_for_audio_path.reindex(['start', 'stop', val_col, 'audio'])
#         # TODO: Figure out how to pass the condition as an argument
#         df_for_audio_path['value'] = df_for_audio_path[val_col] != '0'
#         return df_for_audio_path
#
#     def to_rms(self, x):
#         rms = librosa.feature.rms(y=x, frame_length=self.n_fft, hop_length=self.hop_length)
#         return librosa.util.normalize(rms, axis=1)
#
#     # given an X and y, will take n samples of X (ordered series)
#     # Example, for len(X) = len(y) = 1000, and a frame_length=5 and y_offset=4
#     #  y3, [x0, x1, x2, x3, x4]
#     #  y4, [x1, x2, x3, x4, x5]
#     #  y5, [x2, x3, x4, x5, x6]
#     #  ...
#     #  y999, [x996, x997, x998, x999, x1000]
#     def sliding_window_features(self, X, y=None, window=None, offset=None):
#         if window is None:
#             window = self.sliding_window_size
#         if offset is None:
#             offset = self.sliding_offset
#
#         axis = len(X.shape) - 1
#         # we will need to reduce (by slicing) X to match y when we are done
#         slc_a = [slice(None)] * (axis + 1)
#         x_ = sliding_window_view(X, window, axis=axis)
#
#         y_ = None
#
#         # match y with the correct x
#         if y is not None:
#             y_ = y.flatten()[offset:x_.shape[axis]]
#             slc_a[axis] = slice(0, y_.shape[0])
#
#         slc = tuple(slc_a)
#         # return same shape
#         return x_[slc], y_
