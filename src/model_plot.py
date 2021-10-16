import numpy as np
import matplotlib.pyplot as plt


class ModelPlot:

    @staticmethod
    def pred_threshold(pred_vals, thresh=0.5):
        return np.where(pred_vals > thresh, True, False)

    @staticmethod
    def align(y, offset):
        return np.pad(y, (offset, 0), 'minimum')

    @staticmethod
    def delta(y1, y2, y1_offset, y2_offset):
        min_len = min(len(y1), len(y2))
        return np.logical_xor(ModelPlot.align(y1, y1_offset)[0:min_len], ModelPlot.align(y2, y2_offset)[0:min_len])

    @staticmethod
    def speech_base_delta(y_pred, y, offset):
        return ModelPlot.delta(y_pred, y, offset, 0)

    # noinspection DuplicatedCode
    @staticmethod
    def plot(X, y, y_pred, x_rms, sliding_window_transformer):

        # Reshape to 2d in case the transformer is not doing a sliding window
        # This is equivalent to a sliding window with a window of 1
        x_sw = sliding_window_transformer.transform(X).astype(float).reshape(X.shape[0],-1)

        # cast values to float so we have a common type to deal with
        X = X.astype(float)
        y = y.squeeze().astype(float)
        y_pred = y_pred.squeeze().astype(float)
        x_rms = x_rms.astype(float)

        sw_mean = np.mean(x_sw, axis=1).squeeze()
        y_diff = y_pred - y

        fig, axs = plt.subplots(6, 1, figsize=(18, 20))
        axs[0].set_title(f'Mean of Sliding Window')
        axs[0].plot(X, color='slategray', alpha=0.4, label='X')
        axs[0].plot(X - x_rms, color='cyan', alpha=0.4, label='X-X_rms')
        axs[0].plot(sw_mean, color='lightblue', alpha=0.6, label='sliding window mean (txfm X)')
        axs[0].legend(loc='center right')

        axs[1].set_title(f'Speech/No Speech (manually identified)')
        axs[1].plot(x_rms, color='slategray', alpha=0.4, label='rms of audio')
        axs[1].plot(y, color='lightblue', alpha=0.8, label='1: speech, 0: no speech')
        axs[1].legend(loc='center right')

        axs[2].set_title(f'Predicted vs Input')
        axs[2].plot(y, color='lightblue', alpha=0.8, label='actual 1->speech, 0->no speech')
        axs[2].plot(y_pred, color='cyan', alpha=0.6, label='prediction')
        axs[2].legend(loc='center right')

        axs[3].set_title(f'Prediction Difference to base Speech/No Speech Input')
        axs[3].plot(x_rms, color='slategray', alpha=0.3, label='rms of audio')
        axs[3].plot(np.zeros(len(y)), color='gray', alpha=0.8, label='zero line')
        axs[3].plot(y_diff, color='lightblue', alpha=0.5, label='y_pred - y')
        axs[3].legend(loc='lower right')

        axs[4].set_title(f'Boolean Prediction')
        axs[4].plot(x_rms, color='slategray', alpha=0.3, label='rms of audio')
        axs[4].plot(1 * ModelPlot.pred_threshold(y, 0.5), color='lightblue', alpha=0.4, label='manual')
        axs[4].legend(loc='center right')

        axs[5].set_title(f'Prediction Difference magnitude')
        axs[5].plot(y_diff, color='thistle', alpha=0.5, label='pred vals - base')
        axs[5].axhspan(-0.5, 0.5, color='lightblue', alpha=0.3, label='0.5 thresh'),
        axs[5].axhspan(-0.75, 0.75, color='slategray', alpha=0.3, label='0.75 thresh')
        axs[5].legend(loc='upper right')

        plt.tight_layout()
        plt.show()
