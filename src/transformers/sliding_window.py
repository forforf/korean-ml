import math
import numpy as np
# noinspection PyProtectedMember
from numpy.lib.stride_tricks import sliding_window_view

from src.transformers.base import Transformer


class SlidingWindow(Transformer):
    """
    Transformer for Generating Sliding Window feature sets

    given an X and y, will take n samples of X (ordered series)
    Example, for len(X) = len(y) = 1000, and a frame_length=5 and y_offset=4
     y3, [x0, x1, x2, x3, x4]
     y4, [x1, x2, x3, x4, x5]
     y5, [x2, x3, x4, x5, x6]
     ...
     y999, [x996, x997, x998, x999, x1000]

     Note that the new X,y lengths is (offset - 1) the length of the original data if left unpadded
    """

    # TODO: maybe allow padding, but passing the various pad arguments is kind of a pain
    # TODD: PADDING IS BROKEN -- INVESTIGATE AND FIX
    def __init__(self, window=1, offset_percent=0.0, pad_X=True, axis=0):
        super().__init__()
        self.log.setLevel('INFO')
        self.log.debug(f'window: {window}, offset_percent: {offset_percent}')
        self.window = window
        self.offset_percent = offset_percent
        self.offset = math.floor(self.offset_percent*self.window)
        self.X = None
        self.y = None
        self.pad_X = pad_X
        # TODO: Deprecate axis
        self.axis = axis
        # self.slicer will be used for slicing the X during the transform (after fitting)
        # X[:,:] is equivalent to X[slice(None), slice(None)]
        # whih is what we get if axis=1.
        self.slicer = [slice(None)] * (self.axis + 1)
        # TODO: flag to check if we've ran fit. This is a hack, try to find a better way
        self.fitted = False

    def __eq__(self, o):
        if not isinstance(o, SlidingWindow):
            return False

        are_same = [
            self.window == o.window,
            self.offset_percent == o.offset_percent,
            np.array_equal(self.X, o.X),
            np.array_equal(self.y, o.y),
            self.pad_X == o.pad_X,
            self.axis == o.axis
        ]
        return all(are_same)

    def __ne__(self, o):
        return not self == o

    def fit(self, X, y=None, axis=0):
        """
        For fitting we need to create the windowed features for X and
        make sure the y values align with the proper windowed feature row

        If no y, then we just generate the windowed features
        :param X: Un-windowed (flat) series data
        :param y: labels for X (optional)
        :param axis: [Might get deprecated] axis to window on if X is multi-dimensional
        :return: self
        """
        self.offset = math.floor(self.offset_percent*self.window)
        self.log.debug(f'Fit: window: {self.window}, offset_percent: {self.offset_percent}, offset: {self.offset}')
        self.log.debug(f'Initial X shape: {X.shape}')
        # Note: If unpadded this will change the shape of x, reduces rows (axis=0) by self.window - 1
        # Also: For cases where offset > window, a padded y would be needed to fully cover the padded x
        #       However, we are not doing that here (as guessing at y's seems a bit dangerous)

        x_ = self._fit_x(X)
        self.log.info(f'np sliding window shape: {x_.shape}')

        if y is not None:
            self.y, self.slicer = self._fit_y(x_, y, self.slicer)

        self.X = x_[tuple(self.slicer)]
        self.log.debug(f'Final X shape: {self.X.shape}')
        # per typical fit() contract, return self
        self.fitted = True
        return self

    def _fit_x(self, X):
        # TODO: Figure out the shape we want (1, n) or (n,)
        pad_size = (self.offset, self.window+self.offset-1)
        # flatten to (n,) then pad, then reshape back to (1, n)
        self.log.debug(f'Pad Size: {pad_size}')
        padded_X = np.pad(X.squeeze(), pad_size, 'edge')
        self.log.debug(f'Padded X: {padded_X.shape}')
        return sliding_window_view(padded_X, self.window, axis=len(padded_X.shape)-1)

    # returns (y, slicer) The slicer ensures that the X data and y data are the same size
    #   in the future, maybe allow it to be an option
    def _fit_y(self, x_, y, slc_a):
        self.log.debug(f'Initial y shape: {y.shape}')
        # since we padded the X data, we don't need to offset the y data, so we can start the index at 0
        y_ = y.squeeze()[0:len(x_.squeeze())]
        # Because we've potentially changed y, we need to make sure the final X result
        # is shape consistent with the new y.
        assert len(x_.squeeze()) >= len(y_), f'x < y, this should not happen becuase we should have padded x already'
        # slc_a is initially set to the equivalent of [:,:, ... (axis + 1)] (i.e., axis=1 -> [:, :])
        # we basically make it [len(y_), :, ... (axis+1)]
        slc_a[0] = slice(0, len(y_))
        self.log.debug(f'final y shape: {y_.shape}')
        return y_, slc_a

    def transform(self, X, y=None):
        self.log.debug(f'Transform shapes, X: {X.shape}, y:{y.shape if y is not None else "--"}')
        # return self.X if np.array_equal(X, self.X) else self._fit_x(X)
        # TODO: slicing should be a function of offset and window, not blindly lopping off the end
        x_fit = self._fit_x(X)
        self.log.debug(f'X shape: {X.shape}, fitted shape: {x_fit.shape}')
        # TODO: think about if there's a better way (transpose, squeeze, etc), also get rid of max,
        #      basically we need to have a elegant way to handle trivial dimensions (dimensions that can be squeezed)
        #      that keep the proper shapes for down stream consumers
        # My stupid hack breaks if the offset is larger than window size
        if self.window <= self.offset:
            self.log.warning(
                f'Window size ({self.window}) is less than offset ({self.offset}). This may lead to bad results'
            )
        if max(X.shape) != max(x_fit.shape):
            orig_shape = x_fit.shape
            x_fit_sqz = x_fit.squeeze()
            x_fit = x_fit_sqz[self.window-self.offset:len(X.squeeze())+self.window-self.offset]
        self.log.debug(f'X shape: {X.shape}, fitted shape: {x_fit.shape}')
        return x_fit