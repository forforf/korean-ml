# noinspection PyProtectedMember
from sklearn.base import BaseEstimator, TransformerMixin

from src.log import Log


# TODO: The .transform method should technically return X, not self - but I need y transforms too.
class Transformer(BaseEstimator, TransformerMixin):
    """
    Base Transformer
      fit() is used for training. A pipeline call to fit() will call fit() then transform()
      transform() is used for predictions. A pipeline call to predict() will only call transform()
    """
    def __init__(self):
        self.log = Log.set(self.__class__.__name__)

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        self.log.warning('transform() from the base Transformer class called. This does nothing.')
        # TODO: use super transform
        return X.squeeze()


class NoOpTransformer(Transformer):

    def __init__(self):
        super().__init__()
        self.X = None
        self.y = None

    def fit(self, X, y=None, **kwargs):
        self.X = X
        if y is not None:
            self.y = y
        return self

    def transform(self, X, y=None, **kwargs):
        self.log.warning('transform() from the base Transformer class called. This does nothing.')
        return X.squeeze()

    def __eq__(self, o):
        return type(self) == type(o)