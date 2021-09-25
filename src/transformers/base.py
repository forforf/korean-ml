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


    # def get_parameters(self):
    #     return vars(self)

    def fit(self, X, y=None, **kwargs):
        # TODO: use super fit
        return self

    def transform(self, X, y=None, **kwargs):
        self.log.warning('transform() from the base Transformer class called. This does nothing.')
        # TODO: use super transform
        return X.squeeze()