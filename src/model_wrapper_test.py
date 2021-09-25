import numpy as np

from src.model_wrapper import KerasTransform,

# Loader
# Given a full path, load model and transformer


class TransformerExample:
    def __init__(self, adder):
        self.adder = adder
        self.X = np.zeros((3, 1))
        self.y = np.zeros(3)

    def fit_transform(self, X, y):
        self.X = X + self.adder
        self.y = y + self.adder


class TestKerasTransform:

    def test_default_transform(self):
        feat_size = 5
        x_shape = (feat_size, 1)
        x_transform_shape = (feat_size, 1, 1)
        X = np.ones(x_shape)
        y = np.ones(feat_size)

        wrapper = KerasTransform()

        X_t, y_t = wrapper.transform(X, y)
        expected_X = X.reshape(x_transform_shape)

        np.testing.assert_array_equal(X_t, expected_X)
        np.testing.assert_array_equal(y_t, y)

    def test_transform(self):
        feat_size = 5
        x_shape = (feat_size, 1)
        x_transform_shape = (feat_size, 1, 1)
        X = np.ones(x_shape)
        y = np.ones(feat_size) + 1
        transform_adder = 2

        transformer = TransformerExample(transform_adder)
        wrapper = KerasTransform(transformer)

        X_t, y_t = wrapper.transform(X, y)

        expected_X = (X + transformer.adder).reshape(x_transform_shape)
        expected_y = y + transformer.adder
        np.testing.assert_array_equal(X_t, expected_X)
        np.testing.assert_array_equal(y_t, expected_y)

