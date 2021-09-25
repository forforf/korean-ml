import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def bconvolve_score(a1, a2):
    pn1 = np.where(a1>0, 1, -1)
    pn2 = np.where(a2>0, 1, -1)
    l = max(len(a1), len(a2))
    return np.max(np.convolve(pn1, pn2, mode='valid')) / l


def int_diff(a):
    """
    Make a new array of the same size, where each element is the
    difference between the preceding element and the current element.
    For example:
     [0,0,1,1,1,0,0,0,2,3,-2,-3] -> [0,0,1,0,0,-1,0,0,2,1,-5,1]
    :param a: input array
    :return: array of differences
    """
    return np.diff(a.astype(int), prepend=0)


def event_ticker(a):
    """
    Takes an "event" array (see param). If an event occurs it will increment a counter that will increase for
    positive events or decrease for negative events. The counter will continue to increment (or decrement) until
    a new event occurs. Note that repeated identical events has no effect (i.e., it results in the same values as
    if no event occurred).
    :param a: "event" array. A value of 0 indicates no event. Events can be either positive or negative (typically
               -1 or 1 though).
    :return: same size array that provides a "history" of event durations.
    """
    direction = -1
    counter = 0
    ctr = np.zeros(len(a))
    for i, el in enumerate(a):
        # reset counter if event occurred and set direction
        if el != 0:
            direction = el
        else:
            counter += direction

        ctr[i] = counter
    return ctr


def ticker_r2(a1, a2):
    b1 = int_diff(a1)
    b2 = int_diff(a2)
    ev1 = event_ticker(b1)
    ev2 = event_ticker(b2)
    return r2_score_wrapper(ev1, ev2)


def ticker_mse(a1, a2):
    b1 = int_diff(a1)
    b2 = int_diff(a2)
    ev1 = event_ticker(b1)
    ev2 = event_ticker(b2)
    return mse_wrapper(ev1, ev2)


def r2_score_wrapper(a1, a2):
    s = r2_score(a1.astype(int), a2.astype(int))
    return s if s >= 0 else 0


def mse_wrapper(a1, a2):
    mse =  mean_squared_error(a1.astype(int), a2.astype(int))
    return sigmoid(-mse)


def conv_var(a1, a2, var_penalty=30):
    var_penalty = 30 # var_penalty default empirically chosen to give a decent sized penalty to differing variances
    var_diff = np.var(a1) - np.var(a2)
    # we apply a guassian to normalize the diff between 0 and 1. A diff of 0->1, A diff of ±inf->0
    var_factor = gaussian(var_penalty*var_diff)
    return bconvolve_score(a1, a2) * var_factor


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def gaussian(x):
    return np.exp(-np.power(x, 2.))


class ConvVar():

    def __init__(self, var_penalty=30):
        # var_penalty default empirically chosen to give a decent sized penalty to differing variances
        self.var_penalty = var_penalty

    def score(self, a1, a2):
        var_diff = np.var(a1) - np.var(a2)
        # we apply a guassian to normalize the diff between 0 and 1. A diff of 0->1, A diff of ±inf->0
        var_factor = gaussian(self.var_penalty * var_diff)
        return bconvolve_score(a1, a2) * var_factor
