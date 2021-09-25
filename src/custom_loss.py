import tensorflow as tf

def gaussian(x):
    return tf.exp(-tf.pow(x, 2.))


def bconvolve_tf(a1, a2):
    pn1 = tf.where(a1>0, 1, -1)
    pn2 = tf.where(a2>0, 1, -1)
    l = max(len(a1), len(a2))
    data   = tf.reshape(pn1, [1, int(pn1.shape[0]), 1], name='data')
    kernel = tf.reshape(pn2, [int(pn2.shape[0]), 1, 1], name='kernel')
    # print('data', data.dtype)
    # print('kernel', kernel.dtype)
    conv = tf.squeeze(tf.nn.conv1d(data, kernel, 1, 'VALID') / l)
    conv = tf.cast(conv, tf.float32)
    # print('conv b', conv.dtype)
    return  conv
    # return tf.max(np.convolve(pn1, pn2, mode='valid')) / l


def conv_var_tf(a1, a2, var_penalty=30):
    if not tf.is_tensor(a1):
        a1 = tf.constant(a1, dtype=tf.float64)
    if not tf.is_tensor(a2):
        a2 = tf.constant(a2, dtype=tf.float64)
    _, var1 = tf.nn.moments(a1, axes=[0])
    _, var2 = tf.nn.moments(a2, axes=[0])
    var_penalty = 30.0 # var_penalty default empirically chosen to give a decent sized penalty to differing variances
    var_diff = var1 - var2
    var_factor = gaussian(var_penalty * var_diff)
    # print('var_factor', var_factor)
    # print('bconvolve', bconvolve_tf(a1, a2))
    conv = bconvolve_tf(a1, a2) * var_factor
    # print('conv', conv)
    # tf.cast(conv, tf.float32)
    return conv


def conv_loss(y_true, y_pred):
    # print('loss y true', y_true.dtype)
    # print('loss y pred', y_pred.dtype)
    return 1.0 - conv_var_tf(y_true, y_pred)