""""TensorFlow operations"""

__all__ = [
    'gaussian_log_likelihood',
    'log_eluplusone',
    'reduce_logmeanexp'
]

import numpy as np
import tensorflow as tf


def gaussian_log_likelihood(x, mean=0.0, variance=1.0, axis=-1, name=None):
    """Compute the log-likelihood for independent Gaussian variables

    Parameters
    ----------
    x : :obj:`tf.Tensor`
        Gaussian observations.
    mean : :obj:`tf.Tensor`, optional (default=0.0)
        Mean of the Gaussian variables.
    variance : :obj:`tf.Tensor`, optional (default=1.0)
        Variance of the Gaussian variables.
    axis : int, optional (default=-1)
        Dimension containing the Gaussian variables.
    name : str or None, optional (default=None)
        Name of the operation. Defaults to 'GaussianLogLikelihood'.

    Returns
    -------
    log_likelihood : :obj:`tf.Tensor`

    """
    log2pi = np.log(2.0 * np.pi)
    n_components = x.get_shape().as_list()[axis]
    with tf.name_scope(name, 'GaussianLogLikelihood'):
        x_centered = x
        if mean != 0.0:
            x_centered -= mean

        log_likelihood = -0.5 * log2pi * n_components
        if variance == 1.0:
            log_likelihood -= 0.5 * tf.reduce_sum(
                tf.square(x_centered), axis=axis
            )
        else:
            log_likelihood -= 0.5 * tf.reduce_sum(
                tf.square(x_centered) / variance + tf.log(variance), axis=axis
            )

    return log_likelihood


def log_eluplusone(x, name=None):
    """Compute ``log(elu(x) + 1)`` in a numerically stable manner

    Parameters
    ----------
    x : :obj:`tf.Tensor`
    name : str or None, optional (default=None)
        Name of the operation. Defaults to 'LogOnePlusElu'.

    Returns
    -------
    y : :obj:`tf.Tensor`

    """
    with tf.name_scope(name, 'LogEluPlusOne'):
        return tf.minimum(x, 0.0) + tf.log(tf.nn.relu(x) + 1.0)


def reduce_logmeanexp(x, axis=None, keep_dims=False, name=None):
    """Compute ``log(sum(exp(x)))`` in a numerically stable manner

    Parameters
    ----------
    x : :obj:`tf.Tensor`
    axis : int or list of int or None, optional (default=None)
        A single, or a list of dimensions to reduce. If not provided,
        reduces all dimensions.
    keep_dims : bool, optional (default=None)
        If True, retains reduced dimensions with length one.
    name : str or None, optional (default=None)
        Name of the operation. Defaults to 'ReduceLogMeanExp'.

    Returns
    -------
    y : :obj:`tf.Tensor`

    """
    with tf.name_scope(name, 'ReduceLogMeanExp'):
        x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
        y = x_max + tf.log(
            tf.reduce_mean(tf.exp(x - x_max), axis=axis, keep_dims=True)
        )
        if not keep_dims:
            y = tf.squeeze(y, axis=axis)

    return y
