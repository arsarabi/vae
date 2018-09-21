"""TensorFlow layers"""

__all__ = ['dense']

import copy
import tensorflow as tf

from . import ops


xavier_initializer = tf.contrib.layers.variance_scaling_initializer(
    factor=1.0, mode='FAN_AVG'
)


def _dense(
    inputs, kernel, weight_normalization=False, init=False,
    bias_initializer=tf.zeros_initializer(), scale_init=1.0, bias_init=0.0
):
    """Dense layer with optional weight normalization

    Parameters
    ----------
    inputs : :obj:`tf.Tensor`
        Inputs to the layer.
    kernel : :obj:`tf.Tensor`
        Weights for the layer.
    weight_normalization : bool, optional (default=False)
        Whether to use weight normalization.
    init : bool, optional (default=False)
        Whether to perform data-dependent initialization of parameters.
    bias_initializer : optional (default=tf.zeros_initializer())
        Initializer for biases. Ignored when using data-dependent
        initialization with weight normalization.
    scale_init : :obj:`tf.Tensor`, optional (default=1.0)
        Extra scaling to apply to data-dependent initialization for
        weight normalization.
    bias_init : :obj:`tf.Tensor`, optional (default=0.0)
        Extra bias to apply to data-dependent initialization for weight
        normalization.

    Returns
    -------
    outputs : :obj:`tf.Tensor`
        Outputs from the layer.

    """
    def get_var(name, shape=None, initializer=None):
        if isinstance(initializer, tf.Tensor):
            return tf.get_variable(name, shape=None, initializer=initializer)
        else:
            return tf.get_variable(name, shape=shape, initializer=initializer)

    input_shape = inputs.get_shape().as_list()
    input_ndims = len(input_shape)

    kernel_shape = kernel.get_shape().as_list()
    n_outputs = kernel_shape[-1]
    bias_shape = [n_outputs]

    if weight_normalization:
        name = 'WeightNormalization'
    else:
        name = 'Dense'
        bias = get_var('bias', shape=bias_shape, initializer=bias_initializer)

    with tf.variable_scope(name):
        if input_ndims > 2:
            shape = [-1, input_shape[-1]]
            batch_shape = tf.shape(inputs)[:-1]
            inputs = tf.reshape(inputs, shape)

        if weight_normalization and init:
            kernel_norm = tf.nn.l2_normalize(kernel, 0)
            outputs = tf.matmul(inputs, kernel_norm)

            m_init, v_init = tf.nn.moments(outputs, [0], keep_dims=False)
            scale_init = scale_init / tf.sqrt(v_init + 1e-6)
            bias_init = bias_init - m_init * scale_init

            g = get_var('g', initializer=scale_init)
            bias = get_var('bias', initializer=bias_init)
            outputs = scale_init * outputs + bias_init
        elif weight_normalization:
            g = get_var(
                'g', shape=bias_shape, initializer=tf.ones_initializer()
            )
            bias = get_var(
                'bias', shape=bias_shape, initializer=bias_initializer
            )

            kernel_norm = tf.sqrt(
                tf.reduce_sum(tf.square(kernel), axis=0, keep_dims=True)
            )
            scaler = g / tf.maximum(kernel_norm, 1e-6)
            outputs = scaler * tf.matmul(inputs, kernel) + bias
        else:
            outputs = tf.matmul(inputs, kernel) + bias

        if input_ndims > 2:
            shape = [batch_shape, [n_outputs]]
            outputs = tf.reshape(outputs, tf.concat(shape, axis=0))

    return outputs


def dense(
    inputs, units, activation=None,
    weight_normalization=False, init=False,
    kernel_initializer=xavier_initializer,
    bias_initializer=tf.zeros_initializer(), scale_init=1.0, bias_init=0.0,
    name='Dense', reuse=False
):
    """Dense layer with optional weight normalization

    Parameters
    ----------
    inputs : :obj:`tf.Tensor`
        Inputs to the layer.
    units : int
        Number of units (outputs).
    activation: optional (default=None)
        Activation function to use for the layer.
    weight_normalization : bool, optional (default=False)
        Whether to use weight normalization.
    init : bool, optional (default=False)
        Whether to perform data-dependent initialization of parameters.
    kernel_initializer : optional
        Initializer for weights. Defaults to Xavier initialization.
    bias_initializer : optional (default=tf.zeros_initializer())
        Initializer for biases. Ignored when using data-dependent
        initialization with weight normalization.
    scale_init : :obj:`tf.Tensor`, optional (default=1.0)
        Extra scaling to apply to data-dependent initialization for
        weight normalization.
    bias_init : :obj:`tf.Tensor`, optional (default=0.0)
        Extra bias to apply to data-dependent initialization for weight
        normalization.
    name : optional (default='Dense')
        Name of the layer.
    reuse : bool, optional (default=False)
        Whether to reuse variables.

    Returns
    -------
    outputs : :obj:`tf.Tensor`
        Outputs from the layer.

    """
    input_shape = inputs.get_shape().as_list()
    kernel_shape = [input_shape[-1], units]
    with tf.variable_scope(name, reuse=reuse):
        kernel = tf.get_variable(
            'V' if weight_normalization else 'kernel',
            shape=kernel_shape, initializer=kernel_initializer
        )
        if init:
            kernel = kernel.initialized_value()

        outputs = _dense(
            inputs, kernel,
            weight_normalization=weight_normalization, init=init,
            bias_initializer=bias_initializer,
            scale_init=scale_init, bias_init=bias_init
        )

        if activation is not None:
            outputs = activation(outputs)

    return outputs
