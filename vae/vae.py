import tensorflow as tf

from . import dataset, layers, ops
from .globals import ENQUEUE_BATCH_SIZE
from .unsupervisedmodel import _concat, UnsupervisedModel


VISIBLE_TYPES = ['binary', 'real']


class VAE(UnsupervisedModel):
    """Variational Autoencoder model

    Parameters
    ----------
    n_inputs : int
        Number of input (visible) variables.
    n_latent : int
        Number of latent variables.
    n_encoder : list of int, optional (default=[])
        Number of hidden units for the encoders.
    n_decoder : list of int, optional (default=[])
        Number of hidden units for the decoders.
    visible_type : str, optional (default='binary'):
        Type of the visible variables. Can be 'binary' or 'real'.
    dropout_rate : float or None, optional (default=None)
        Dropout rate, if any, to apply to the input.
    nonlinearity : optional (default=tf.nn.relu)
        Nonlinearity to use for hidden units.
    weight_normalization : bool, optional (default=False)
        Whether to use weight normalization.
    importance_weighting : bool, optional (default=False)
        Whether to use importance weighting for training the model.
    min_divergence : float, optional (default=0.0)
        Minimum KL divergence per latent dimension. Default=0.0.
    optimizer : str, optional (default='Adam')
        Optimizer to use for training the model. See
        ``tf.contrib.layers.OPTIMIZER_CLS_NAMES`` for available options.
    learning_rate : float, optional (default=0.001)
        Learning rate for training the model.
    learning_rate_decay_fn : optional (default=None)
        Function for decaying the learning rate. Takes `learning_rate`
        and `global_step`, and return the decayed learning rate.
    clip_gradients : float or None, optional (default=None)
        If provided, global clipping is applied to prevent the gradient
        norms from exceeding the provided value.
    model_dir : str or None, optional (default=None)
        Path to the model directory. Defaults to the current working
        directory.
    debug : bool, optional (default=False):
        Whether to open the TensorFlow session in debug mode.

    """

    def __init__(
        self, n_inputs, n_latent, n_encoder=[], n_decoder=[],
        visible_type='binary', dropout_rate=None, nonlinearity=tf.nn.relu,
        weight_normalization=False, importance_weighting=False,
        min_divergence=0.0, **kwargs
    ):
        if visible_type not in VISIBLE_TYPES:
            raise ValueError(
                'Invalid value for visible_unit_type. Available options are {}'
                .format(UNIT_TYPE_OPTIONS)
            )

        self._params.update({
            'n_latent': n_latent,
            'n_encoder': n_encoder,
            'n_decoder': n_decoder,
            'visible_type': visible_type,
            'dropout_rate': dropout_rate,
            'nonlinearity': nonlinearity,
            'weight_normalization': weight_normalization,
            'importance_weighting': importance_weighting,
            'min_divergence': min_divergence
        })
        super().__init__(n_inputs, **kwargs)

    def _create_placeholders(self):
        """Create the TensorFlow placeholders for the model"""
        super()._create_placeholders()
        n_samples = tf.placeholder(tf.int32, shape=[], name='nSamples')
        self._placeholders['n_samples'] = n_samples
        if self.dropout_rate is not None:
            self._placeholders['training'] = tf.placeholder(
                tf.bool, shape=[], name='isTraining'
            )
        if self.weight_normalization:
            self._placeholders['inputs_init'] = tf.placeholder(
                tf.float32, shape=[None, self.n_inputs], name='Inputs-Init'
            )

    def _build_recognition_layer(self, x, init=False):
        """Create a stochastic recognition layer

        Parameters
        ----------
        x : :obj:`tf.Tensor`
            Visible variables.
        init : bool , optional (default=False)
            Whether to perform data-dependent initialization of
            parameters when using weight normalization.

        Returns
        -------
        z_moments : tuple of :obj:`tf.Tensor`
            Moment tensors (mean, standard devitation, and log(SD)) for
            latent variables.
        z : :obj:`tf.Tensor`
            Sampled latent variables.
        log_q_z : :obj:`tf.Tensor` or None
            If `init` is True, a tensor containing the log-likelihood
            ``log[q(z|x)]``, None otherwise.

        """
        initializer = tf.contrib.layers.variance_scaling_initializer
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN'
        )
        reuse = self.weight_normalization and not init

        # Apply dropout
        if self.dropout_rate is None or init:
            inputs = x
        else:
            training = self._placeholders['training']
            inputs = tf.layers.dropout(
                x, rate=self.dropout_rate, training=training, name='Dropout'
            )

        # Build the encoder MLP
        with tf.variable_scope('Encoder'):
            for i, n_output in enumerate(self.n_encoder):
                inputs = layers.dense(
                    inputs, n_output, activation=self.nonlinearity,
                    weight_normalization=self.weight_normalization, init=init,
                    kernel_initializer=kernel_initializer,
                    name='Encoder-{}'.format(i + 1), reuse=reuse
                )

            h = inputs

        # Create moments (mean and standard deviation) for q(z|x)
        z_mean = layers.dense(
            h, self.n_latent,
            weight_normalization=self.weight_normalization,
            init=init,
            kernel_initializer=kernel_initializer,
            #scale_initializer=tf.constant_initializer(0.1),
            name='z-Mean', reuse=reuse
        )

        with tf.variable_scope('z-SD'):
            z_sd_pre_activation = layers.dense(
                h, self.n_latent,
                weight_normalization=self.weight_normalization, init=init,
                kernel_initializer=kernel_initializer,
                name='z-SD-PreActivation', reuse=reuse
            )
            z_log_sd = ops.log_eluplusone(z_sd_pre_activation, name='z-LogSD')
            z_sd = tf.add(1.0, tf.nn.elu(z_sd_pre_activation), name='z-SD')

        # Sample latent variables
        with tf.name_scope('Reparameterization'):
            n_samples = self._placeholders['n_samples']
            shape = [tf.shape(z_mean)[0], n_samples, self.n_latent]
            epsilon = tf.random_normal(shape, name='epsilon')
            z = tf.add(
                tf.expand_dims(z_mean, 1), epsilon * tf.expand_dims(z_sd, 1),
                name='z'
            )

        # Compute log[q(z|x)] for the sampled latent variables
        if not init:
            with tf.variable_scope('Stats/z-Posterior-LogLikelihood'):
                log_q_z = ops.gaussian_log_likelihood(epsilon)
                log_q_z -= tf.reduce_sum(z_log_sd, axis=1, keep_dims=True)
        else:
            log_q_z = None

        return (z_mean, z_sd, z_log_sd), z, log_q_z

    def _build_generative_layer(self, z, x, init=False):
        """Create a stochastic generative layer

        Parameters
        ----------
        z : :obj:`tf.Tensor`
            Latent variables.
        x : :obj:`tf.Tensor`
            Visible variables (for computing the reconstruction error).
        init : bool, optional (default=False)
            Whether to perform data-dependent initialization of
            parameters when using weight normalization.

        Returns
        -------
        x_moments : tuple of :obj:`tf.Tensor`
            Moment tensors (mean, standard devitation, and log(SD)) for
            real-valued visible variables, or the mean for binary
            visible variables.
        log_p_x_given_z : :obj:`tf.Tensor` or None
            If `init` is True, a tensor containing the log-likelihood
            ``log[p(x|z)]``, None otherwise.

        """
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN'
        )
        reuse = self.weight_normalization and not init

        # Build the decoder MLP
        n_visible = x.get_shape().as_list()[-1]
        inputs = z
        with tf.variable_scope('Decoder'):
            for i, n_output in enumerate(self.n_decoder):
                inputs = layers.dense(
                    inputs, n_output, activation=self.nonlinearity,
                    weight_normalization=self.weight_normalization, init=init,
                    kernel_initializer=kernel_initializer,
                    name='Decoder-{}'.format(i + 1), reuse=reuse
                )

        # Create the required moments for p(x|z)
        if self.visible_type == 'real':
            mean = layers.dense(
                inputs, n_visible,
                init=init, kernel_initializer=kernel_initializer,
                name='x-Mean', reuse=reuse
            )

            with tf.variable_scope('x-SD'):
                sd_pre_activation = layers.dense(
                    inputs, n_visible,
                    init=init, kernel_initializer=kernel_initializer,
                    name='x-SD-PreActivation', reuse=reuse
                )
                log_sd = ops.log_eluplusone(sd_pre_activation, 'x-LogSD')
                sd = tf.add(1.0, tf.nn.elu(sd_pre_activation), name='x-SD')

            x_moments = (mean, sd, log_sd)
        else:
            mean_logits = layers.dense(
                inputs, n_visible,
                init=init, kernel_initializer=kernel_initializer,
                name='x-Mean-Logits', reuse=reuse
            )
            mean = tf.nn.sigmoid(mean_logits, name='x-Mean')
            x_moments = (mean,)

        # Compute log[p(x|z)] for the provided labels
        if not init:
            with tf.name_scope('Stats/x-Given-z-LogLikelihood'):
                n_samples = self._placeholders['n_samples']
                labels = tf.tile(tf.expand_dims(x, axis=1), [1, n_samples, 1])
                if self.visible_type == 'real':
                    log_p_x_given_z = ops.gaussian_log_likelihood(
                        labels, mean, variance
                    )
                else:
                    log_p_x_given_z = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels, logits=mean_logits
                    )
                    log_p_x_given_z = -tf.reduce_sum(log_p_x_given_z, axis=2)
        else:
            log_p_x_given_z = None

        return x_moments, log_p_x_given_z

    def _build_layers(self, inputs, init=False):
        """Create the recognition and generative layers

        Parameters
        ----------
        inputs : :obj:`tf.Tensor`
            Visible variables.
        init : bool, optional (default=False)
            Whether to perform data-dependent initialization of
            parameters when using weight normalization.

        Returns
        -------
        z_moments : tuple of :obj:`tf.Tensor`
            Moment tensors (mean, standard devitation, and log(SD)) for
            latent variables.
        z : :obj:`tf.Tensor`
            Sampled latent variables.
        log_q_z : :obj:`tf.Tensor` or None
            If `init` is True, a tensor containing the log-likelihood
            ``log[q(z|x)]``, None otherwise.
        x_moments : tuple of :obj:`tf.Tensor`
            Moment tensors (mean, standard devitation, and log(SD)) for
            real-valued visible variables, or the mean for binary
            visible variables.
        log_p_x_given_z : :obj:`tf.Tensor` or None
            If `init` is True, a tensor containing the log-likelihood
            ``log[p(x|z)]``, None otherwise.

        """
        # Create the recognition model
        z_moments, z, log_q_z = self._build_recognition_layer(
            inputs, init=init
        )

        # Create the generative model
        x_moments, log_p_x_given_z = self._build_generative_layer(
            z, inputs, init=init
        )

        return z_moments, z, log_q_z, x_moments, log_p_x_given_z

    def _build_model(self):
        """Build the computational graph for the model"""
        if self.weight_normalization:
            # Perform data-dependent initialization of parameters
            with tf.name_scope('Initialization'):
                inputs_init = self._placeholders['inputs_init']
                self._build_layers(inputs_init, init=True)

        # Build the model and compute statistics
        z_moments, z, log_q_z, x_moments, log_p_x_given_z = self._build_layers(
            self._tensors['inputs'], init=False
        )
        z_mean, z_sd, z_log_sd = z_moments
        z_mean, z_sd, z_log_sd = z_moments
        if self.visible_type == 'real':
            x_mean, x_sd, x_log_sd = x_moments
            x_mean_combined = tf.reduce_mean(
                x_mean, axis=1, name='x-Mean-Combined'
            )
            x_sd_combined = tf.reduce_mean(x_sd, axis=1, name='x-SD-Combined')
        else:
            x_mean, = x_moments
            x_mean_combined = tf.reduce_mean(
                x_mean, axis=1, name='x-Mean-Combined'
            )

        log_p_z = ops.gaussian_log_likelihood(
            z, name='Stats/z-Prior-LogLikelihood'
        )

        with tf.name_scope('Stats/ReconstructionError'):
            log_p_x_given_z_combined = ops.reduce_logmeanexp(
                log_p_x_given_z, axis=1
            )
            reconstruction_error = -tf.reduce_mean(log_p_x_given_z_combined)

        with tf.name_scope('Stats/NLL'):
            log_p_x = log_p_x_given_z + log_p_z - log_q_z
            log_p_x_combined = ops.reduce_logmeanexp(log_p_x, axis=1)
            nll = -tf.reduce_mean(log_p_x_combined)

        # Create the training operation
        with tf.name_scope('Training'):
            if self.importance_weighting:
                elbo = log_p_x_given_z + log_p_z - log_q_z
                elbo = tf.reduce_mean(ops.reduce_logmeanexp(elbo, axis=1))
                cost = tf.negative(elbo, name='Cost')
            else:
                with tf.name_scope('KLDivergence'):
                    z_variance = tf.square(z_sd)
                    kl_divergence = -0.5 * tf.reduce_mean(
                        1.0 - tf.square(z_mean) + 2 * z_log_sd - z_variance,
                        axis=0
                    )
                    kl_divergence = tf.reduce_sum(
                        tf.maximum(kl_divergence, self.min_divergence)
                    )

                elbo = tf.reduce_mean(log_p_x_given_z) - kl_divergence
                cost = tf.negative(elbo, name='Cost')

        self._create_train_op(cost)

        # Create summaries and training/validation statistics
        with tf.name_scope('Training/'):
            tf.summary.scalar('ReconstructionError', reconstruction_error)
            tf.summary.scalar('NLL', nll)

        self._training_stats = self._validation_stats = [
            'NLL', 'Reconstruction Error'
        ]

        # Store output tensors
        self._tensors.update({
            'z_mean': z_mean, 'z_sd': z_sd,
            'x_mean': x_mean, 'x_mean_combined': x_mean_combined,
            'log_p_x_given_z': log_p_x_given_z,
            'log_p_x_given_z_combined': log_p_x_given_z_combined,
            'log_p_x': log_p_x, 'log_p_x_combined': log_p_x_combined,
            'NLL': nll, 'Reconstruction Error': reconstruction_error
        })
        if self.visible_type == 'real':
            self._tensors.update({
                'x_sd': x_sd, 'x_sd_combined': x_sd_combined
            })

    def evaluate(
        self, data, tensors=None, combine_func=_concat, enable_summaries=False,
        batch_size=100, n_samples=1, training=False
    ):
        """Evaluate the model on the provided data.

	Any of the following can be provided for the `tensors` argument:

        * **z_mean** : Mean of latent variables.
 	* **z_sd** : Standard deviation of latent variables.
        * **x_mean** : Mean of visible variables (reconstructions).
        * **x_sd** : Standard deviation of real-valued visible variables
	  (reconstructions).
        * **x_mean_combined** : Mean of visible variables
          (reconstructions), averaged over drawn samples.
        * **x_sd_combined** : Standard deviation of real-valued visible
	  variables (reconstructions), averaged over drawn samples.
        * **log_p_x_given_z**: ``log[p(x|z)]``.
        * **log_p_x_given_z_combined**: ``log[p(x|z)]``, combined (using
          :meth:`ops.reduce_logmeanexp`) over drawn samples.
        * **log_p_x**: ``log[p(x|z)]``.
        * **log_p_x_combined**: ``log[p(x)]``, combined (using
          :meth:`ops.reduce_logmeanexp`) over drawn samples.
        * **NLL**: Negative log-likelihood, i.e.,
          ``mean(log_p_x_combined)``.
        * **Reconstruction Error**: Reconstruction error, i.e.,
          ``mean(log_p_x_given_z_combined)``.

        Parameters
        ----------
        data
            Input data. Can be a NumPy array, a SciPy sparse matrix, or
            a :obj:`Dataset` object.
        tensors : list of str or None, optional (default=None)
            List of tensor names to collect. Defaults to all available
            tensors.
        combine_func : optional
            Function to use to combine collected data from all batches.
            If set to None, each batch is returned separately. Defaults
            to concatenation along the first axis.
        enable_summaries : bool, optional (default=False)
            Whether to write summaries to disk.
        batch_size : int, optional (default=100)
            Size of mini-batches.
        n_samples : int, optional (default=1)
            Number of samples for the Monte Carlo estimator, i.e.,
            number of samples to draw from the encoder.
        training : bool, optional (default=False)
            Whether to evaluate in training mode. Ignored if not using
            dropout.

        """
        feed_dict = {'batch_size': batch_size, 'n_samples': n_samples}
        if self.dropout_rate is not None:
            feed_dict['training'] = training

        return super().evaluate(
            data, tensors=tensors, combine_func=combine_func,
            enable_summaries=enable_summaries, **feed_dict
        )

    def fit(
        self, train_data, validation_data=None,
        epochs=1, shuffle=True, restore=False, summary_steps=None,
        init_feed_dict={}, validation_feed_dict={},
        batch_size=100, n_samples=1
    ):
        """Train the model using the provided training data

        Parameters
        ----------
        train_data
            Training data. Can be a NumPy array, a SciPy sparse matrix,
            or a :obj:`Dataset` object.
        validation_data : optional (default=None)
            Validation data. Can be a NumPy array, a SciPy sparse
            matrix, or a :obj:`Dataset` object.
        epochs : int (default=1)
            Number of training epochs.
        shuffle : bool, optional (default=True)
            Whether to shuffle training data before each epoch. Ignored
            if `training data` is a :obj:`Dataset` object.
        restore : bool, optional (default=False)
            Whether to restore the model from a previous checkpoint.
        summary_steps : int or None, optional (default=None)
            Number of steps between writing summaries, or None for
            disabling summaries.
        init_feed_dict : dict, optional (default={})
            Feed dictionary to append for initialization.
        validation_feed_dict : dict, optional (default={})
            Feed dictionary to append for validation.
        batch_size : int, optional (default=100)
            Size of mini-batches.
        n_samples : int, optional (default=1)
            Number of samples for the Monte Carlo estimator, i.e.,
            number of samples to draw from the encoder.

        Returns
        -------
        self

        """
        if not isinstance(train_data, dataset.Dataset):
            train_data = dataset.Dataset(
                train_data, batch_size=ENQUEUE_BATCH_SIZE, shuffle=shuffle
            )

        training_feed_dict = {'batch_size': batch_size, 'n_samples': n_samples}
        if self.dropout_rate is not None:
            training_feed_dict['training'] = True
            validation_feed_dict['training'] = False
        if self.weight_normalization:
            init_batch_size = init_feed_dict.get('batch_size', batch_size)
            inputs_init = train_data.__next__(batch_size=init_batch_size)
            init_feed_dict['inputs_init'] = inputs_init
            train_data.reset()

        return super().fit(
            train_data, validation_data=validation_data,
            epochs=epochs, shuffle=shuffle,
            restore=restore, summary_steps=summary_steps,
            init_feed_dict=init_feed_dict,
            validation_feed_dict=validation_feed_dict, **training_feed_dict
        )
