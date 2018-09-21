=============================
Variational Autoencoder (VAE)
=============================

.. image:: https://readthedocs.org/projects/vae/badge/?version=latest
    :target: http://vae.readthedocs.io

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: ./LICENSE

.. sphinx-start

Overview
========

This package contains an implementation of a `variational autoencoder`_ in
TensorFlow, with optional `importance weighting`_, and `weight normalization`_.
Trained model can be saved and then restored for evaluation.

Installation
============

Install using:

.. code-block:: sh

    python setup.py install

Usage
=====

The following example shows how to instantiate and train a VAE model, save the
trained model, and then load it for evaluating samples.

First we instantitate a ``VAE`` object and use some training data (a NumPy
array or a SciPy sparse matrix containing real-valued/binary observations) to
train the model.

.. code-block:: python

    import dill
    import vae

    model = vae.VAE(
        n_inputs=train_data.shape[1],
        n_latent=2,
        n_encoder=[1000, 1000],
        n_decoder=[1000, 1000],
        visible_type='binary',
        nonlinearity=tf.nn.relu,
        weight_normalization=True,
        importance_weighting=False,
        optimizer='Adam',
        learning_rate=0.001,
        model_dir='vae'
    )

    with open('vae/model.pkl', 'wb') as f:
        dill.dump(model, f)

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=10,
        shuffle=True,
        summary_steps=100,
        init_feed_dict={'batch_size': 1000},
        batch_size=100,
        n_samples=10
    )

Note that ``VAE`` object can be serialized using ``dill``, however separate
TensorFlow checkpoint files are created after training each epoch in the
provided directory for saving the trained weights/biases.

One can also monitor training using TensorBoard:

.. code-block:: sh

    tensorboard --logdir vae

We can then restore the trained model and use it to evaluate samples:

.. code-block:: python

    with open('vae/model.pkl', 'wb') as f:
        model = dill.load(f)

    Z_mean, Z_sd = model.evaluate(X, ['z_mean', 'z_sd'], n_samples=1)

See the documentation for the full list of variables that can be evaluated.

.. _Variational Autoencoder: https://arxiv.org/abs/1312.6114
.. _Importance Weighting: https://arxiv.org/abs/1509.00519
.. _Weight Normalization: https://arxiv.org/abs/1602.07868
.. _Dropout: https://arxiv.org/abs/1207.0580
