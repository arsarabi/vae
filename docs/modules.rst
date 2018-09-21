=================
API Documentation
=================

Core classes
============

.. automodule:: vae

.. currentmodule:: vae

.. autosummary::
    :toctree: generated
    :template: class.rst

    Dataset
    Model
    UnsupervisedModel
    VAE

Layers
======

.. automodule:: vae.layers

.. currentmodule:: vae

.. autosummary::
    :toctree: generated
    :template: function.rst

    layers.dense

Operations
==========

.. automodule:: vae.ops

.. currentmodule:: vae

.. autosummary::
    :toctree: generated
    :template: function.rst

    ops.gaussian_log_likelihood
    ops.log_eluplusone
    ops.reduce_logmeanexp
