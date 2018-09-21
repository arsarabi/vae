"""Implementation of a variational autoencoder in TensorFlow"""

__all__ = ['Dataset', 'Model', 'UnsupervisedModel', 'VAE']
__version__ = '0.1.0'

from .dataset import Dataset
from .model import Model
from .unsupervisedmodel import UnsupervisedModel
from .vae import VAE
