import logging
import numpy as np
import os
import tensorflow as tf

from . import dataset
from .globals import ENQUEUE_BATCH_SIZE
from .model import Model


def _concat(values):
    if np.ndim(values[0]):
        return np.concatenate(values)
    else:
        return np.asarray(values)


class UnsupervisedModel(Model):
    __doc__ = Model.__doc__.replace('models', 'unsupervised models')

    def _train_model(
        self, dataset, restore=False, metadata=False, summary_steps=None,
        init_feed_dict={}, train_feed_dict={}
    ):
        """Train the model for one epoch

        Parameters
        ----------
        dataset : :obj:`Dataset`
            Training data to feed to the model.
        restore : bool, optional (default=False)
            Whether to restore the model from a previously saved graph.
        metadata : bool, optional (default=False)
            Whether to collect and write runtime metadata on the first
            training batch.
        summary_steps : int or None, optional (default=None)
            Number of steps between writing summaries, or None for
            disabling summaries.
        init_feed_dict : dict, optional (default={})
            Feed dictionary to use for initialization.
        train_feed_dict : dict, optional (default={})
            Feed dictionary to use for training.

        Returns
        -------
        global_step : int
            Final global step for the trained model.

        """
        logger = logging.getLogger(__name__)
        model_file = os.path.join(self.model_dir, 'model.cpkt')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with self._graph.as_default():
            with tf.Session(config=config) as self._session:
                self._init_tf(
                    restore=restore,
                    enable_summaries=summary_steps is not None,
                    feed_dict=init_feed_dict
                )

                collect = [
                    self._tensors[name] for name in self._training_stats
                ]
                fetches = {
                    'operation': self._operations['train'],
                    'summary': self._merged_summaries,
                    'collect': collect
                }
                results = list(self._run(
                    dataset, fetches,
                    metadata=metadata, summary_steps=summary_steps,
                    feed_dict=train_feed_dict
                ))

                for i, name in enumerate(self._training_stats):
                    value = np.average([values[i] for values in results])
                    logger.info('{} (Training): {:.3e}'.format(name, value))

                global_step = self._tensors['global_step'].eval()
                self._saver.save(
                    self._session, model_file, global_step=global_step
                )

        return global_step

    def evaluate(
        self, data, tensors=None, combine_func=_concat, enable_summaries=False,
        batch_size=100, **feed_dict
    ):
        """Evaluate the model on the provided data.

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
        **feed_dict
            Additional keyword arguments to append to the feed
            dictionary used for evaluation.

        Yields
        ------
        results : list
            If `combine_func` is None, a list containing outputs from a
            single batch, otherwise yields the combined results from all
            batches at once.

        """
        if not isinstance(data, dataset.Dataset):
            data = dataset.Dataset(data, batch_size=ENQUEUE_BATCH_SIZE)
        if tensors is None:
            tensors = sorted(self._tensors)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with self._graph.as_default():
            with tf.Session(config=config) as self._session:
                self._init_tf(restore=True, enable_summaries=enable_summaries)

                fetches = {
                    'collect': [self._tensors[name] for name in tensors]
                }
                feed_dict = {'batch_size': batch_size, **feed_dict}
                generator = self._run(data, fetches, feed_dict=feed_dict)

                if combine_func is None:
                    for values in generator:
                        yield values
                else:
                    accumulator = [[] for i in range(len(tensors))]
                    for values in generator:
                        for i, value in enumerate(values):
                            accumulator[i].append(value)

                    for values in accumulator:
                        yield combine_func(values)

    def collect_stats(
        self, data, tensors, enable_summaries=False, **feed_dict
    ):
        """Collect statistics on the provided data

        Parameters
        ----------
        data
            Input data. Can be a NumPy array, a SciPy sparse matrix, or
            a :obj:`Dataset` object.
        tensors: list of str
            List containing names of statistics to collect.
        enable_summaries : bool, optional (default=False)
            Whether to write summaries to disk.
        **feed_dict
            Additional keyword arguments to append to the feed
            dictionary used for evaluation.

        Returns
        -------
        results : list
            The collected statistics.

        """
        logger = logging.getLogger(__name__)

        results = list(self.evaluate(
            data, tensors + ['global_step'],
            enable_summaries=enable_summaries, combine_func=np.average,
            **feed_dict
        ))
        global_step = results[-1]

        if enable_summaries:
            for (name, value) in zip(tensors, results[:-1]):
                logger.info('{} (Validation): {:.3e}'.format(name, value))
                summary = tf.Summary(value=[tf.Summary.Value(
                    tag='Validation/{}'.format(name), simple_value=value
                )])
                self._summary_writer.add_summary(
                    summary, global_step=global_step
                )

        return results[:-1]

    def fit(
        self, train_data, validation_data=None,
        epochs=1, shuffle=True, restore=False, summary_steps=None,
        init_feed_dict={}, validation_feed_dict={},
        batch_size=100,  **train_feed_dict
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
        **feed_dict
            Additional keyword arguments to append to the feed
            dictionary used for training

        Returns
        -------
        self

        """
        logger = logging.getLogger(__name__)
        model_file = os.path.join(self.model_dir, 'model.cpkt')

        if not isinstance(train_data, dataset.Dataset):
            train_data = dataset.Dataset(
                train_data, batch_size=ENQUEUE_BATCH_SIZE, shuffle=shuffle
            )
        if not isinstance(validation_data, dataset.Dataset):
            validation_data = dataset.Dataset(
                validation_data, batch_size=ENQUEUE_BATCH_SIZE
            )

        train_feed_dict = {'batch_size': batch_size, **train_feed_dict}
        init_feed_dict = {**train_feed_dict, **init_feed_dict}
        validation_feed_dict = {**train_feed_dict, **validation_feed_dict}
        for i in range(epochs):
            logger.info('Training epoch {}'.format(i + 1))
            self._train_model(
                train_data, restore=restore or i > 0,
                metadata=(i ==0), summary_steps=summary_steps,
                init_feed_dict=init_feed_dict, train_feed_dict=train_feed_dict
            )
            if i != epochs - 1:
                train_data.reset()

            if validation_data is not None:
                logger.info('Validating epoch {}'.format(i + 1))
                self.collect_stats(
                    validation_data, self._validation_stats,
                    enable_summaries=summary_steps is not None,
                    **validation_feed_dict
                )
                if i != epochs - 1:
                    validation_data.reset()

        return self
