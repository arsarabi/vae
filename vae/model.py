import abc
import copy
import logging
import os
import threading
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from .globals import QUEUE_CAPACITY, TIMEOUT


class Model(metaclass=abc.ABCMeta):
    """Base abstract class for models

    Parameters
    ----------
    n_inputs : int
        Length of the input vector.
    n_labels : int or None, optional (default=None)
        Length of the label vector. If not provided, assumes data is
        un-labeled.
    optimizer : str, optional (default='Adam')
        Optimizer to use for training the model. See
        ``tf.contrib.layers.OPTIMIZER_CLS_NAMES`` for available options.
    learning_rate : float, optional (default=0.001)
        Learning rate for training the model.
    learning_rate_decay_fn : optional (default=None)
        Function for decaying the learning rate. Takes `learning_rate`
        and `global_step`, and returns the decayed learning rate.
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
        self, n_inputs, n_labels=None,
        optimizer='Adam', learning_rate=0.001, learning_rate_decay_fn=None,
        clip_gradients=None, model_dir=None, debug=False
    ):
        logger = logging.getLogger(__name__)
        if optimizer not in tf.contrib.layers.OPTIMIZER_CLS_NAMES:
            raise ValueError(
                'Invalid value for optimizer. Available options are {}'
                .format(sorted(tf.contrib.layers.OPTIMIZER_CLS_NAMES))
            )

        self._params.update({
            'n_inputs': n_inputs,
            'n_labels': n_labels,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'learning_rate_decay_fn': learning_rate_decay_fn,
            'clip_gradients': clip_gradients,
            'model_dir': model_dir or os.getcwd(),
            'debug': debug
        })
        self._build_graph()

    def __getattr__(self, name):
        if '_params' not in self.__dict__:
            self._params = {}

        if name in self._params:
            return self._params[name]
        else:
            return super().__getattribute__(name)

    def __getstate__(self):
        return self._params

    def __setstate__(self, state):
        logger = logging.getLogger(__name__)
        self._params = state
        self._build_graph()

    @property
    def model_dir(self):
        """str: Directory for saving and restoring the model"""
        return self._params['model_dir']

    @model_dir.setter
    def model_dir(self, value):
        if not os.path.isdir(value):
            raise ValueError("'{}' is not a valid directory".format(model_dir))

        self._params['model_dir'] = value

    def _create_placeholders(self):
        """Create the TensorFlow placeholders for the model"""
        input_feed = tf.placeholder(
            dtype=tf.float32, shape=(None, self.n_inputs), name='InputFeed'
        )
        if self.n_labels is None:
            label_feed = None
        else:
            label_feed = tf.placeholder(
                dtype=tf.float32, shape=(None, self.n_labels), name='LabelFeed'
            )

        self._placeholders.update({
            'input_feed': input_feed,
            'label_feed': label_feed,
            'batch_size': tf.placeholder(tf.int32, name='BatchSize')
        })

    def _create_input_pipeline(self):
        """Create the input pipeline

        Creates the FIFO queue, and enqueue and dequeue operations for
        feeding input into the model.

        """
        input_feed = self._placeholders['input_feed']
        label_feed = self._placeholders['label_feed']
        batch_size = self._placeholders['batch_size']
        with tf.device('/cpu:0'), tf.name_scope('InputPipeline'):
            dtypes = [input_feed.dtype]
            shapes = [[input_feed.get_shape().as_list()[-1]]]
            if label_feed is not None:
                dtypes.append(label_feed.dtype)
                shapes.append([label_feed.get_shape().as_list()[-1]])

            queue = tf.FIFOQueue(
                capacity=QUEUE_CAPACITY, dtypes=dtypes, shapes=shapes,
                name='Queue'
            )
            self._operations['queue_closer'] = queue.close()

            if label_feed is None:
                self._operations['enqueue'] = queue.enqueue_many(
                    input_feed, name='Enqueue'
                )
                self._tensors['inputs'] = queue.dequeue_up_to(
                    batch_size, name='Dequeue'
                )
            else:
                self._operations['enqueue'] = queue.enqueue_many(
                    [input_feed, label_feed], name='Enqueue'
                )
                self._tensors['inputs'], self._tensors['labels'] = (
                    queue.dequeue_up_to(batch_size, name='Dequeue')
                )

    def _create_train_op(self, cost):
        """Create the training operation

        Parameters
        ----------
        cost : :obj:`tf.Tensor`
            Cost to be minimized.

        Returns
        -------
        train_op : :obj:`tf.Operation`
            Operation for training the model.

        """
        self._operations['train'] = tf.contrib.layers.optimize_loss(
            cost, self._tensors['global_step'],
            self.learning_rate, self.optimizer,
            clip_gradients=self.clip_gradients,
            learning_rate_decay_fn=self.learning_rate_decay_fn,
            summaries=['learning_rate', 'gradient_norm']
        )

    @abc.abstractmethod
    def _build_model(self):
        """Build the model"""
        pass

    def _build_graph(self):
        """Build the computational graph for the model"""
        logger = logging.getLogger(__name__)
        logger.info('Building computational graph')

        self._graph = tf.Graph()
        with self._graph.as_default():
            global_step = tf.Variable(0, name='GlobalStep', trainable=False)
            self._tensors = {'global_step': global_step}
            self._placeholders = {}
            self._operations = {}

            self._create_placeholders()
            self._create_input_pipeline()
            self._build_model()

            self._operations['init'] = tf.global_variables_initializer()
            self._merged_summaries = tf.summary.merge_all()
            self._summary_writer = None

    def _init_tf(
        self, restore=False, enable_summaries=False,
        checkpoint_file=None, feed_dict={}
    ):
        """Initialize TensorFlow operations

        Parameters
        ----------
        restore : bool, optional (default=False)
            Whether to restore the model from a previous checkpoint.
        enable_summaries : bool, optional (default=False)
            Whether to initialize a :obj:`tf.summary.FileWriter`
            object for writing summaries to disk.
        checkpoint_file : str or None, optional (default=None)
            Checkpoint file to use. Defaults to the latest checkpoint.
        feed_dict : dict, optional (default={}):
            Feed dictionary to use for initialization.

        """
        if enable_summaries and self._summary_writer is None:
            self._summary_writer = tf.summary.FileWriter(
                self.model_dir, graph=self._graph
            )
        if not enable_summaries and self._summary_writer is not None:
            self._summary_writer.close()
            self._summary_writer = None

        if restore and checkpoint_file is None:
            checkpoint_file = tf.train.latest_checkpoint(self.model_dir)

        self._saver = tf.train.Saver()
        if restore and checkpoint_file is not None:
            checkpoint_file = os.path.join(
                self.model_dir, os.path.split(checkpoint_file)[1]
            )
            self._saver.restore(self._session, checkpoint_file)
        else:
            feed_dict = {
                self._placeholders[key]: value
                for key, value in feed_dict.items()
            }
            self._session.run(self._operations['init'], feed_dict=feed_dict)

    def _enqueue(self, dataset, coord):
        """Enqueue data

        Parameters
        ----------
        dataset : :obj:`Dataset`
            Data to enqueue.
        coord : :obj:`tf.train.Coordinator`

        """
        input_feed = self._placeholders['input_feed']
        label_feed = self._placeholders['label_feed']
        run_options = tf.RunOptions(timeout_in_ms=TIMEOUT)
        try:
            for batch in dataset:
                if self.n_labels is None:
                    feed_dict = {input_feed: batch}
                else:
                    feed_dict = {input_feed: batch[0], label_feed: batch[1]}

                while not coord.should_stop():
                    try:
                        self._session.run(
                            self._operations['enqueue'],
                            feed_dict=feed_dict, options=run_options
                        )
                    except tf.errors.DeadlineExceededError:
                        continue
                    else:
                        break

            self._session.run(self._operations['queue_closer'])
            while not coord.should_stop():
                pass
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()

    def _start_enqueue_threads(self, datasets, coord):
        """Start the threads for enqueuing data

        Parameters
        ----------
        datasets : :obj:`Dataset` or list of :obj:`Dataset`
            Data to feed to the model.
        coord : :obj:`tf.train.Coordinator`

        Returns
        -------
        threads : list of :obj:`threading.Thread`

        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        threads = []
        for dataset in datasets:
            thread = threading.Thread(
                target=self._enqueue, args=(dataset, coord)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)

        return threads

    def _run(
        self, datasets, fetches,
        metadata=False, summary_steps=None, feed_dict={}
    ):
        """Run a list of nodes over the provided data

        Parameters
        ----------
        datasets : :obj:`Dataset` or list of :obj:`Dataset`
            Data to feed to the model.
        fetches
            Operations to run. If a dictionary, every graph element
            under 'summary' is added to summary data, and outputs from
            graph elements under 'collect' are collected and returned.
        metadata : bool, optional (default=False)
            Whether to collect and write runtime metadata on the first
            training batch.
        summary_steps : int or None, optional (default=None)
            Number of steps between writing summaries, or None for
            disabling summaries. Ignored if fetches is not a dictionary,
            does not have the 'summary' field, or a summary writer has
            not been defined.
        feed_dict : dict, optional (default={})
            Feed dictionary to use.

        Yields
        ------
        results
            Outputs from a single mini-batch (if `fetches` is a
            dictionary and contains a 'collect' field).

        """
        logger = logging.getLogger(__name__)
        if summary_steps is not None:
            if not isinstance(summary_steps, int) or summary_steps <= 0:
                raise ValueError(
                    'summary_steps should be a positive integer, or None'
                )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self._session, coord=coord)
        enqueue_threads = self._start_enqueue_threads(datasets, coord)

        if self.debug:
            session = tf_debug.LocalCLIDebugWrapperSession(self._session)
            session.add_tensor_filter(
                'has_inf_or_nan', tf_debug.has_inf_or_nan
            )
            run_options_kwargs = {}
        else:
            session = self._session
            run_options_kwargs = {'timeout_in_ms': TIMEOUT}

        i = 0
        feed_dict = {
            self._placeholders[key]: value for key, value in feed_dict.items()
        }
        write_summary = (
            self._summary_writer is not None and summary_steps is not None
        )
        try:
            while True:
                kwargs = copy.copy(run_options_kwargs)
                if self._summary_writer is not None and metadata and i == 0:
                    kwargs['trace_level'] = tf.RunOptions.FULL_TRACE
                    run_metadata = tf.RunMetadata()
                else:
                    run_metadata = None

                run_options = tf.RunOptions(**kwargs)
                try:
                    results = session.run(
                        fetches, feed_dict=feed_dict,
                        options=run_options, run_metadata=run_metadata
                    )
                except tf.errors.DeadlineExceededError:
                    continue
                except tf.errors.OutOfRangeError:
                    break
                else:
                    i += 1
                    logger.debug('Processed batch {}'.format(i))
                    if run_metadata is not None:
                        self._summary_writer.add_run_metadata(
                            run_metadata, 'RunMetadata',
                            global_step=self._tensors['global_step'].eval()
                        )
                    if write_summary and i % summary_steps == 0:
                        if isinstance(results, dict) and 'summary' in results:
                            self._summary_writer.add_summary(
                                results['summary'],
                                global_step=self._tensors['global_step'].eval()
                            )
                    if isinstance(results, dict) and 'collect' in results:
                        yield results['collect']
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
            for thread in enqueue_threads:
                thread.join()
