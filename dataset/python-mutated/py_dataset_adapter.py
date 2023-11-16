import multiprocessing
import queue
import random
import threading
import time
import warnings
import weakref
from contextlib import closing
import numpy as np
import tree
from keras.api_export import keras_export
from keras.trainers.data_adapters import data_adapter_utils
from keras.trainers.data_adapters.data_adapter import DataAdapter

@keras_export(['keras.utils.PyDataset', 'keras.utils.Sequence'])
class PyDataset:
    """Base class for defining a parallel dataset using Python code.

    Every `PyDataset` must implement the `__getitem__()` and the `__len__()`
    methods. If you want to modify your dataset between epochs,
    you may additionally implement `on_epoch_end()`.
    The `__getitem__()` method should return a complete batch
    (not a single sample), and the `__len__` method should return
    the number of batches in the dataset (rather than the number of samples).

    Args:
        workers: Number of workers to use in multithreading or
            multiprocessing.
        use_multiprocessing: Whether to use Python multiprocessing for
            parallelism. Setting this to `True` means that your
            dataset will be replicated in multiple forked processes.
            This is necessary to gain compute-level (rather than I/O level)
            benefits from parallelism. However it can only be set to
            `True` if your dataset can be safely pickled.
        max_queue_size: Maximum number of batches to keep in the queue
            when iterating over the dataset in a multithreaded or
            multipricessed setting.
            Reduce this value to reduce the CPU memory consumption of
            your dataset. Defaults to 10.

    Notes:

    - `PyDataset` is a safer way to do multiprocessing.
        This structure guarantees that the model will only train
        once on each sample per epoch, which is not the case
        with Python generators.
    - The arguments `workers`, `use_multiprocessing`, and `max_queue_size`
        exist to configure how `fit()` uses parallelism to iterate
        over the dataset. They are not being used by the `PyDataset` class
        directly. When you are manually iterating over a `PyDataset`,
        no parallelism is applied.

    Example:

    ```python
    from skimage.io import imread
    from skimage.transform import resize
    import numpy as np
    import math

    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.

    class CIFAR10PyDataset(keras.utils.PyDataset):

        def __init__(self, x_set, y_set, batch_size, **kwargs):
            super().__init__(**kwargs)
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            # Return number of batches.
            return math.ceil(len(self.x) / self.batch_size)

        def __getitem__(self, idx):
            # Return x, y for batch idx.
            low = idx * self.batch_size
            # Cap upper bound at array length; the last batch may be smaller
            # if the total number of items is not a multiple of batch size.
            high = min(low + self.batch_size, len(self.x))
            batch_x = self.x[low:high]
            batch_y = self.y[low:high]

            return np.array([
                resize(imread(file_name), (200, 200))
                   for file_name in batch_x]), np.array(batch_y)
    ```
    """

    def __init__(self, workers=1, use_multiprocessing=False, max_queue_size=10):
        if False:
            while True:
                i = 10
        self._workers = workers
        self._use_multiprocessing = use_multiprocessing
        self._max_queue_size = max_queue_size

    def _warn_if_super_not_called(self):
        if False:
            for i in range(10):
                print('nop')
        warn = False
        if not hasattr(self, '_workers'):
            self._workers = 1
            warn = True
        if not hasattr(self, '_use_multiprocessing'):
            self._use_multiprocessing = False
            warn = True
        if not hasattr(self, '_max_queue_size'):
            self._max_queue_size = 10
            warn = True
        if warn:
            warnings.warn('Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.', stacklevel=2)

    @property
    def workers(self):
        if False:
            print('Hello World!')
        self._warn_if_super_not_called()
        return self._workers

    @workers.setter
    def workers(self, value):
        if False:
            return 10
        self._workers = value

    @property
    def use_multiprocessing(self):
        if False:
            while True:
                i = 10
        self._warn_if_super_not_called()
        return self._use_multiprocessing

    @use_multiprocessing.setter
    def use_multiprocessing(self, value):
        if False:
            return 10
        self._use_multiprocessing = value

    @property
    def max_queue_size(self):
        if False:
            return 10
        self._warn_if_super_not_called()
        return self._max_queue_size

    @max_queue_size.setter
    def max_queue_size(self, value):
        if False:
            while True:
                i = 10
        self._max_queue_size = value

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        'Gets batch at position `index`.\n\n        Args:\n            index: position of the batch in the PyDataset.\n\n        Returns:\n            A batch\n        '
        raise NotImplementedError

    def __len__(self):
        if False:
            while True:
                i = 10
        'Number of batch in the PyDataset.\n\n        Returns:\n            The number of batches in the PyDataset.\n        '
        raise NotImplementedError

    def on_epoch_end(self):
        if False:
            for i in range(10):
                print('nop')
        'Method called at the end of every epoch.'
        pass

    def __iter__(self):
        if False:
            return 10
        'Create a generator that iterate over the PyDataset.'
        for i in range(len(self)):
            yield self[i]

class PyDatasetAdapter(DataAdapter):
    """Adapter for `keras.utils.PyDataset` instances."""

    def __init__(self, x, class_weight=None, shuffle=False):
        if False:
            return 10
        self.py_dataset = x
        self.class_weight = class_weight
        self.enqueuer = None
        self.shuffle = shuffle
        self._output_signature = None

    def _set_tf_output_signature(self):
        if False:
            i = 10
            return i + 15
        from keras.utils.module_utils import tensorflow as tf

        def get_tensor_spec(x):
            if False:
                return 10
            shape = x.shape
            if len(shape) < 1:
                raise ValueError(f'The arrays returned by PyDataset.__getitem__() must be at least rank 1. Received: {x} of rank {len(x.shape)}')
            shape = list(shape)
            shape[0] = None
            return tf.TensorSpec(shape=shape, dtype=x.dtype.name)
        batch = self.py_dataset[0]
        batch = self._standardize_batch(batch)
        self._output_signature = tree.map_structure(get_tensor_spec, batch)

    def _standardize_batch(self, batch):
        if False:
            return 10
        if isinstance(batch, np.ndarray):
            batch = (batch,)
        if isinstance(batch, list):
            batch = tuple(batch)
        if not isinstance(batch, tuple) or len(batch) not in {1, 2, 3}:
            raise ValueError(f'PyDataset.__getitem__() must return a tuple, either (input,) or (inputs, targets) or (inputs, targets, sample_weights). Received: {str(batch)[:100]}... of type {type(batch)}')
        if self.class_weight is not None:
            if len(batch) == 3:
                raise ValueError('You cannot specify `class_weight` and `sample_weight` at the same time.')
            if len(batch) == 2:
                sw = data_adapter_utils.class_weight_to_sample_weights(batch[1], self.class_weight)
                batch = batch + (sw,)
        return batch

    def _make_multiprocessed_generator_fn(self):
        if False:
            i = 10
            return i + 15
        workers = self.py_dataset.workers
        use_multiprocessing = self.py_dataset.use_multiprocessing
        if workers > 1 or (workers > 0 and use_multiprocessing):

            def generator_fn():
                if False:
                    i = 10
                    return i + 15
                self.enqueuer = OrderedEnqueuer(self.py_dataset, use_multiprocessing=use_multiprocessing, shuffle=self.shuffle)
                self.enqueuer.start(workers=workers, max_queue_size=self.py_dataset.max_queue_size)
                return self.enqueuer.get()
        else:

            def generator_fn():
                if False:
                    print('Hello World!')
                order = range(len(self.py_dataset))
                if self.shuffle:
                    order = list(order)
                    random.shuffle(order)
                for i in order:
                    yield self.py_dataset[i]
        return generator_fn

    def get_numpy_iterator(self):
        if False:
            for i in range(10):
                print('nop')
        gen_fn = self._make_multiprocessed_generator_fn()
        for (i, batch) in enumerate(gen_fn()):
            batch = self._standardize_batch(batch)
            yield batch
            if i >= len(self.py_dataset) - 1 and self.enqueuer:
                self.enqueuer.stop()

    def get_tf_dataset(self):
        if False:
            while True:
                i = 10
        from keras.utils.module_utils import tensorflow as tf
        if self._output_signature is None:
            self._set_tf_output_signature()
        ds = tf.data.Dataset.from_generator(self.get_numpy_iterator, output_signature=self._output_signature)
        if self.shuffle:
            ds = ds.shuffle(8)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def on_epoch_end(self):
        if False:
            i = 10
            return i + 15
        if self.enqueuer:
            self.enqueuer.stop()
        self.py_dataset.on_epoch_end()

    @property
    def num_batches(self):
        if False:
            return 10
        return len(self.py_dataset)

    @property
    def batch_size(self):
        if False:
            i = 10
            return i + 15
        return None
_SHARED_SEQUENCES = {}
_SEQUENCE_COUNTER = None
_DATA_POOLS = weakref.WeakSet()
_WORKER_ID_QUEUE = None
_FORCE_THREADPOOL = False

def get_pool_class(use_multiprocessing):
    if False:
        while True:
            i = 10
    global _FORCE_THREADPOOL
    if not use_multiprocessing or _FORCE_THREADPOOL:
        return multiprocessing.dummy.Pool
    return multiprocessing.Pool

def get_worker_id_queue():
    if False:
        i = 10
        return i + 15
    'Lazily create the queue to track worker ids.'
    global _WORKER_ID_QUEUE
    if _WORKER_ID_QUEUE is None:
        _WORKER_ID_QUEUE = multiprocessing.Queue()
    return _WORKER_ID_QUEUE

def init_pool(seqs):
    if False:
        i = 10
        return i + 15
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = seqs

def get_index(uid, i):
    if False:
        while True:
            i = 10
    'Get the value from the PyDataset `uid` at index `i`.\n\n    To allow multiple PyDatasets to be used at the same time, we use `uid` to\n    get a specific one. A single PyDataset would cause the validation to\n    overwrite the training PyDataset.\n\n    Args:\n        uid: int, PyDataset identifier\n        i: index\n\n    Returns:\n        The value at index `i`.\n    '
    return _SHARED_SEQUENCES[uid][i]

class PyDatasetEnqueuer:
    """Base class to enqueue inputs.

    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.

    Example:

    ```python
        enqueuer = PyDatasetEnqueuer(...)
        enqueuer.start()
        datas = enqueuer.get()
        for data in datas:
            # Use the inputs; training, evaluating, predicting.
            # ... stop sometime.
        enqueuer.stop()
    ```

    The `enqueuer.get()` should be an infinite stream of data.
    """

    def __init__(self, py_dataset, use_multiprocessing=False):
        if False:
            return 10
        self.py_dataset = py_dataset
        self.use_multiprocessing = use_multiprocessing
        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = multiprocessing.Value('i', 0)
            except OSError:
                _SEQUENCE_COUNTER = 0
        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1
        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        if False:
            return 10
        return self.stop_signal is not None and (not self.stop_signal.is_set())

    def start(self, workers=1, max_queue_size=10):
        if False:
            print('Hello World!')
        "Starts the handler's workers.\n\n        Args:\n            workers: Number of workers.\n            max_queue_size: queue size\n                (when full, workers could block on `put()`)\n        "
        if self.use_multiprocessing:
            self.executor_fn = self._get_executor_init(workers)
        else:
            self.executor_fn = lambda _: get_pool_class(False)(workers)
        self.workers = workers
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _send_py_dataset(self):
        if False:
            return 10
        'Sends current Iterable to all workers.'
        _SHARED_SEQUENCES[self.uid] = self.py_dataset

    def stop(self, timeout=None):
        if False:
            print('Hello World!')
        'Stops running threads and wait for them to exit, if necessary.\n\n        Should be called by the same thread which called `start()`.\n\n        Args:\n            timeout: maximum time to wait on `thread.join()`\n        '
        if not self.is_running():
            return
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        _SHARED_SEQUENCES[self.uid] = None

    def __del__(self):
        if False:
            print('Hello World!')
        if self.is_running():
            self.stop()

    def _run(self):
        if False:
            print('Hello World!')
        'Submits request to the executor and queue the `Future` objects.'
        raise NotImplementedError

    def _get_executor_init(self, workers):
        if False:
            i = 10
            return i + 15
        'Gets the Pool initializer for multiprocessing.\n\n        Args:\n            workers: Number of workers.\n\n        Returns:\n            Function, a Function to initialize the pool\n        '
        raise NotImplementedError

    def get(self):
        if False:
            i = 10
            return i + 15
        'Creates a generator to extract data from the queue.\n\n        Skip the data if it is `None`.\n\n        Returns:\n            Generator yielding tuples `(inputs, targets)`\n                or `(inputs, targets, sample_weights)`.\n        '
        raise NotImplementedError

class OrderedEnqueuer(PyDatasetEnqueuer):
    """Builds a Enqueuer from a PyDataset.

    Args:
        py_dataset: A `keras.utils.PyDataset` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """

    def __init__(self, py_dataset, use_multiprocessing=False, shuffle=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(py_dataset, use_multiprocessing)
        self.shuffle = shuffle

    def _get_executor_init(self, workers):
        if False:
            while True:
                i = 10
        'Gets the Pool initializer for multiprocessing.\n\n        Args:\n            workers: Number of workers.\n\n        Returns:\n            Function, a Function to initialize the pool\n        '

        def pool_fn(seqs):
            if False:
                for i in range(10):
                    print('nop')
            pool = get_pool_class(True)(workers, initializer=init_pool_generator, initargs=(seqs, None, get_worker_id_queue()))
            _DATA_POOLS.add(pool)
            return pool
        return pool_fn

    def _wait_queue(self):
        if False:
            for i in range(10):
                print('nop')
        'Wait for the queue to be empty.'
        while True:
            time.sleep(0.1)
            if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
                return

    def _run(self):
        if False:
            return 10
        'Submits request to the executor and queue the `Future` objects.'
        indices = list(range(len(self.py_dataset)))
        if self.shuffle:
            random.shuffle(indices)
        self._send_py_dataset()
        while True:
            with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
                for i in indices:
                    if self.stop_signal.is_set():
                        return
                    self.queue.put(executor.apply_async(get_index, (self.uid, i)), block=True)
                self._wait_queue()
                if self.stop_signal.is_set():
                    return
            self.py_dataset.on_epoch_end()
            self._send_py_dataset()

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        'Creates a generator to extract data from the queue.\n\n        Skip the data if it is `None`.\n\n        Yields:\n            The next element in the queue, i.e. a tuple\n            `(inputs, targets)` or\n            `(inputs, targets, sample_weights)`.\n        '
        while self.is_running():
            try:
                inputs = self.queue.get(block=True, timeout=5).get()
                if self.is_running():
                    self.queue.task_done()
                if inputs is not None:
                    yield inputs
            except queue.Empty:
                pass
            except Exception as e:
                self.stop()
                raise e

def init_pool_generator(gens, random_seed=None, id_queue=None):
    if False:
        while True:
            i = 10
    'Initializer function for pool workers.\n\n    Args:\n        gens: State which should be made available to worker processes.\n        random_seed: An optional value with which to seed child processes.\n        id_queue: A multiprocessing Queue of worker ids.\n            This is used to indicate that a worker process\n            was created by Keras.\n    '
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = gens
    worker_proc = multiprocessing.current_process()
    worker_proc.name = f'Keras_worker_{worker_proc.name}'
    if random_seed is not None:
        np.random.seed(random_seed + worker_proc.ident)
    if id_queue is not None:
        id_queue.put(worker_proc.ident, block=True, timeout=0.1)