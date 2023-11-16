"""Adapter module that convert different input data objects into tf.dataset."""
import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest

class DataAdapter(object, metaclass=abc.ABCMeta):
    """Base class for input data adapter.

  In TF 2.0, tf.data is the preferred API for user to feed in data. In order
  to simplify the training code path, all the input data object will be
  converted to `tf.data.Dataset` if possible.

  Note that since this class is mainly targeted for TF 2.0, it might have a lot
  of assumptions under the hood, eg eager context by default, distribution
  strategy, etc. In the meantime, some legacy feature support might be dropped,
  eg, Iterator from dataset API in v1, etc.

  The sample usage of this class is like:

  ```
  x = tf.data.Dataset.range(100)
  adapter_cls = [NumpyArrayDataAdapter, ..., DatasetAdapter]
  applicable_adapters = [cls for cls in adapter_cls if cls.can_handle(x)]
  if len(applicable_adapters) != 1:
    raise ValueError("Expect only one adapter class to handle the input")

  dataset = applicable_adapters[0](x).get_dataset()
  for data in dataset:
    # training
  ```
  """

    @staticmethod
    def can_handle(x, y=None):
        if False:
            for i in range(10):
                print('nop')
        'Whether the current DataAdapter could handle the input x and y.\n\n    Structure wise, x and y can be single object, or list of objects if there\n    multiple input/output, or dictionary of objects when the intput/output are\n    named.\n\n    Args:\n      x: input features.\n      y: target labels. Note that y could be None in the case of prediction.\n\n    Returns:\n      boolean\n    '
        raise NotImplementedError

    @abc.abstractmethod
    def __init__(self, x, y=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Create a DataAdapter based on data inputs.\n\n    The caller must make sure to call `can_handle()` first before invoking this\n    method. Provide unsupported data type will result into unexpected behavior.\n\n    Args:\n      x: input features.\n      y: target labels. Note that y could be None in the case of prediction.\n      **kwargs: Other keyword arguments for DataAdapter during the construction\n        of the tf.dataset.Dataset. For example:\n        - Numpy data might have `sample_weights` which will be used for\n          weighting the loss function during training.\n        - Numpy data might need to have `batch_size` parameter when constructing\n          the dataset and iterator.\n        - Certain input might need to be distribution strategy aware. When\n          `distribution_strategy` is passed, the created dataset need to respect\n          the strategy.\n        DataAdapter might choose to ignore any keyword argument if it doesn't\n        use it, or raise exception if any required argument is not provide.\n    "
        if not self.can_handle(x, y):
            raise ValueError('{} Cannot handle input {}, {}'.format(self.__class__, x, y))

    @abc.abstractmethod
    def get_dataset(self):
        if False:
            print('Hello World!')
        'Get a dataset instance for the current DataAdapter.\n\n    Note that the dataset returned does not repeat for epoch, so caller might\n    need to create new iterator for the same dataset at the beginning of the\n    epoch. This behavior might change in future.\n\n    Returns:\n      An tf.dataset.Dataset. Caller might use the dataset in different\n      context, eg iter(dataset) in eager to get the value directly, or in graph\n      mode, provide the iterator tensor to Keras model function.\n    '
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the size (number of batches) for the dataset created.\n\n    For certain type of the data input, the number of batches is known, eg for\n    Numpy data, the size is same as (number_of_element / batch_size). Whereas\n    for dataset or python generator, the size is unknown since it may or may not\n    have a end state.\n\n    Returns:\n      int, the number of batches for the dataset, or None if it is unknown. The\n      caller could use this to control the loop of training, show progress bar,\n      or handle unexpected StopIteration error.\n    '
        raise NotImplementedError

    @abc.abstractmethod
    def batch_size(self):
        if False:
            print('Hello World!')
        'Return the batch size of the dataset created.\n\n    For certain type of the data input, the batch size is known, and even\n    required, like numpy array. Where as for dataset, the batch is unknown\n    unless we take a peek.\n\n    Returns:\n      int, the batch size of the dataset, or None if it is unknown.\n    '
        raise NotImplementedError

    def representative_batch_size(self):
        if False:
            print('Hello World!')
        'Return a representative size for batches in the dataset.\n\n    This is not guaranteed to be the batch size for all batches in the\n    dataset. It just needs to be a rough approximation for batch sizes in\n    the dataset.\n\n    Returns:\n      int, a representative size for batches found in the dataset,\n      or None if it is unknown.\n    '
        return self.batch_size()

    @abc.abstractmethod
    def has_partial_batch(self):
        if False:
            for i in range(10):
                print('nop')
        'Whether the dataset has partial batch at the end.'
        raise NotImplementedError

    @abc.abstractmethod
    def partial_batch_size(self):
        if False:
            return 10
        'The size of the final partial batch for dataset.\n\n    Will return None if has_partial_batch is False or batch_size is None.\n    '
        raise NotImplementedError

    @abc.abstractmethod
    def should_recreate_iterator(self):
        if False:
            return 10
        'Returns whether a new iterator should be created every epoch.'
        raise NotImplementedError

    def get_samples(self):
        if False:
            print('Hello World!')
        'Returns number of samples in the data, or `None`.'
        if not self.get_size() or not self.batch_size():
            return None
        total_sample = self.get_size() * self.batch_size()
        if self.has_partial_batch():
            total_sample -= self.batch_size() - self.partial_batch_size()
        return total_sample

    def on_epoch_end(self):
        if False:
            i = 10
            return i + 15
        'A hook called after each epoch.'
        pass

class TensorLikeDataAdapter(DataAdapter):
    """Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy."""

    @staticmethod
    def can_handle(x, y=None):
        if False:
            for i in range(10):
                print('nop')
        flat_inputs = nest.flatten(x)
        if y is not None:
            flat_inputs += nest.flatten(y)
        tensor_types = _get_tensor_types()

        def _is_tensor(v):
            if False:
                return 10
            if isinstance(v, tensor_types):
                return True
            return False
        return all((_is_tensor(v) for v in flat_inputs))

    def __init__(self, x, y=None, sample_weights=None, sample_weight_modes=None, batch_size=None, epochs=1, steps=None, shuffle=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super(TensorLikeDataAdapter, self).__init__(x, y, **kwargs)
        (x, y, sample_weights) = _process_tensorlike((x, y, sample_weights))
        sample_weight_modes = broadcast_sample_weight_modes(sample_weights, sample_weight_modes)
        (sample_weights, _, _) = training_utils.handle_partial_sample_weights(y, sample_weights, sample_weight_modes, check_all_flat=True)
        inputs = pack_x_y_sample_weight(x, y, sample_weights)
        num_samples = set((int(i.shape[0]) for i in nest.flatten(inputs))).pop()
        _check_data_cardinality(inputs)
        if not batch_size:
            batch_size = int(math.ceil(num_samples / steps)) if steps else 32
        self._size = int(math.ceil(num_samples / batch_size))
        self._batch_size = batch_size
        num_full_batches = int(num_samples // batch_size)
        self._partial_batch_size = num_samples % batch_size
        if isinstance(shuffle, str):
            shuffle = shuffle.lower()
        self._shuffle = shuffle
        indices_dataset = dataset_ops.DatasetV2.range(1)
        if shuffle != 'batch':
            indices_dataset = indices_dataset.repeat(epochs)

        def permutation(_):
            if False:
                while True:
                    i = 10
            indices = math_ops.range(num_samples, dtype=dtypes.int64)
            if shuffle and shuffle != 'batch':
                indices = random_ops.random_shuffle(indices)
            return indices
        indices_dataset = indices_dataset.map(permutation).prefetch(1)

        def slice_batch_indices(indices):
            if False:
                i = 10
                return i + 15
            'Convert a Tensor of indices into a dataset of batched indices.\n\n      This step can be accomplished in several ways. The most natural is to\n      slice the Tensor in a Dataset map. (With a condition on the upper index to\n      handle the partial batch.) However it turns out that coercing the Tensor\n      into a shape which is divisible by the batch size (and handling the last\n      partial batch separately) allows for a much more favorable memory access\n      pattern and improved performance.\n\n      Args:\n        indices: Tensor which determines the data order for an entire epoch.\n\n      Returns:\n        A Dataset of batched indices.\n      '
            num_in_full_batch = num_full_batches * batch_size
            first_k_indices = array_ops.slice(indices, [0], [num_in_full_batch])
            first_k_indices = array_ops.reshape(first_k_indices, [num_full_batches, batch_size])
            flat_dataset = dataset_ops.DatasetV2.from_tensor_slices(first_k_indices)
            if self._partial_batch_size:
                index_remainder = dataset_ops.DatasetV2.from_tensors(array_ops.slice(indices, [num_in_full_batch], [self._partial_batch_size]))
                flat_dataset = flat_dataset.concatenate(index_remainder)
            if shuffle == 'batch':
                flat_dataset = flat_dataset.shuffle(1024).repeat(epochs)
            return flat_dataset
        indices_dataset = indices_dataset.flat_map(slice_batch_indices)
        dataset = self.slice_inputs(indices_dataset, inputs)
        if shuffle == 'batch':

            def shuffle_batch(*batch):
                if False:
                    print('Hello World!')
                return nest.map_structure(random_ops.random_shuffle, batch)
            dataset = dataset.map(shuffle_batch)
        self._dataset = dataset

    def slice_inputs(self, indices_dataset, inputs):
        if False:
            for i in range(10):
                print('nop')
        'Slice inputs into a Dataset of batches.\n\n    Given a Dataset of batch indices and the unsliced inputs,\n    this step slices the inputs in a parallelized fashion\n    and produces a dataset of input batches.\n\n    Args:\n      indices_dataset: A Dataset of batched indices\n      inputs: A python data structure that contains the inputs, targets,\n        and possibly sample weights.\n\n    Returns:\n      A Dataset of input batches matching the batch indices.\n    '
        dataset = dataset_ops.DatasetV2.zip((indices_dataset, dataset_ops.DatasetV2.from_tensors(inputs).repeat()))

        def grab_batch(i, data):
            if False:
                while True:
                    i = 10
            return nest.map_structure(lambda d: array_ops.gather(d, i, axis=0), data)
        dataset = dataset.map(grab_batch, num_parallel_calls=dataset_ops.AUTOTUNE)
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        if self._shuffle:
            options.experimental_external_state_policy = options_lib.ExternalStatePolicy.IGNORE
        dataset = dataset.with_options(options)
        return dataset

    def get_dataset(self):
        if False:
            while True:
                i = 10
        return self._dataset

    def get_size(self):
        if False:
            print('Hello World!')
        return self._size

    def batch_size(self):
        if False:
            while True:
                i = 10
        return self._batch_size

    def has_partial_batch(self):
        if False:
            return 10
        return self._partial_batch_size > 0

    def partial_batch_size(self):
        if False:
            return 10
        return self._partial_batch_size or None

    def should_recreate_iterator(self):
        if False:
            return 10
        return False

class GenericArrayLikeDataAdapter(TensorLikeDataAdapter):
    """Adapter that handles array-like data without forcing it into memory.

  This adapter handles array-like datasets that may be too big to fully
  fit into memory.

  Specifically, this adapter handles any Python class which implements:
  `__get_item__`, `__len__`, `shape`, and `dtype` with the same meanings
  as Numpy, but it ignores any case where all the inputs are Tensors or Numpy
  arrays (because that case is handled by the base TensorLikeDataAdapter).

  It ignores scipy sparse matrices and Composite Tensors because those are
  handled by the CompositeTensorDataAdapter.

  It also does not handle lists/tuples of scalars, because those are handled
  by the ListsOfScalarsDataAdapter.
  """

    @staticmethod
    def can_handle(x, y=None):
        if False:
            return 10
        flat_inputs = nest.flatten(x)
        if y is not None:
            flat_inputs += nest.flatten(y)

        def _is_array_like(v):
            if False:
                while True:
                    i = 10
            'Return True if v is a Tensor, array, or is array-like.'
            return hasattr(v, '__getitem__') and hasattr(v, 'shape') and hasattr(v, 'dtype') and hasattr(v, '__len__')
        if not TensorLikeDataAdapter.can_handle(x, y) and (not CompositeTensorDataAdapter.can_handle(x, y)):
            return all((_is_array_like(v) for v in flat_inputs))
        else:
            return False

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        logging.warning('Keras is training/fitting/evaluating on array-like data. Keras may not be optimized for this format, so if your input data format is supported by TensorFlow I/O (https://github.com/tensorflow/io) we recommend using that to load a Dataset instead.')
        super(GenericArrayLikeDataAdapter, self).__init__(*args, **kwargs)

    def slice_inputs(self, indices_dataset, inputs):
        if False:
            i = 10
            return i + 15
        'Slice inputs into a Dataset of batches.\n\n    Given a Dataset of batch indices and the unsliced inputs,\n    this step slices the inputs in a parallelized fashion\n    and produces a dataset of input batches.\n\n    Args:\n      indices_dataset: A Dataset of batched indices\n      inputs: A python data structure that contains the inputs, targets,\n        and possibly sample weights.\n\n    Returns:\n      A Dataset of input batches matching the batch indices.\n    '
        flat_inputs = nest.flatten(inputs)

        def dynamic_shape_like(t):
            if False:
                for i in range(10):
                    print('nop')
            shape = list(t.shape)
            shape[0] = None
            return tuple(shape)
        flat_dtypes = [inp.dtype for inp in flat_inputs]
        contiguous = True
        if self._shuffle and self._shuffle != 'batch':
            contiguous = False

        def grab_batch(indices):
            if False:
                while True:
                    i = 10
            'Grab a batch of data from the inputs.'

            def py_method(ind):
                if False:
                    print('Hello World!')

                def slice_array(data):
                    if False:
                        print('Hello World!')
                    return training_utils.slice_arrays(data, ind.numpy(), contiguous=contiguous)
                return [slice_array(inp) for inp in flat_inputs]
            flat_out = script_ops.eager_py_func(py_method, [indices], flat_dtypes)
            for (v, original_inp) in zip(flat_out, flat_inputs):
                v.set_shape(dynamic_shape_like(original_inp))
            return nest.pack_sequence_as(inputs, flat_out)
        dataset = indices_dataset.map(grab_batch, num_parallel_calls=dataset_ops.AUTOTUNE)
        return dataset

class DatasetCreatorAdapter(DataAdapter):
    """Adapter that handles dataset functions."""

    def __init__(self, x, y, steps=None, distribution_strategy=None, **kwargs):
        if False:
            while True:
                i = 10
        super(DatasetCreatorAdapter, self).__init__(x, **kwargs)
        if not isinstance(x, dataset_creator.DatasetCreator):
            raise TypeError('The input of a `DatasetCreatorAdapter` should be a `DatasetCreator` but it received type {}.'.format(type(x)))
        if steps is None:
            raise ValueError('When using a `tf.keras.utils.experimental.DatasetCreator`, `steps_per_epoch`, `validation_steps` or `steps` argument must be provided in `Model.fit`, `Model.evaluate`, or `Model.predict`.')
        self.dataset_creator = x
        self.steps = steps
        self.strategy = distribution_strategy

    @staticmethod
    def can_handle(x, y=None):
        if False:
            i = 10
            return i + 15
        if isinstance(x, dataset_creator.DatasetCreator):
            assert y is None
            return True

    def should_recreate_iterator(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def get_size(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def get_dataset(self):
        if False:
            return 10
        return self.strategy.distribute_datasets_from_function(self.dataset_creator, options=self.dataset_creator.input_options)

    def batch_size(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def has_partial_batch(self):
        if False:
            return 10
        raise NotImplementedError()

    def partial_batch_size(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class CompositeTensorDataAdapter(DataAdapter):
    """Adapter that handles composite tensor."""

    @staticmethod
    def can_handle(x, y=None):
        if False:
            return 10
        flat_inputs = nest.flatten(x)
        if y is not None:
            flat_inputs += nest.flatten(y)

        def _is_composite(v):
            if False:
                print('Hello World!')
            if tf_utils.is_extension_type(v) and (not isinstance(v, (dataset_ops.DatasetV2, iterator_ops.IteratorBase))) and (not _is_distributed_dataset(v)):
                return True
            return _is_scipy_sparse(v)

        def _is_tensor_or_composite(v):
            if False:
                while True:
                    i = 10
            if isinstance(v, (tensor.Tensor, np.ndarray)):
                return True
            return _is_composite(v)
        return any((_is_composite(v) for v in flat_inputs)) and all((_is_tensor_or_composite(v) for v in flat_inputs))

    def __init__(self, x, y=None, sample_weights=None, sample_weight_modes=None, batch_size=None, steps=None, shuffle=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super(CompositeTensorDataAdapter, self).__init__(x, y, **kwargs)
        (x, y, sample_weights) = _process_tensorlike((x, y, sample_weights))
        sample_weight_modes = broadcast_sample_weight_modes(sample_weights, sample_weight_modes)
        (sample_weights, _, _) = training_utils.handle_partial_sample_weights(y, sample_weights, sample_weight_modes, check_all_flat=True)
        inputs = pack_x_y_sample_weight(x, y, sample_weights)
        dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)
        num_samples = int(nest.flatten(x)[0].shape[0])
        if shuffle:
            dataset = dataset.shuffle(num_samples)
        if not batch_size:
            batch_size = int(math.ceil(num_samples / steps)) if steps else 32
        dataset = dataset.batch(batch_size)
        self._size = int(math.ceil(num_samples / batch_size))
        self._batch_size = batch_size
        self._has_partial_batch = self._size != num_samples // batch_size
        self._partial_batch_size = None
        if self._has_partial_batch:
            self._partial_batch_size = num_samples - (self._size - 1) * self._batch_size
        self._dataset = dataset

    def get_dataset(self):
        if False:
            while True:
                i = 10
        return self._dataset

    def get_size(self):
        if False:
            i = 10
            return i + 15
        return self._size

    def batch_size(self):
        if False:
            return 10
        return self._batch_size

    def has_partial_batch(self):
        if False:
            for i in range(10):
                print('nop')
        return self._has_partial_batch

    def partial_batch_size(self):
        if False:
            while True:
                i = 10
        return self._partial_batch_size

    def should_recreate_iterator(self):
        if False:
            i = 10
            return i + 15
        return True

class ListsOfScalarsDataAdapter(DataAdapter):
    """Adapter that handles lists of scalars and lists of lists of scalars."""

    @staticmethod
    def can_handle(x, y=None):
        if False:
            i = 10
            return i + 15
        handles_x = ListsOfScalarsDataAdapter._is_list_of_scalars(x)
        handles_y = True
        if y is not None:
            handles_y = ListsOfScalarsDataAdapter._is_list_of_scalars(y)
        return handles_x and handles_y

    @staticmethod
    def _is_list_of_scalars(inp):
        if False:
            return 10
        if isinstance(inp, (float, int, str, bytes, bytearray)):
            return True
        if isinstance(inp, (list, tuple)) and inp:
            return ListsOfScalarsDataAdapter._is_list_of_scalars(inp[0])
        return False

    def __init__(self, x, y=None, sample_weights=None, sample_weight_modes=None, batch_size=None, shuffle=False, **kwargs):
        if False:
            while True:
                i = 10
        super(ListsOfScalarsDataAdapter, self).__init__(x, y, **kwargs)
        x = np.asarray(x)
        if y is not None:
            y = np.asarray(y)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights)
        sample_weight_modes = broadcast_sample_weight_modes(sample_weights, sample_weight_modes)
        self._internal_adapter = TensorLikeDataAdapter(x, y=y, sample_weights=sample_weights, sample_weight_modes=sample_weight_modes, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def get_dataset(self):
        if False:
            print('Hello World!')
        return self._internal_adapter.get_dataset()

    def get_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._internal_adapter.get_size()

    def batch_size(self):
        if False:
            return 10
        return self._internal_adapter.batch_size()

    def has_partial_batch(self):
        if False:
            i = 10
            return i + 15
        return self._internal_adapter.has_partial_batch()

    def partial_batch_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._internal_adapter.partial_batch_size()

    def should_recreate_iterator(self):
        if False:
            i = 10
            return i + 15
        return True

class DatasetAdapter(DataAdapter):
    """Adapter that handles `tf.data.Dataset`."""

    @staticmethod
    def can_handle(x, y=None):
        if False:
            print('Hello World!')
        return isinstance(x, (data_types.DatasetV1, data_types.DatasetV2)) or _is_distributed_dataset(x)

    def __init__(self, x, y=None, sample_weights=None, steps=None, **kwargs):
        if False:
            print('Hello World!')
        super(DatasetAdapter, self).__init__(x, y, **kwargs)
        self._dataset = x
        self._user_steps = steps
        self._validate_args(y, sample_weights, steps)

    def get_dataset(self):
        if False:
            while True:
                i = 10
        return self._dataset

    def get_size(self):
        if False:
            i = 10
            return i + 15
        return

    def batch_size(self):
        if False:
            i = 10
            return i + 15
        return None

    def has_partial_batch(self):
        if False:
            print('Hello World!')
        return False

    def partial_batch_size(self):
        if False:
            return 10
        return None

    def should_recreate_iterator(self):
        if False:
            i = 10
            return i + 15
        if _is_distributed_dataset(self._dataset):
            return False
        return self._user_steps is None or cardinality.cardinality(self._dataset).numpy() == self._user_steps

    def _validate_args(self, y, sample_weights, steps):
        if False:
            while True:
                i = 10
        'Validates `__init__` arguments.'
        if not is_none_or_empty(y):
            raise ValueError('`y` argument is not supported when using dataset as input.')
        if not is_none_or_empty(sample_weights):
            raise ValueError('`sample_weight` argument is not supported when using dataset as input.')
        if steps is None:
            if _is_distributed_dataset(self._dataset):
                raise ValueError('When providing a distributed dataset, you must specify the number of steps to run.')
            size = cardinality.cardinality(self._dataset).numpy()
            if size == cardinality.INFINITE and steps is None:
                raise ValueError('When providing an infinite dataset, you must specify the number of steps to run (if you did not intend to create an infinite dataset, make sure to not call `repeat()` on the dataset).')

class GeneratorDataAdapter(DataAdapter):
    """Adapter that handles python generators and iterators."""

    @staticmethod
    def can_handle(x, y=None):
        if False:
            print('Hello World!')
        return (hasattr(x, '__next__') or hasattr(x, 'next')) and hasattr(x, '__iter__') and (not isinstance(x, data_utils.Sequence))

    def __init__(self, x, y=None, sample_weights=None, workers=1, use_multiprocessing=False, max_queue_size=10, model=None, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.pop('shuffle', None)
        if not is_none_or_empty(y):
            raise ValueError('`y` argument is not supported when using python generator as input.')
        if not is_none_or_empty(sample_weights):
            raise ValueError('`sample_weight` argument is not supported when using python generator as input.')
        super(GeneratorDataAdapter, self).__init__(x, y, **kwargs)
        (peek, x) = self._peek_and_restore(x)
        peek = self._standardize_batch(peek)
        peek = _process_tensorlike(peek)
        if model is not None and (not model.built):
            (concrete_x, _, _) = unpack_x_y_sample_weight(peek)
            model.distribute_strategy.run(lambda x: model(x, training=False), args=(concrete_x,))
        self._first_batch_size = int(nest.flatten(peek)[0].shape[0])

        def _get_dynamic_shape(t):
            if False:
                print('Hello World!')
            shape = t.shape
            if shape.rank is None:
                return shape
            return tensor_shape.TensorShape([None for _ in shape.as_list()])
        output_shapes = nest.map_structure(_get_dynamic_shape, peek)
        output_types = nest.map_structure(lambda t: t.dtype, peek)
        generator_fn = self._handle_multiprocessing(x, workers, use_multiprocessing, max_queue_size)

        def wrapped_generator():
            if False:
                return 10
            for data in generator_fn():
                yield self._standardize_batch(data)
        dataset = dataset_ops.DatasetV2.from_generator(wrapped_generator, output_types, output_shapes=output_shapes)
        if workers == 1 and (not use_multiprocessing):
            dataset = dataset.prefetch(1)
        self._dataset = dataset

    def _standardize_batch(self, data):
        if False:
            while True:
                i = 10
        'Standardizes a batch output by a generator.'
        (x, y, sample_weight) = unpack_x_y_sample_weight(data)
        data = pack_x_y_sample_weight(x, y, sample_weight)
        data = nest.list_to_tuple(data)

        def _convert_dtype(t):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(t, np.ndarray) and issubclass(t.dtype.type, np.floating):
                return np.array(t, dtype=backend.floatx())
            return t
        data = nest.map_structure(_convert_dtype, data)
        return data

    @staticmethod
    def _peek_and_restore(x):
        if False:
            while True:
                i = 10
        peek = next(x)
        return (peek, itertools.chain([peek], x))

    def _handle_multiprocessing(self, x, workers, use_multiprocessing, max_queue_size):
        if False:
            i = 10
            return i + 15
        'Create a callable, possibly including an Enqueuer.'
        if workers > 1 or (workers > 0 and use_multiprocessing):

            def generator_fn():
                if False:
                    print('Hello World!')
                enqueuer = data_utils.GeneratorEnqueuer(x, use_multiprocessing=use_multiprocessing)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                return enqueuer.get()
        else:
            generator_fn = lambda : x
        return generator_fn

    def get_dataset(self):
        if False:
            return 10
        return self._dataset

    def get_size(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def batch_size(self):
        if False:
            print('Hello World!')
        return None

    def representative_batch_size(self):
        if False:
            print('Hello World!')
        return self._first_batch_size

    def has_partial_batch(self):
        if False:
            i = 10
            return i + 15
        return False

    def partial_batch_size(self):
        if False:
            return 10
        return

    def should_recreate_iterator(self):
        if False:
            return 10
        return False

class KerasSequenceAdapter(GeneratorDataAdapter):
    """Adapter that handles `keras.utils.Sequence`."""

    @staticmethod
    def can_handle(x, y=None):
        if False:
            print('Hello World!')
        return isinstance(x, data_utils.Sequence)

    def __init__(self, x, y=None, sample_weights=None, shuffle=False, workers=1, use_multiprocessing=False, max_queue_size=10, model=None, **kwargs):
        if False:
            print('Hello World!')
        if not is_none_or_empty(y):
            raise ValueError('`y` argument is not supported when using `keras.utils.Sequence` as input.')
        if not is_none_or_empty(sample_weights):
            raise ValueError('`sample_weight` argument is not supported when using `keras.utils.Sequence` as input.')
        self._size = len(x)
        self._shuffle_sequence = shuffle
        self._keras_sequence = x
        self._enqueuer = None
        super(KerasSequenceAdapter, self).__init__(x, shuffle=False, workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size, model=model, **kwargs)

    @staticmethod
    def _peek_and_restore(x):
        if False:
            i = 10
            return i + 15
        return (x[0], x)

    def _handle_multiprocessing(self, x, workers, use_multiprocessing, max_queue_size):
        if False:
            while True:
                i = 10
        if workers > 1 or (workers > 0 and use_multiprocessing):

            def generator_fn():
                if False:
                    return 10
                self._enqueuer = data_utils.OrderedEnqueuer(x, use_multiprocessing=use_multiprocessing, shuffle=self._shuffle_sequence)
                self._enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                return self._enqueuer.get()
        else:

            def generator_fn():
                if False:
                    print('Hello World!')
                order = range(len(x))
                if self._shuffle_sequence:
                    order = list(order)
                    random.shuffle(order)
                for i in order:
                    yield x[i]
        return generator_fn

    def get_size(self):
        if False:
            return 10
        return self._size

    def should_recreate_iterator(self):
        if False:
            i = 10
            return i + 15
        return True

    def on_epoch_end(self):
        if False:
            for i in range(10):
                print('nop')
        if self._enqueuer:
            self._enqueuer.stop()
        self._keras_sequence.on_epoch_end()
ALL_ADAPTER_CLS = [ListsOfScalarsDataAdapter, TensorLikeDataAdapter, GenericArrayLikeDataAdapter, DatasetAdapter, GeneratorDataAdapter, KerasSequenceAdapter, CompositeTensorDataAdapter, DatasetCreatorAdapter]

def select_data_adapter(x, y):
    if False:
        print('Hello World!')
    'Selects a data adapter than can handle a given x and y.'
    adapter_cls = [cls for cls in ALL_ADAPTER_CLS if cls.can_handle(x, y)]
    if not adapter_cls:
        raise ValueError('Failed to find data adapter that can handle input: {}, {}'.format(_type_name(x), _type_name(y)))
    elif len(adapter_cls) > 1:
        raise RuntimeError('Data adapters should be mutually exclusive for handling inputs. Found multiple adapters {} to handle input: {}, {}'.format(adapter_cls, _type_name(x), _type_name(y)))
    return adapter_cls[0]

def _type_name(x):
    if False:
        return 10
    'Generates a description of the type of an object.'
    if isinstance(x, dict):
        key_types = set((_type_name(key) for key in x.keys()))
        val_types = set((_type_name(key) for key in x.values()))
        return '({} containing {} keys and {} values)'.format(type(x), key_types, val_types)
    if isinstance(x, (list, tuple)):
        types = set((_type_name(val) for val in x))
        return '({} containing values of types {})'.format(type(x), types)
    return str(type(x))

def _process_tensorlike(inputs):
    if False:
        while True:
            i = 10
    'Process tensor-like inputs.\n\n  This function:\n\n  (1) Converts `Numpy` arrays to `Tensor`s.\n  (2) Converts `Scipy` sparse matrices to `SparseTensor`s.\n  (2) Converts `list`s to `tuple`s (for `tf.data` support).\n\n  Args:\n    inputs: Structure of `Tensor`s, `NumPy` arrays, or tensor-like.\n\n  Returns:\n    Structure of `Tensor`s or tensor-like.\n  '

    def _convert_numpy_and_scipy(x):
        if False:
            return 10
        if isinstance(x, np.ndarray):
            dtype = None
            if issubclass(x.dtype.type, np.floating):
                dtype = backend.floatx()
            return tensor_conversion.convert_to_tensor_v2_with_dispatch(x, dtype=dtype)
        elif _is_scipy_sparse(x):
            return _scipy_sparse_to_sparse_tensor(x)
        return x
    inputs = nest.map_structure(_convert_numpy_and_scipy, inputs)
    return nest.list_to_tuple(inputs)

def is_none_or_empty(inputs):
    if False:
        return 10
    return inputs is None or not nest.flatten(inputs)

def broadcast_sample_weight_modes(target_structure, sample_weight_modes):
    if False:
        while True:
            i = 10
    'Match sample_weight_modes structure with output structure.'
    if target_structure is None or not nest.flatten(target_structure):
        return sample_weight_modes
    if isinstance(sample_weight_modes, str):
        if isinstance(target_structure, dict):
            return {key: sample_weight_modes for key in target_structure.keys()}
        return [sample_weight_modes for _ in target_structure]
    if sample_weight_modes:
        try:
            nest.assert_same_structure(training_utils.list_to_tuple(target_structure), training_utils.list_to_tuple(sample_weight_modes))
        except (ValueError, TypeError):
            target_str = str(nest.map_structure(lambda _: '...', target_structure))
            mode_str = str(nest.map_structure(lambda _: '...', sample_weight_modes))
            try:
                sample_weight_modes = nest.pack_sequence_as(target_structure, nest.flatten(sample_weight_modes))
                logging.warning('sample_weight modes were coerced from\n  {}\n    to  \n  {}'.format(target_str, mode_str))
            except (ValueError, TypeError):
                raise ValueError('Unable to match target structure and sample_weight_modes structure:\n  {}\n    to  \n  {}'.format(target_str, mode_str))
    return sample_weight_modes

class DataHandler(object):
    """Handles iterating over epoch-level `tf.data.Iterator` objects."""

    def __init__(self, x, y=None, sample_weight=None, batch_size=None, steps_per_epoch=None, initial_epoch=0, epochs=1, shuffle=False, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, model=None, steps_per_execution=None, distribute=True):
        if False:
            return 10
        'Initializes a `DataHandler`.\n\n    Arguments:\n      x: See `Model.fit`.\n      y: See `Model.fit`.\n      sample_weight: See `Model.fit`.\n      batch_size: See `Model.fit`.\n      steps_per_epoch: See `Model.fit`.\n      initial_epoch: See `Model.fit`.\n      epochs: See `Model.fit`.\n      shuffle: See `Model.fit`.\n      class_weight: See `Model.fit`.\n      max_queue_size: See `Model.fit`.\n      workers: See `Model.fit`.\n      use_multiprocessing: See `Model.fit`.\n      model: The `Model` instance. Needed in order to correctly `build` the\n        `Model` using generator-like inputs (see `GeneratorDataAdapter`).\n      steps_per_execution: See `Model.compile`.\n      distribute: Whether to distribute the `tf.dataset`.\n        `PreprocessingLayer.adapt` does not support distributed datasets,\n        `Model` should always set this to `True`.\n    '
        self._initial_epoch = initial_epoch
        self._epochs = epochs
        self._insufficient_data = False
        self._model = model
        if steps_per_execution is None:
            self._steps_per_execution = 1
            self._steps_per_execution_value = 1
        else:
            self._steps_per_execution = steps_per_execution
            self._steps_per_execution_value = steps_per_execution.numpy().item()
        adapter_cls = select_data_adapter(x, y)
        self._adapter = adapter_cls(x, y, batch_size=batch_size, steps=steps_per_epoch, epochs=epochs - initial_epoch, sample_weights=sample_weight, shuffle=shuffle, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing, distribution_strategy=distribute_lib.get_strategy(), model=model)
        strategy = distribute_lib.get_strategy()
        self._current_step = 0
        self._step_increment = self._steps_per_execution_value - 1
        self._insufficient_data = False
        self._configure_dataset_and_inferred_steps(strategy, x, steps_per_epoch, class_weight, distribute)

    def _configure_dataset_and_inferred_steps(self, strategy, x, steps_per_epoch, class_weight, distribute):
        if False:
            print('Hello World!')
        'Configure the `_dataset` and `_inferred_steps` attributes.'
        del x
        dataset = self._adapter.get_dataset()
        if class_weight:
            dataset = dataset.map(_make_class_weight_map_fn(class_weight))
        self._inferred_steps = self._infer_steps(steps_per_epoch, dataset)
        if distribute and (not _is_distributed_dataset(dataset)):
            dataset = strategy.experimental_distribute_dataset(dataset)
        self._dataset = dataset
        self._validate_data_handler()

    def enumerate_epochs(self):
        if False:
            while True:
                i = 10
        'Yields `(epoch, tf.data.Iterator)`.'
        with self._truncate_execution_to_epoch():
            data_iterator = iter(self._dataset)
            for epoch in range(self._initial_epoch, self._epochs):
                if self._insufficient_data:
                    break
                if self._adapter.should_recreate_iterator():
                    data_iterator = iter(self._dataset)
                yield (epoch, data_iterator)
                self._adapter.on_epoch_end()

    @contextlib.contextmanager
    def _truncate_execution_to_epoch(self):
        if False:
            return 10
        'Truncates steps per execution to at most one epoch.'
        should_truncate = self._inferred_steps is not None and self._steps_per_execution_value > self._inferred_steps
        original_value = self._steps_per_execution_value
        try:
            if should_truncate:
                self._steps_per_execution.assign(self._inferred_steps)
                self._steps_per_execution_value = self._inferred_steps
            yield
        finally:
            if should_truncate:
                self._steps_per_execution.assign(original_value)
                self._steps_per_execution_value = original_value

    def sync(self):
        if False:
            return 10
        context.async_wait()

    @contextlib.contextmanager
    def catch_stop_iteration(self):
        if False:
            for i in range(10):
                print('nop')
        'Catches errors when an iterator runs out of data.'
        try:
            yield
            self.sync()
        except (StopIteration, errors.OutOfRangeError):
            if self._inferred_steps is None:
                self._inferred_steps = self._current_step
            else:
                self._insufficient_data = True
                total_epochs = self._epochs - self._initial_epoch
                logging.warning('Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, {} batches). You may need to use the repeat() function when building your dataset.'.format(total_epochs * self._inferred_steps))

    def steps(self):
        if False:
            while True:
                i = 10
        'Yields steps for the current epoch.'
        self._current_step = 0
        while self._inferred_steps is None or self._current_step < self._inferred_steps:
            if self._insufficient_data:
                break
            can_run_full_execution = self._steps_per_execution_value == 1 or self._inferred_steps is None or self._inferred_steps - self._current_step >= self._steps_per_execution_value
            if can_run_full_execution:
                self._step_increment = self._steps_per_execution_value - 1
                yield self._current_step
                self._current_step += self._steps_per_execution_value
            else:
                steps_remaining = self._inferred_steps - self._current_step
                self._steps_per_execution.assign(steps_remaining)
                self._step_increment = steps_remaining - 1
                yield self._current_step
                self._current_step += steps_remaining
                self._steps_per_execution.assign(self._steps_per_execution_value)

    @property
    def step_increment(self):
        if False:
            for i in range(10):
                print('nop')
        'The number to increment the step for `on_batch_end` methods.'
        return self._step_increment

    @property
    def inferred_steps(self):
        if False:
            print('Hello World!')
        'The inferred steps per epoch of the created `Dataset`.\n\n    This will be `None` in the case where:\n\n    (1) A `Dataset` of unknown cardinality was passed to the `DataHandler`, and\n    (2) `steps_per_epoch` was not provided, and\n    (3) The first epoch of iteration has not yet completed.\n\n    Returns:\n      The inferred steps per epoch of the created `Dataset`.\n    '
        return self._inferred_steps

    @property
    def should_sync(self):
        if False:
            i = 10
            return i + 15
        return self._inferred_steps is None

    def _log_indefinite_training_warning(self):
        if False:
            while True:
                i = 10
        logging.warning('The training loop will run indefinitely since you have set `steps_per_epoch=-1`. Please use batch-level callbacks to save checkpoints or log training progress, etc')

    def _infer_steps(self, steps, dataset):
        if False:
            i = 10
            return i + 15
        'Infers steps_per_epoch needed to loop through a dataset.'
        if steps == -1:
            self._log_indefinite_training_warning()
            return None
        if steps is not None:
            return steps
        adapter_steps = self._adapter.get_size()
        if adapter_steps is not None:
            return adapter_steps
        size = cardinality.cardinality(dataset)
        if size == cardinality.INFINITE and steps is None:
            raise ValueError('When passing an infinitely repeating dataset, please specify a `steps_per_epoch` value so that epoch level callbacks continue to work. The value can be arbitrary, or a number that you think correctly defines the size of an epoch. Epoch-level callbacks will then be called at this interval.')
        if size >= 0:
            return size.numpy().item()
        return None

    @property
    def _samples(self):
        if False:
            while True:
                i = 10
        return self._adapter.get_samples()

    def _validate_data_handler(self):
        if False:
            i = 10
            return i + 15
        if self._steps_per_execution_value > 1 and self._inferred_steps is None:
            raise ValueError('Could not infer the size of the data. With `steps_per_execution > 1`, you must specify the number of steps to run.')

class _ClusterCoordinatorDataHandler(DataHandler):
    """A `DataHandler` that is compatible with `ClusterCoordinator`."""

    def __init__(self, x, y=None, **kwargs):
        if False:
            print('Hello World!')
        if not isinstance(x, dataset_creator.DatasetCreator):
            x = self._convert_to_dataset_creator(x, y, **kwargs)
        super().__init__(x=x, **kwargs)

    def _convert_to_dataset_creator(self, x, y, **kwargs):
        if False:
            print('Hello World!')
        'Converts non-tf.data.Dataset to `DatasetCreator` instances.'

        def _dataset_fn(input_context):
            if False:
                for i in range(10):
                    print('nop')
            del input_context
            data_adapter_cls = select_data_adapter(x, y)
            return data_adapter_cls(x=x, y=y, **kwargs).get_dataset()
        if isinstance(x, _get_tensor_types()) and isinstance(y, _get_tensor_types()):
            return dataset_creator.DatasetCreator(_dataset_fn)
        else:
            raise NotImplementedError('Only `tf.keras.utils.experimental.DatasetCreator`, `tf.Tensor`, numpy arrays and pandas dataframes are supported types at this time.')

    def _configure_dataset_and_inferred_steps(self, strategy, x, steps_per_epoch, class_weight, distribute):
        if False:
            print('Hello World!')
        if not isinstance(x, dataset_creator.DatasetCreator):
            raise TypeError('When using `ParameterServerStrategy`, `x` must be a `DatasetCreator`.')

        def per_worker_dataset_fn():
            if False:
                print('Hello World!')
            return strategy.distribute_datasets_from_function(x, options=x.input_options)
        self._dataset = self._model._cluster_coordinator.create_per_worker_dataset(per_worker_dataset_fn)
        if steps_per_epoch == -1:
            self._inferred_steps = None
            self._log_indefinite_training_warning()
        else:
            self._inferred_steps = steps_per_epoch

    def sync(self):
        if False:
            return 10
        self._model._cluster_coordinator.join()

def get_data_handler(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if getattr(kwargs['model'], '_cluster_coordinator', None):
        return _ClusterCoordinatorDataHandler(*args, **kwargs)
    return DataHandler(*args, **kwargs)

def _make_class_weight_map_fn(class_weight):
    if False:
        for i in range(10):
            print('nop')
    'Applies class weighting to a `Dataset`.\n\n  The `Dataset` is assumed to be in format `(x, y)` or `(x, y, sw)`, where\n  `y` must be a single `Tensor`.\n\n  Args:\n    class_weight: A map where the keys are integer class ids and values are\n      the class weights, e.g. `{0: 0.2, 1: 0.6, 2: 0.3}`\n\n  Returns:\n    A function that can be used with `tf.data.Dataset.map` to apply class\n    weighting.\n  '
    class_ids = list(sorted(class_weight.keys()))
    expected_class_ids = list(range(len(class_ids)))
    if class_ids != expected_class_ids:
        error_msg = 'Expected `class_weight` to be a dict with keys from 0 to one less than the number of classes, found {}'.format(class_weight)
        raise ValueError(error_msg)
    class_weight_tensor = tensor_conversion.convert_to_tensor_v2_with_dispatch([class_weight[int(c)] for c in class_ids])

    def _class_weights_map_fn(*data):
        if False:
            print('Hello World!')
        'Convert `class_weight` to `sample_weight`.'
        (x, y, sw) = unpack_x_y_sample_weight(data)
        if nest.is_nested(y):
            raise ValueError('`class_weight` is only supported for Models with a single output.')
        if y.shape.rank > 2:
            raise ValueError('`class_weight` not supported for 3+ dimensional targets.')
        y_classes = smart_cond.smart_cond(y.shape.rank == 2 and backend.shape(y)[1] > 1, lambda : backend.argmax(y, axis=1), lambda : math_ops.cast(backend.reshape(y, (-1,)), dtypes.int64))
        cw = array_ops.gather_v2(class_weight_tensor, y_classes)
        if sw is not None:
            cw = math_ops.cast(cw, sw.dtype)
            (sw, cw) = expand_1d((sw, cw))
            sw = sw * cw
        else:
            sw = cw
        return (x, y, sw)
    return _class_weights_map_fn

def expand_1d(data):
    if False:
        i = 10
        return i + 15
    'Expands 1-dimensional `Tensor`s into 2-dimensional `Tensor`s.'

    def _expand_single_1d_tensor(t):
        if False:
            return 10
        if isinstance(t, tensor.Tensor) and isinstance(t.shape, tensor_shape.TensorShape) and (t.shape.rank == 1):
            return array_ops.expand_dims_v2(t, axis=-1)
        return t
    return nest.map_structure(_expand_single_1d_tensor, data)

def train_validation_split(arrays, validation_split):
    if False:
        for i in range(10):
            print('nop')
    'Split arrays into train and validation subsets in deterministic order.\n\n  The last part of data will become validation data.\n\n  Args:\n    arrays: Tensors to split. Allowed inputs are arbitrarily nested structures\n      of Tensors and NumPy arrays.\n    validation_split: Float between 0 and 1. The proportion of the dataset to\n      include in the validation split. The rest of the dataset will be included\n      in the training split.\n  Returns:\n    `(train_arrays, validation_arrays)`\n  '

    def _can_split(t):
        if False:
            print('Hello World!')
        tensor_types = _get_tensor_types()
        return isinstance(t, tensor_types) or t is None
    flat_arrays = nest.flatten(arrays)
    unsplitable = [type(t) for t in flat_arrays if not _can_split(t)]
    if unsplitable:
        raise ValueError('`validation_split` is only supported for Tensors or NumPy arrays, found following types in the input: {}'.format(unsplitable))
    if all((t is None for t in flat_arrays)):
        return (arrays, arrays)
    first_non_none = None
    for t in flat_arrays:
        if t is not None:
            first_non_none = t
            break
    batch_dim = int(first_non_none.shape[0])
    split_at = int(math.floor(batch_dim * (1.0 - validation_split)))
    if split_at == 0 or split_at == batch_dim:
        raise ValueError('Training data contains {batch_dim} samples, which is not sufficient to split it into a validation and training set as specified by `validation_split={validation_split}`. Either provide more data, or a different value for the `validation_split` argument.'.format(batch_dim=batch_dim, validation_split=validation_split))

    def _split(t, start, end):
        if False:
            while True:
                i = 10
        if t is None:
            return t
        return t[start:end]
    train_arrays = nest.map_structure(functools.partial(_split, start=0, end=split_at), arrays)
    val_arrays = nest.map_structure(functools.partial(_split, start=split_at, end=batch_dim), arrays)
    return (train_arrays, val_arrays)

def unpack_x_y_sample_weight(data):
    if False:
        return 10
    'Unpacks user-provided data tuple.\n\n  This is a convenience utility to be used when overriding\n  `Model.train_step`, `Model.test_step`, or `Model.predict_step`.\n  This utility makes it easy to support data of the form `(x,)`,\n  `(x, y)`, or `(x, y, sample_weight)`.\n\n  Standalone usage:\n\n  >>> features_batch = tf.ones((10, 5))\n  >>> labels_batch = tf.zeros((10, 5))\n  >>> data = (features_batch, labels_batch)\n  >>> # `y` and `sample_weight` will default to `None` if not provided.\n  >>> x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)\n  >>> sample_weight is None\n  True\n\n  Example in overridden `Model.train_step`:\n\n  ```python\n  class MyModel(tf.keras.Model):\n\n    def train_step(self, data):\n      # If `sample_weight` is not provided, all samples will be weighted\n      # equally.\n      x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)\n\n      with tf.GradientTape() as tape:\n        y_pred = self(x, training=True)\n        loss = self.compiled_loss(\n          y, y_pred, sample_weight, regularization_losses=self.losses)\n        trainable_variables = self.trainable_variables\n        gradients = tape.gradient(loss, trainable_variables)\n        self.optimizer.apply_gradients(zip(gradients, trainable_variables))\n\n      self.compiled_metrics.update_state(y, y_pred, sample_weight)\n      return {m.name: m.result() for m in self.metrics}\n  ```\n\n  Args:\n    data: A tuple of the form `(x,)`, `(x, y)`, or `(x, y, sample_weight)`.\n\n  Returns:\n    The unpacked tuple, with `None`s for `y` and `sample_weight` if they are not\n    provided.\n  '
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])
    else:
        error_msg = 'Data is expected to be in format `x`, `(x,)`, `(x, y)`, or `(x, y, sample_weight)`, found: {}'.format(data)
        raise ValueError(error_msg)

def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    if False:
        while True:
            i = 10
    'Packs user-provided data into a tuple.\n\n  This is a convenience utility for packing data into the tuple formats\n  that `Model.fit` uses.\n\n  Standalone usage:\n\n  >>> x = tf.ones((10, 1))\n  >>> data = tf.keras.utils.pack_x_y_sample_weight(x)\n  >>> isinstance(data, tf.Tensor)\n  True\n  >>> y = tf.ones((10, 1))\n  >>> data = tf.keras.utils.pack_x_y_sample_weight(x, y)\n  >>> isinstance(data, tuple)\n  True\n  >>> x, y = data\n\n  Args:\n    x: Features to pass to `Model`.\n    y: Ground-truth targets to pass to `Model`.\n    sample_weight: Sample weight for each element.\n\n  Returns:\n    Tuple in the format used in `Model.fit`.\n  '
    if y is None:
        if not nest.is_nested(x):
            return x
        else:
            return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)

def single_batch_iterator(strategy, x, y=None, sample_weight=None, class_weight=None):
    if False:
        return 10
    'Creates a single-batch dataset.'
    (x, y, sample_weight) = _process_tensorlike((x, y, sample_weight))
    if y is None:
        data = (x,)
    elif sample_weight is None:
        data = (x, y)
    else:
        data = (x, y, sample_weight)
    _check_data_cardinality(data)
    dataset = dataset_ops.DatasetV2.from_tensors(data)
    if class_weight:
        dataset = dataset.map(_make_class_weight_map_fn(class_weight))
    dataset = strategy.experimental_distribute_dataset(dataset)
    return iter(dataset)

def _check_data_cardinality(data):
    if False:
        return 10
    num_samples = set((int(i.shape[0]) for i in nest.flatten(data)))
    if len(num_samples) > 1:
        msg = 'Data cardinality is ambiguous:\n'
        for (label, single_data) in zip(['x', 'y', 'sample_weight'], data):
            msg += '  {} sizes: {}\n'.format(label, ', '.join((str(i.shape[0]) for i in nest.flatten(single_data))))
        msg += 'Make sure all arrays contain the same number of samples.'
        raise ValueError(msg)

def _get_tensor_types():
    if False:
        i = 10
        return i + 15
    try:
        import pandas as pd
        return (tensor.Tensor, np.ndarray, pd.Series, pd.DataFrame)
    except ImportError:
        return (tensor.Tensor, np.ndarray)

def _is_scipy_sparse(x):
    if False:
        while True:
            i = 10
    try:
        from scipy.sparse import issparse
        return issparse(x)
    except ImportError:
        return False

def _scipy_sparse_to_sparse_tensor(t):
    if False:
        for i in range(10):
            print('nop')
    'Converts a SciPy sparse matrix to a SparseTensor.'
    sparse_coo = t.tocoo()
    (row, col) = (sparse_coo.row, sparse_coo.col)
    (data, shape) = (sparse_coo.data, sparse_coo.shape)
    if issubclass(data.dtype.type, np.floating):
        data = data.astype(backend.floatx())
    indices = np.concatenate((np.expand_dims(row, axis=1), np.expand_dims(col, axis=1)), axis=1)
    return sparse_tensor.SparseTensor(indices, data, shape)

def _is_distributed_dataset(ds):
    if False:
        while True:
            i = 10
    return isinstance(ds, input_lib.DistributedDatasetInterface)