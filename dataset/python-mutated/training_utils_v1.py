"""Training-related utilities."""
import abc
import atexit
import collections
import functools
import multiprocessing.pool
import threading
import time
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest

def is_composite_or_composite_value(tensor):
    if False:
        return 10
    "Returns true if 'tensor' is a CompositeTensor or a CT Value object."
    return isinstance(tensor, (composite_tensor.CompositeTensor, sparse_tensor.SparseTensorValue, ragged_tensor_value.RaggedTensorValue))

class Aggregator(object, metaclass=abc.ABCMeta):
    """Abstract base class used to aggregate batch-level outputs of a loop.

  Attributes:
    use_steps: Whether the loop is using `step` or `batch_size`.
    num_samples: Total number of samples: `batch_size * num_batches`.
    steps: Total number of steps.
    batch_size: Batch size. It is used for validation checks between inputs and
      outputs.
    results: What to return at the end of the aggregation loop.
  """

    def __init__(self, use_steps, num_samples=None, steps=None, batch_size=None):
        if False:
            i = 10
            return i + 15
        self.use_steps = use_steps
        self.num_samples = num_samples
        self.steps = steps
        self.batch_size = batch_size
        self.results = []

    @abc.abstractmethod
    def create(self, batch_outs):
        if False:
            print('Hello World!')
        'Creates the initial results from the first batch outputs.\n\n    Args:\n      batch_outs: A list of batch-level outputs.\n    '
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def aggregate(self, batch_outs, batch_start=None, batch_end=None):
        if False:
            return 10
        'Aggregates batch-level results into total results.\n\n    Args:\n      batch_outs: A list of batch-level outputs.\n      batch_start: The start index of this batch. Always `None` if `use_steps`\n        is `True`.\n      batch_end: The end index of this batch. Always `None` if `use_steps` is\n        `True`.\n    '
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def finalize(self):
        if False:
            print('Hello World!')
        'Prepares the total results to be returned.'
        raise NotImplementedError('Must be implemented in subclasses.')

class MetricsAggregator(Aggregator):
    """Aggregator that calculates loss and metrics info.

  Attributes:
    use_steps: Whether the loop is using `step` or `batch_size`.
    num_samples: Total number of samples: `batch_size*num_batches`.
    steps: Total number of steps, ie number of times to iterate over a dataset
      to cover all samples.
  """

    def __init__(self, use_steps, num_samples=None, steps=None):
        if False:
            i = 10
            return i + 15
        super(MetricsAggregator, self).__init__(use_steps=use_steps, num_samples=num_samples, steps=steps, batch_size=None)

    def create(self, batch_outs):
        if False:
            while True:
                i = 10
        self.results = [0.0] * len(batch_outs)

    def aggregate(self, batch_outs, batch_start=None, batch_end=None):
        if False:
            for i in range(10):
                print('nop')
        if self.use_steps:
            self.results[0] += batch_outs[0]
        else:
            self.results[0] += batch_outs[0] * (batch_end - batch_start)
        self.results[1:] = batch_outs[1:]

    def finalize(self):
        if False:
            while True:
                i = 10
        if not self.results:
            raise ValueError('Empty training data.')
        self.results[0] /= self.num_samples or self.steps

def _append_sparse_tensor_value(target, to_append):
    if False:
        while True:
            i = 10
    'Append sparse tensor value objects.'
    if len(target.dense_shape) != len(to_append.dense_shape):
        raise RuntimeError('Unable to concatenate %s and %s. The inner dense shapes do not have the same number of dimensions (%s vs %s)' % (target, to_append, target.dense_shape, to_append.dense_shape))
    if target.dense_shape[1:] != to_append.dense_shape[1:]:
        raise RuntimeError('Unable to concatenate %s and %s. The inner dense shapes do not match inner dimensions (%s vs %s)' % (target, to_append, target.dense_shape[1:], to_append.dense_shape[1:]))
    base_dim0_value = target.dense_shape[0]
    max_dim0_value = target.dense_shape[0]
    new_indices = target.indices
    for index in to_append.indices:
        index[0] += base_dim0_value
        max_dim0_value = max(max_dim0_value, index[0])
        new_indices = np.append(new_indices, [index], axis=0)
    new_values = np.concatenate((target.values, to_append.values), axis=0)
    new_dense_shape = list(target.dense_shape)
    new_dense_shape[0] = max_dim0_value + 1
    new_dense_shape = tuple(new_dense_shape)
    return sparse_tensor.SparseTensorValue(indices=new_indices, values=new_values, dense_shape=new_dense_shape)

def _append_ragged_tensor_value(target, to_append):
    if False:
        i = 10
        return i + 15
    'Append ragged tensor value objects.'
    if len(target.shape) != len(to_append.shape):
        raise RuntimeError('Unable to concatenate %s and %s' % (target, to_append))
    if target.shape[1:] != to_append.shape[1:]:
        raise RuntimeError('Unable to concatenate %s and %s' % (target, to_append))
    adjusted_row_splits = to_append.row_splits[1:] + target.row_splits[-1]
    new_row_splits = np.append(target.row_splits, adjusted_row_splits)
    if isinstance(target.values, ragged_tensor_value.RaggedTensorValue):
        new_values = _append_ragged_tensor_value(target.values, to_append.values)
    else:
        new_values = np.concatenate((target.values, to_append.values), axis=0)
    return ragged_tensor_value.RaggedTensorValue(new_values, new_row_splits)

def _append_composite_tensor(target, to_append):
    if False:
        return 10
    "Helper function to append composite tensors to each other in the 0 axis.\n\n  In order to support batching within a fit/evaluate/predict call, we need\n  to be able to aggregate within a CompositeTensor. Unfortunately, the CT\n  API currently does not make this easy - especially in V1 mode, where we're\n  working with CompositeTensor Value objects that have no connection with the\n  CompositeTensors that created them.\n\n  Args:\n    target: CompositeTensor or CompositeTensor value object that will be\n      appended to.\n    to_append: CompositeTensor or CompositeTensor value object to append to.\n      'target'.\n\n  Returns:\n    A CompositeTensor or CompositeTensor value object.\n\n  Raises:\n    RuntimeError: if concatenation is not possible.\n  "
    if type(target) is not type(to_append):
        raise RuntimeError('Unable to concatenate %s and %s' % (type(target), type(to_append)))
    if isinstance(target, sparse_tensor.SparseTensor):
        return sparse_ops.sparse_concat(sp_inputs=[target, to_append], axis=0)
    elif isinstance(target, ragged_tensor.RaggedTensor):
        return array_ops.concat([target, to_append], axis=0)
    elif isinstance(target, sparse_tensor.SparseTensorValue):
        return _append_sparse_tensor_value(target, to_append)
    elif isinstance(target, ragged_tensor_value.RaggedTensorValue):
        return _append_ragged_tensor_value(target, to_append)
    else:
        raise RuntimeError('Attempted to concatenate unsupported object %s.' % type(target))

class ConcatAggregator(Aggregator):
    """Combine tensor-likes which cannot be merged on the fly.

  This class expects to aggregate a single tensor-like rather than a nested
  structure of tensor-likes.
  """

    def __init__(self, batch_size):
        if False:
            return 10
        self.composite = None
        super(ConcatAggregator, self).__init__(use_steps=True, num_samples=None, steps=None, batch_size=batch_size)

    def create(self, batch_element):
        if False:
            i = 10
            return i + 15
        self.composite = is_composite_or_composite_value(batch_element)

    def aggregate(self, batch_element, batch_start=None, batch_end=None):
        if False:
            for i in range(10):
                print('nop')
        if self.batch_size and self.batch_size < batch_element.shape[0]:
            raise ValueError('Mismatch between expected batch size and model output batch size. Output shape = {}, expected output shape = shape {}'.format(batch_element.shape, (self.batch_size,) + batch_element.shape[1:]))
        self.results.append(batch_element)

    def finalize(self):
        if False:
            while True:
                i = 10
        if len(self.results) == 1:
            self.results = self.results[0]
        elif self.composite:
            results = self.results[0]
            for r in self.results[1:]:
                results = _append_composite_tensor(results, r)
            self.results = results
        else:
            self.results = np.concatenate(self.results, axis=0)
_COPY_THREADS = 4
_COPY_POOL = None

def get_copy_pool():
    if False:
        for i in range(10):
            print('nop')
    'Shared threadpool for copying arrays.\n\n  Pool instantiation takes ~ 2ms, so a singleton pool is used rather than\n  creating a pool per SliceAggregator.\n\n  Returns:\n    The global copy threadpool.\n  '
    global _COPY_POOL
    if _COPY_POOL is None:
        _COPY_POOL = multiprocessing.pool.ThreadPool(_COPY_THREADS)
        atexit.register(_COPY_POOL.close)
    return _COPY_POOL

class SliceAggregator(Aggregator):
    """Combine arrays where the final size is known.

  This class expects to aggregate a single tensor-like rather than a nested
  structure of tensor-likes.

  NumPy copies are an operation that threads handle quite well because all of
  the heavy lifting is in c and does not need the GIL. Moreover, we can perform
  lock-free writes to the same buffer in multiple threads because the nature of
  result aggregation guarantees that either the indices are disjoint or the
  aggregator will throw an exception in finalize. Moreover, because aggregation
  is performed on the slowest varying dimension, assignments for a given batch
  will write to contiguous blocks of memory, further minimizing contention.

  There is, however, some scheduling and context switching overhead which will
  offset the gains from pipelining the slice assignment. Below a given threshold
  it is faster to simply assign in the main thread rather than enqueue the
  assignment in a side thread. The exact threshold will vary from system to
  system, but the time is not very sensitive to the exact transition so a value
  of 2 ** 14 was chosen which should be reasonable on most systems.
  """
    _BINARY_SIZE_THRESHOLD = 2 ** 14
    _MAX_COPY_SECONDS = 300

    def __init__(self, num_samples, batch_size):
        if False:
            return 10
        self._async_copies = []
        self._pool = get_copy_pool()
        self._errors = []
        super(SliceAggregator, self).__init__(use_steps=False, num_samples=num_samples, steps=None, batch_size=batch_size)

    def create(self, batch_element):
        if False:
            while True:
                i = 10
        shape = (self.num_samples,) + batch_element.shape[1:]
        dtype = batch_element.dtype
        self.results = np.empty(shape=shape, dtype=dtype)

    def aggregate(self, batch_element, batch_start, batch_end):
        if False:
            i = 10
            return i + 15
        if self._errors:
            raise self._errors[0]
        if batch_end - batch_start == self.num_samples:
            if self.num_samples != batch_element.shape[0]:
                raise ValueError('Mismatch between expected batch size and model output batch size. Output shape = {}, expected output shape = shape {}'.format(batch_element.shape, self.results.shape))
            self.results = batch_element
            return
        num_elements = np.prod(batch_element.shape)
        if num_elements < self._BINARY_SIZE_THRESHOLD:
            self.results[batch_start:batch_end] = batch_element
        else:
            is_finished = threading.Event()
            self._pool.apply_async(self._slice_assign, args=(batch_element, batch_start, batch_end, is_finished))
            self._async_copies.append(is_finished)

    def _slice_assign(self, batch_element, batch_start, batch_end, is_finished):
        if False:
            while True:
                i = 10
        'Legacy utility method to slice input arrays.'
        try:
            self.results[batch_start:batch_end] = batch_element
        except Exception as e:
            self._errors.append(e)
        finally:
            is_finished.set()

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        start_time = time.time()
        for is_finished in self._async_copies:
            timeout = max([0.0, self._MAX_COPY_SECONDS - (time.time() - start_time)])
            if not is_finished.wait(timeout):
                raise ValueError('Timed out waiting for copy to complete.')
        if self._errors:
            raise self._errors[0]

class OutputsAggregator(Aggregator):
    """Aggregator that concatenates outputs."""
    _structure = None

    def create(self, batch_outs):
        if False:
            while True:
                i = 10
        self._structure = nest.get_traverse_shallow_structure(lambda x: not is_composite_or_composite_value(x), batch_outs)
        batch_outs = nest.flatten_up_to(self._structure, batch_outs)
        for batch_element in batch_outs:
            if is_composite_or_composite_value(batch_element):
                self.results.append(ConcatAggregator(self.batch_size))
            elif isinstance(batch_element, np.ndarray):
                self.results.append(ConcatAggregator(self.batch_size) if self.use_steps else SliceAggregator(self.num_samples, self.batch_size))
            else:
                raise RuntimeError('Attempted to aggregate unsupported object {}.'.format(batch_element))
            self.results[-1].create(batch_element)

    def aggregate(self, batch_outs, batch_start=None, batch_end=None):
        if False:
            while True:
                i = 10
        batch_outs = nest.flatten_up_to(self._structure, batch_outs)
        for (batch_element, result) in zip(batch_outs, self.results):
            result.aggregate(batch_element, batch_start, batch_end)

    def finalize(self):
        if False:
            print('Hello World!')
        for result in self.results:
            result.finalize()
        self.results = [i.results for i in self.results]
        self.results = nest.pack_sequence_as(self._structure, self.results)

def get_progbar(model, count_mode, include_metrics=True):
    if False:
        for i in range(10):
            print('nop')
    'Get Progbar.'
    if include_metrics:
        stateful_metric_names = getattr(model, 'metrics_names', None)
        if stateful_metric_names:
            stateful_metric_names = stateful_metric_names[1:]
    else:
        stateful_metric_names = None
    return cbks.ProgbarLogger(count_mode, stateful_metrics=stateful_metric_names)

def check_num_samples(ins, batch_size=None, steps=None, steps_name='steps'):
    if False:
        i = 10
        return i + 15
    "Determine the number of samples provided for training and evaluation.\n\n  The number of samples is not defined when running with `steps`,\n  in which case the number of samples is set to `None`.\n\n  Args:\n      ins: List of tensors to be fed to the Keras function.\n      batch_size: Integer batch size or `None` if not defined.\n      steps: Total number of steps (batches of samples) before declaring\n        `_predict_loop` finished. Ignored with the default value of `None`.\n      steps_name: The public API's parameter name for `steps`.\n\n  Raises:\n      ValueError: when `steps` is `None` and the attribute `ins.shape`\n      does not exist. Also raises ValueError when `steps` is not `None`\n      and `batch_size` is not `None` because they are mutually\n      exclusive.\n\n  Returns:\n      When steps is `None`, returns the number of samples to be\n      processed based on the size of the first dimension of the\n      first input numpy array. When steps is not `None` and\n      `batch_size` is `None`, returns `None`.\n  "
    if steps is not None and batch_size is not None:
        raise ValueError('If ' + steps_name + ' is set, the `batch_size` must be None.')
    if check_steps_argument(ins, steps, steps_name):
        return None
    if hasattr(ins[0], 'shape'):
        return int(ins[0].shape[0])
    return None

def standardize_single_array(x, expected_shape=None):
    if False:
        i = 10
        return i + 15
    'Expand data of shape (x,) to (x, 1), unless len(expected_shape)==1.'
    if x is None:
        return None
    if is_composite_or_composite_value(x):
        return x
    if isinstance(x, int):
        raise ValueError('Expected an array data type but received an integer: {}'.format(x))
    if x.shape is not None and len(x.shape) == 1 and (expected_shape is None or len(expected_shape) != 1):
        if tensor_util.is_tf_type(x):
            x = array_ops.expand_dims(x, axis=1)
        else:
            x = np.expand_dims(x, 1)
    return x

def get_composite_shape(tensor):
    if False:
        i = 10
        return i + 15
    'Returns the shape of the passed composite tensor.'
    if isinstance(tensor, sparse_tensor.SparseTensorValue):
        return tensor.dense_shape
    else:
        return tensor.shape

def standardize_input_data(data, names, shapes=None, check_batch_axis=True, exception_prefix=''):
    if False:
        print('Hello World!')
    "Normalizes inputs and targets provided by users.\n\n  Users may pass data as a list of arrays, dictionary of arrays,\n  or as a single array. We normalize this to an ordered list of\n  arrays (same order as `names`), while checking that the provided\n  arrays have shapes that match the network's expectations.\n\n  Args:\n      data: User-provided input data (polymorphic).\n      names: List of expected array names.\n      shapes: Optional list of expected array shapes.\n      check_batch_axis: Boolean; whether to check that the batch axis of the\n        arrays matches the expected value found in `shapes`.\n      exception_prefix: String prefix used for exception formatting.\n\n  Returns:\n      List of standardized input arrays (one array per model input).\n\n  Raises:\n      ValueError: in case of improperly formatted user-provided data.\n  "
    try:
        data_len = len(data)
    except TypeError:
        data_len = None
    if not names:
        if data_len and (not isinstance(data, dict)):
            raise ValueError('Error when checking model ' + exception_prefix + ': expected no data, but got:', data)
        return []
    if data is None:
        return [None for _ in range(len(names))]
    if isinstance(data, dict):
        try:
            data = [data[x].values if data[x].__class__.__name__ == 'DataFrame' else data[x] for x in names]
        except KeyError as e:
            raise ValueError('No data provided for "' + e.args[0] + '". Need data for each key in: ' + str(names))
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], (list, tuple)):
            data = [np.asarray(d) for d in data]
        elif len(names) == 1 and isinstance(data[0], (float, int)):
            data = [np.asarray(data)]
        else:
            data = [x.values if x.__class__.__name__ == 'DataFrame' else x for x in data]
    else:
        data = data.values if data.__class__.__name__ == 'DataFrame' else data
        data = [data]
    if shapes is not None:
        data = [standardize_single_array(x, shape) for (x, shape) in zip(data, shapes)]
    else:
        data = [standardize_single_array(x) for x in data]
    if len(data) != len(names):
        if data and hasattr(data[0], 'shape'):
            raise ValueError('Error when checking model ' + exception_prefix + ': the list of Numpy arrays that you are passing to your model is not the size the model expected. Expected to see ' + str(len(names)) + ' array(s), ' + 'for inputs ' + str(names) + ' but instead got the following list of ' + str(len(data)) + ' arrays: ' + str(data)[:200] + '...')
        elif len(names) > 1:
            raise ValueError('Error when checking model ' + exception_prefix + ': you are passing a list as input to your model, but the model expects a list of ' + str(len(names)) + ' Numpy arrays instead. The list you passed was: ' + str(data)[:200])
        elif len(data) == 1 and (not hasattr(data[0], 'shape')):
            raise TypeError('Error when checking model ' + exception_prefix + ': data should be a Numpy array, or list/dict of Numpy arrays. Found: ' + str(data)[:200] + '...')
        elif len(names) == 1:
            data = [np.asarray(data)]
    if shapes:
        for i in range(len(names)):
            if shapes[i] is not None:
                if tensor_util.is_tf_type(data[i]):
                    tensorshape = data[i].shape
                    if not tensorshape:
                        continue
                    data_shape = tuple(tensorshape.as_list())
                elif is_composite_or_composite_value(data[i]):
                    tensorshape = get_composite_shape(data[i])
                    data_shape = tuple(tensorshape.as_list())
                else:
                    data_shape = data[i].shape
                shape = shapes[i]
                if len(data_shape) != len(shape):
                    raise ValueError('Error when checking ' + exception_prefix + ': expected ' + names[i] + ' to have ' + str(len(shape)) + ' dimensions, but got array with shape ' + str(data_shape))
                if not check_batch_axis:
                    data_shape = data_shape[1:]
                    shape = shape[1:]
                for (dim, ref_dim) in zip(data_shape, shape):
                    if ref_dim != dim and ref_dim is not None and (dim is not None):
                        raise ValueError('Error when checking ' + exception_prefix + ': expected ' + names[i] + ' to have shape ' + str(shape) + ' but got array with shape ' + str(data_shape))
    return data

def standardize_sample_or_class_weights(x_weight, output_names, weight_type):
    if False:
        for i in range(10):
            print('nop')
    'Maps `sample_weight` or `class_weight` to model outputs.\n\n  Args:\n      x_weight: User-provided `sample_weight` or `class_weight` argument.\n      output_names: List of output names (strings) in the model.\n      weight_type: A string used purely for exception printing.\n\n  Returns:\n      A list of `sample_weight` or `class_weight` where there are exactly\n          one element per model output.\n\n  Raises:\n      ValueError: In case of invalid user-provided argument.\n  '
    if x_weight is None or (isinstance(x_weight, (list, tuple)) and len(x_weight) == 0):
        return [None for _ in output_names]
    if len(output_names) == 1:
        if isinstance(x_weight, (list, tuple)) and len(x_weight) == 1:
            return x_weight
        if isinstance(x_weight, dict) and output_names[0] in x_weight:
            return [x_weight[output_names[0]]]
        else:
            return [x_weight]
    if isinstance(x_weight, (list, tuple)):
        if len(x_weight) != len(output_names):
            raise ValueError('Provided `' + weight_type + '` was a list of ' + str(len(x_weight)) + ' elements, but the model has ' + str(len(output_names)) + ' outputs. You should provide one `' + weight_type + '`array per model output.')
        return x_weight
    if isinstance(x_weight, collections.abc.Mapping):
        generic_utils.check_for_unexpected_keys(weight_type, x_weight, output_names)
        x_weights = []
        for name in output_names:
            x_weights.append(x_weight.get(name))
        return x_weights
    else:
        raise TypeError('The model has multiple outputs, so `' + weight_type + '` should be either a list or a dict. Provided `' + weight_type + '` type not understood: ' + str(x_weight))

def standardize_class_weights(class_weight, output_names):
    if False:
        i = 10
        return i + 15
    return standardize_sample_or_class_weights(class_weight, output_names, 'class_weight')

def standardize_sample_weights(sample_weight, output_names):
    if False:
        print('Hello World!')
    return standardize_sample_or_class_weights(sample_weight, output_names, 'sample_weight')

def check_array_lengths(inputs, targets, weights=None):
    if False:
        print('Hello World!')
    'Does user input validation for numpy arrays.\n\n  Args:\n      inputs: list of Numpy arrays of inputs.\n      targets: list of Numpy arrays of targets.\n      weights: list of Numpy arrays of sample weights.\n\n  Raises:\n      ValueError: in case of incorrectly formatted data.\n  '

    def is_tensor_or_composite_tensor(x):
        if False:
            i = 10
            return i + 15
        return tensor_util.is_tf_type(x) or is_composite_or_composite_value(x)

    def set_of_lengths(x):
        if False:
            i = 10
            return i + 15
        if x is None:
            return {}
        else:
            return set([y.shape[0] for y in x if y is not None and (not is_tensor_or_composite_tensor(y))])
    set_x = set_of_lengths(inputs)
    set_y = set_of_lengths(targets)
    set_w = set_of_lengths(weights)
    if len(set_x) > 1:
        raise ValueError('All input arrays (x) should have the same number of samples. Got array shapes: ' + str([x.shape for x in inputs]))
    if len(set_y) > 1:
        raise ValueError('All target arrays (y) should have the same number of samples. Got array shapes: ' + str([y.shape for y in targets]))
    if set_x and set_y and (list(set_x)[0] != list(set_y)[0]):
        raise ValueError('Input arrays should have the same number of samples as target arrays. Found ' + str(list(set_x)[0]) + ' input samples and ' + str(list(set_y)[0]) + ' target samples.')
    if len(set_w) > 1:
        raise ValueError('All sample_weight arrays should have the same number of samples. Got array shapes: ' + str([w.shape for w in weights]))
    if set_y and set_w and (list(set_y)[0] != list(set_w)[0]):
        raise ValueError('Sample_weight arrays should have the same number of samples as target arrays. Got ' + str(list(set_y)[0]) + ' input samples and ' + str(list(set_w)[0]) + ' target samples.')

def check_loss_and_target_compatibility(targets, loss_fns, output_shapes):
    if False:
        return 10
    'Does validation on the compatibility of targets and loss functions.\n\n  This helps prevent users from using loss functions incorrectly. This check\n  is purely for UX purposes.\n\n  Args:\n      targets: list of Numpy arrays of targets.\n      loss_fns: list of loss functions.\n      output_shapes: list of shapes of model outputs.\n\n  Raises:\n      ValueError: if a loss function or target array\n          is incompatible with an output.\n  '
    key_loss_fns = {losses.mean_squared_error, losses.binary_crossentropy, losses.categorical_crossentropy}
    key_loss_classes = (losses.MeanSquaredError, losses.BinaryCrossentropy, losses.CategoricalCrossentropy)
    for (y, loss, shape) in zip(targets, loss_fns, output_shapes):
        if y is None or loss is None or tensor_util.is_tf_type(y):
            continue
        if losses.is_categorical_crossentropy(loss):
            if y.shape[-1] == 1:
                raise ValueError('You are passing a target array of shape ' + str(y.shape) + ' while using as loss `categorical_crossentropy`. `categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape (samples, classes). If your targets are integer classes, you can convert them to the expected format via:\n```\nfrom keras.utils import to_categorical\ny_binary = to_categorical(y_int)\n```\n\nAlternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.')
        is_loss_wrapper = isinstance(loss, losses.LossFunctionWrapper)
        if isinstance(loss, key_loss_classes) or (is_loss_wrapper and loss.fn in key_loss_fns):
            for (target_dim, out_dim) in zip(y.shape[1:], shape[1:]):
                if out_dim is not None and target_dim != out_dim:
                    loss_name = loss.name
                    if loss_name is None:
                        loss_type = loss.fn if is_loss_wrapper else type(loss)
                        loss_name = loss_type.__name__
                    raise ValueError('A target array with shape ' + str(y.shape) + ' was passed for an output of shape ' + str(shape) + ' while using as loss `' + loss_name + '`. This loss expects targets to have the same shape as the output.')

def collect_per_output_metric_info(metrics, output_names, output_shapes, loss_fns, from_serialized=False, is_weighted=False):
    if False:
        return 10
    'Maps metric names and functions to model outputs.\n\n  Args:\n      metrics: a list or a list of lists or a dict of metric functions.\n      output_names: a list of the names (strings) of model outputs.\n      output_shapes: a list of the shapes (strings) of model outputs.\n      loss_fns: a list of the loss functions corresponding to the model outputs.\n      from_serialized: whether the model the metrics are being sourced from is\n        being initialized from a serialized format.\n      is_weighted: Boolean indicating whether the given metrics are weighted.\n\n  Returns:\n      A list (one entry per model output) of dicts.\n      For instance, if the model has 2 outputs, and for the first output\n      we want to compute "binary_accuracy" and "binary_crossentropy",\n      and just "binary_accuracy" for the second output,\n      the list would look like: `[{\n          \'acc\': binary_accuracy(),\n          \'ce\': binary_crossentropy(),\n        }, {\n          \'acc\': binary_accuracy(),\n        }]`\n\n  Raises:\n      TypeError: if an incorrect type is passed for the `metrics` argument.\n  '
    if not metrics:
        return [{} for _ in output_names]
    if isinstance(metrics, list):
        any_sub_list = any((isinstance(m, list) for m in metrics))
        if any_sub_list:
            if len(metrics) != len(output_names):
                raise ValueError('When passing a list of lists as `metrics`, it should have one entry per model output. The model has ' + str(len(output_names)) + ' outputs, but you passed metrics=' + str(metrics))
            nested_metrics = [generic_utils.to_list(m) for m in metrics]
        elif len(output_names) > 1:
            nested_metrics = []
            for _ in output_names:
                nested_metrics.append([metrics_module.clone_metric(m) for m in metrics])
        else:
            nested_metrics = [metrics]
    elif isinstance(metrics, collections.abc.Mapping):
        generic_utils.check_for_unexpected_keys('metrics', metrics, output_names)
        nested_metrics = []
        for name in output_names:
            output_metrics = generic_utils.to_list(metrics.get(name, []))
            nested_metrics.append(output_metrics)
    else:
        raise TypeError('Type of `metrics` argument not understood. Expected a list or dictionary, found: ' + str(metrics))
    per_output_metrics = []
    for (i, metrics) in enumerate(nested_metrics):
        metrics_dict = collections.OrderedDict()
        for metric in metrics:
            metric_name = get_metric_name(metric, is_weighted)
            metric_fn = get_metric_function(metric, output_shape=output_shapes[i], loss_fn=loss_fns[i])
            metric_fn._from_serialized = from_serialized
            if not isinstance(metric_fn, metrics_module.Metric):
                metric_fn = metrics_module.MeanMetricWrapper(metric_fn, name=metric_name)
                metric_fn._from_serialized = False
            metrics_dict[metric_name] = metric_fn
        per_output_metrics.append(metrics_dict)
    return per_output_metrics

def batch_shuffle(index_array, batch_size):
    if False:
        i = 10
        return i + 15
    'Shuffles an array in a batch-wise fashion.\n\n  Useful for shuffling HDF5 arrays\n  (where one cannot access arbitrary indices).\n\n  Args:\n      index_array: array of indices to be shuffled.\n      batch_size: integer.\n\n  Returns:\n      The `index_array` array, shuffled in a batch-wise fashion.\n  '
    batch_count = int(len(index_array) / batch_size)
    last_batch = index_array[batch_count * batch_size:]
    index_array = index_array[:batch_count * batch_size]
    index_array = index_array.reshape((batch_count, batch_size))
    np.random.shuffle(index_array)
    index_array = index_array.flatten()
    return np.append(index_array, last_batch)

def standardize_weights(y, sample_weight=None, class_weight=None, sample_weight_mode=None):
    if False:
        for i in range(10):
            print('nop')
    'Performs sample weight validation and standardization.\n\n  Everything gets normalized to a single sample-wise (or timestep-wise)\n  weight array. If both `sample_weight` and `class_weight` are provided,\n  the weights are multiplied.\n\n  Args:\n      y: Numpy array or Tensor of model targets to be weighted.\n      sample_weight: User-provided `sample_weight` argument.\n      class_weight: User-provided `class_weight` argument.\n      sample_weight_mode: One of `None` or `"temporal"`. `"temporal"` indicated\n        that we expect 2D weight data that will be applied to the last 2\n        dimensions of the targets (i.e. we are weighting timesteps, not\n        samples).\n\n  Returns:\n      A numpy array of target weights, one entry per sample to weight.\n\n  Raises:\n      ValueError: In case of invalid user-provided arguments.\n  '
    if isinstance(sample_weight, tuple):
        sample_weight = sample_weight[0]
    if sample_weight_mode is not None and sample_weight_mode != 'samplewise':
        if sample_weight_mode != 'temporal':
            raise ValueError('"sample_weight_mode should be None or "temporal". Found: ' + str(sample_weight_mode))
        if len(y.shape) < 3:
            raise ValueError('Found a sample_weight array for an input with shape ' + str(y.shape) + '. Timestep-wise sample weighting (use of sample_weight_mode="temporal") is restricted to outputs that are at least 3D, i.e. that have a time dimension.')
        if sample_weight is not None and len(sample_weight.shape) != 2:
            raise ValueError('Found a sample_weight array with shape ' + str(sample_weight.shape) + '. In order to use timestep-wise sample weighting, you should pass a 2D sample_weight array.')
    elif sample_weight is not None and len(sample_weight.shape) != 1:
        raise ValueError('Found a sample_weight array with shape {}. In order to use timestep-wise sample weights, you should specify sample_weight_mode="temporal" in compile(); founssd "{}" instead. If you just mean to use sample-wise weights, make sure your sample_weight array is 1D.'.format(sample_weight.shape, sample_weight_mode))
    if sample_weight is not None:
        if len(sample_weight.shape) > len(y.shape):
            raise ValueError('Found a sample_weight with shape' + str(sample_weight.shape) + '.Expected sample_weight with rank less than or equal to ' + str(len(y.shape)))
        if not tensor_util.is_tf_type(sample_weight) and y.shape[:sample_weight.ndim] != sample_weight.shape:
            raise ValueError('Found a sample_weight array with shape ' + str(sample_weight.shape) + ' for an input with shape ' + str(y.shape) + '. sample_weight cannot be broadcast.')
    class_sample_weight = None
    if isinstance(class_weight, dict):
        if len(y.shape) > 2:
            raise ValueError('`class_weight` not supported for 3+ dimensional targets.')
        if tensor_util.is_tf_type(y):
            keys = np.array(sorted(class_weight.keys()))
            values = np.array([class_weight[i] for i in keys])
            weight_vector = np.zeros(np.max(keys) + 1)
            weight_vector[:] = np.nan
            weight_vector[keys] = values
            y_classes = smart_cond.smart_cond(len(y.shape.as_list()) == 2 and backend.shape(y)[1] > 1, lambda : backend.argmax(y, axis=1), lambda : math_ops.cast(backend.reshape(y, (-1,)), dtypes.int64))
            class_sample_weight = array_ops.gather(weight_vector, y_classes)
            gen_array_ops.check_numerics(class_sample_weight, 'Invalid classes or class weights detected. NaN values indicate that an appropriate class weight could not be determined.')
            class_sample_weight = math_ops.cast(class_sample_weight, backend.floatx())
            if sample_weight is not None:
                sample_weight = math_ops.cast(tensor_conversion.convert_to_tensor_v2_with_dispatch(sample_weight), backend.floatx())
        else:
            y_classes = y
            if len(y.shape) == 2:
                if y.shape[1] > 1:
                    y_classes = np.argmax(y, axis=1)
                elif y.shape[1] == 1:
                    y_classes = np.reshape(y, y.shape[0])
            class_sample_weight = np.asarray([class_weight[cls] for cls in y_classes if cls in class_weight])
            if len(class_sample_weight) != len(y_classes):
                existing_classes = set(y_classes)
                existing_class_weight = set(class_weight.keys())
                raise ValueError('`class_weight` must contain all classes in the data. The classes %s exist in the data but not in `class_weight`.' % (existing_classes - existing_class_weight))
    if class_sample_weight is not None and sample_weight is not None:
        return class_sample_weight * sample_weight
    if sample_weight is not None:
        return sample_weight
    if class_sample_weight is not None:
        return class_sample_weight
    return None

def has_symbolic_tensors(ls):
    if False:
        while True:
            i = 10
    if context.executing_eagerly():
        return False
    return has_tensors(ls)

def has_tensors(ls):
    if False:
        while True:
            i = 10
    'Returns true if `ls` contains tensors.'
    if isinstance(ls, (list, tuple)):
        return any((tensor_util.is_tf_type(v) and (not isinstance(v, ragged_tensor.RaggedTensor)) for v in ls))
    if isinstance(ls, dict):
        return any((tensor_util.is_tf_type(v) and (not isinstance(v, ragged_tensor.RaggedTensor)) for (_, v) in ls.items()))
    return tensor_util.is_tf_type(ls) and (not isinstance(ls, ragged_tensor.RaggedTensor))

def get_metric_name(metric, weighted=False):
    if False:
        print('Hello World!')
    'Returns the name corresponding to the given metric input.\n\n  Args:\n    metric: Metric function name or reference.\n    weighted: Boolean indicating if the given metric is weighted.\n\n  Returns:\n      The metric name.\n  '
    if tf2.enabled():
        if isinstance(metric, str):
            return metric
        metric = metrics_module.get(metric)
        return metric.name if hasattr(metric, 'name') else metric.__name__
    else:
        metric_name_prefix = 'weighted_' if weighted else ''
        if metric in ('accuracy', 'acc', 'crossentropy', 'ce'):
            if metric in ('accuracy', 'acc'):
                suffix = 'acc'
            elif metric in ('crossentropy', 'ce'):
                suffix = 'ce'
        else:
            metric_fn = metrics_module.get(metric)
            if hasattr(metric_fn, 'name'):
                suffix = metric_fn.name
            else:
                suffix = metric_fn.__name__
        metric_name = metric_name_prefix + suffix
        return metric_name

def get_metric_function(metric, output_shape=None, loss_fn=None):
    if False:
        while True:
            i = 10
    'Returns the metric function corresponding to the given metric input.\n\n  Args:\n      metric: Metric function name or reference.\n      output_shape: The shape of the output that this metric will be calculated\n        for.\n      loss_fn: The loss function used.\n\n  Returns:\n      The metric function.\n  '
    if metric not in ['accuracy', 'acc', 'crossentropy', 'ce']:
        return metrics_module.get(metric)
    is_sparse_categorical_crossentropy = isinstance(loss_fn, losses.SparseCategoricalCrossentropy) or (isinstance(loss_fn, losses.LossFunctionWrapper) and loss_fn.fn == losses.sparse_categorical_crossentropy)
    is_binary_crossentropy = isinstance(loss_fn, losses.BinaryCrossentropy) or (isinstance(loss_fn, losses.LossFunctionWrapper) and loss_fn.fn == losses.binary_crossentropy)
    if metric in ['accuracy', 'acc']:
        if output_shape[-1] == 1 or is_binary_crossentropy:
            return metrics_module.binary_accuracy
        elif is_sparse_categorical_crossentropy:
            return metrics_module.sparse_categorical_accuracy
        return metrics_module.categorical_accuracy
    else:
        if output_shape[-1] == 1 or is_binary_crossentropy:
            return metrics_module.binary_crossentropy
        elif is_sparse_categorical_crossentropy:
            return metrics_module.sparse_categorical_crossentropy
        return metrics_module.categorical_crossentropy

def call_metric_function(metric_fn, y_true, y_pred=None, weights=None, mask=None):
    if False:
        return 10
    'Invokes metric function and returns the metric result tensor.'
    if mask is not None:
        mask = math_ops.cast(mask, y_pred.dtype)
        if weights is None:
            weights = mask
        else:
            weights = math_ops.cast(weights, dtype=y_pred.dtype)
            (mask, _, weights) = losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=weights)
            weights *= mask
    if y_pred is not None:
        return metric_fn(y_true, y_pred, sample_weight=weights)
    return metric_fn(y_true, sample_weight=weights)

def get_loss_function(loss):
    if False:
        for i in range(10):
            print('nop')
    'Returns the loss corresponding to the loss input in `compile` API.'
    if loss is None or isinstance(loss, losses.Loss):
        return loss
    if tf_inspect.isclass(loss) and issubclass(loss, losses.Loss):
        raise ValueError('Received uninstantiated Loss class: {}\nPlease call loss ""classes before passing them to Model.compile.'.format(loss))
    if isinstance(loss, collections.abc.Mapping):
        loss = losses.get(loss)
    if callable(loss) and (not hasattr(loss, '__name__')):
        return loss
    loss_fn = losses.get(loss)
    return losses.LossFunctionWrapper(loss_fn, name=loss_fn.__name__, reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)

def validate_dataset_input(x, y, sample_weight, validation_split=None):
    if False:
        return 10
    'Validates user input arguments when a dataset iterator is passed.\n\n  Args:\n    x: Input data. A `tf.data` dataset or iterator.\n    y: Target data. It could be either Numpy array(s) or TensorFlow tensor(s).\n      Expected to be `None` when `x` is a dataset iterator.\n    sample_weight: An optional sample-weight array passed by the user to weight\n      the importance of each sample in `x`. Expected to be `None` when `x` is a\n      dataset iterator\n    validation_split: Float between 0 and 1. Fraction of the training data to be\n      used as validation data. Expected to be `None` when `x` is a dataset\n      iterator.\n\n  Raises:\n    ValueError: if argument `y` or `sample_weight` or `validation_split` are\n        provided by user.\n  '
    if y is not None:
        raise ValueError('You passed a dataset or dataset iterator (%s) as input `x` to your model. In that case, you should not specify a target (`y`) argument, since the dataset or dataset iterator generates both input data and target data. Received: %s' % (x, y))
    if sample_weight is not None:
        raise ValueError('`sample_weight` argument is not supported when input `x` is a dataset or a dataset iterator. Instead, youcan provide sample_weight as the third element  of yourdataset, i.e. (inputs, targets, sample_weight). Received: x=%s, sample_weight=%s' % (x, sample_weight))
    if validation_split is not None and validation_split != 0.0:
        raise ValueError('`validation_split` argument is not supported when input `x` is a dataset or a dataset iterator. Received: x=%s, validation_split=%f' % (x, validation_split))

def validate_input_types(inp, orig_inp, allow_dict=True, field_name='inputs'):
    if False:
        return 10
    'Helper function to validate either inputs or targets.'
    if isinstance(inp, (list, tuple)):
        if not all((isinstance(v, np.ndarray) or tensor_util.is_tf_type(v) for v in inp)):
            raise ValueError('Please provide as model inputs either a single array or a list of arrays. You passed: {}={}'.format(field_name, str(orig_inp)))
    elif isinstance(inp, dict):
        if not allow_dict:
            raise ValueError('You cannot pass a dictionary as model {}.'.format(field_name))
    elif not isinstance(inp, np.ndarray) and (not tensor_util.is_tf_type(inp)):
        raise ValueError('Please provide as model inputs either a single array or a list of arrays. You passed: {}={}'.format(field_name, orig_inp))

def check_generator_arguments(y=None, sample_weight=None, validation_split=None):
    if False:
        for i in range(10):
            print('nop')
    'Validates arguments passed when using a generator.'
    if y is not None:
        raise ValueError('`y` argument is not supported when data isa generator or Sequence instance. Instead pass targets as the second element of the generator.')
    if sample_weight is not None:
        raise ValueError('`sample_weight` argument is not supported when data isa generator or Sequence instance. Instead pass sample weights as the third element of the generator.')
    if validation_split:
        raise ValueError('If your data is in the form of a Python generator, you cannot use `validation_split`.')

def check_steps_argument(input_data, steps, steps_name):
    if False:
        while True:
            i = 10
    "Validates `steps` argument based on input data's type.\n\n  The cases when `steps` value must be provided are when\n    1. input data passed is an iterator.\n    2. model was built on top of symbolic tensors, input data is not\n       required and is `None`.\n    3. input data passed is a symbolic tensor.\n\n  Args:\n      input_data: Input data. Can be Numpy array(s) or TensorFlow tensor(s) or\n        tf.data.Dataset iterator or `None`.\n      steps: Integer or `None`. Total number of steps (batches of samples) to\n        execute.\n      steps_name: The public API's parameter name for `steps`.\n\n  Returns:\n    boolean, True if `steps` argument is required, else False.\n\n  Raises:\n      ValueError: if `steps` argument is required for given input data type\n        but not provided.\n  "
    is_x_iterator = isinstance(input_data, (iterator_ops.Iterator, iterator_ops.IteratorBase))
    if input_data is None or is_x_iterator or has_symbolic_tensors(input_data) or (isinstance(input_data, list) and (not input_data)):
        if steps is None:
            input_type_str = 'a Dataset iterator' if is_x_iterator else 'data tensors'
            raise ValueError('When using {input_type} as input to a model, you should specify the `{steps_name}` argument.'.format(input_type=input_type_str, steps_name=steps_name))
        return True
    if isinstance(input_data, (data_types.DatasetV1, data_types.DatasetV2)):
        return True
    if steps is not None:
        list_types = (np.ndarray, list, tuple)
        if isinstance(input_data, list_types) or (isinstance(input_data, dict) and any((isinstance(v, list_types) for v in input_data.values()))):
            logging.warning('When passing input data as arrays, do not specify `steps_per_epoch`/`steps` argument. Please use `batch_size` instead.')
    return False

def cast_single_tensor(x, dtype=None):
    if False:
        return 10
    if isinstance(x, np.ndarray):
        x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x)
    dtype = dtype or backend.floatx()
    if x.dtype.is_floating:
        return math_ops.cast(x, dtype=dtype)
    return x

def cast_if_floating_dtype_and_mismatch(targets, outputs):
    if False:
        i = 10
        return i + 15
    "Returns target data tensors using correct datatype.\n\n  Checks that each target and output pair are the same datatype. If not, casts\n  the target to the output's datatype.\n\n  Args:\n    targets: tensor or list of targets.\n    outputs: tensor or list of outputs.\n\n  Returns:\n    Targets in appropriate datatype.\n  "
    if tensor_util.is_tf_type(targets):
        return cast_single_tensor(targets, dtype=outputs[0].dtype)
    new_targets = []
    for (target, out) in zip(targets, outputs):
        if isinstance(target, np.ndarray):
            target = tensor_conversion.convert_to_tensor_v2_with_dispatch(target)
        if target.dtype != out.dtype:
            new_targets.append(cast_single_tensor(target, dtype=out.dtype))
        else:
            new_targets.append(target)
    return new_targets

def cast_if_floating_dtype(x, dtype=None):
    if False:
        while True:
            i = 10
    'Casts the given data tensors to the default floating point type.\n\n  Casts only if the input is already a floating point type.\n  Args:\n    x: tensor or list/tuple of tensors.\n    dtype: The dtype to which Tensors should be cast.\n\n  Returns:\n    Converted input.\n  '
    return nest.map_structure(functools.partial(cast_single_tensor, dtype=dtype), x)

def cast_to_model_input_dtypes(x, model):
    if False:
        i = 10
        return i + 15
    'Casts the given data tensors to the dtypes of the model inputs.\n\n  Args:\n    x: tensor or list/tuple of tensors.\n    model: The model.\n\n  Returns:\n    Converted input. Each tensor is casted to the corresponding input in\n    `model.inputs`.\n  '
    input_dtypes = nest.map_structure(lambda t: t.dtype, model.inputs)
    return nest.map_structure(math_ops.cast, x, input_dtypes)

def prepare_sample_weight_modes(training_endpoints, sample_weight_mode):
    if False:
        return 10
    'Prepares sample weight modes for the model.\n\n  Args:\n    training_endpoints: List of model _TrainingEndpoints.\n    sample_weight_mode: sample weight mode user input passed from compile API.\n\n  Raises:\n    ValueError: In case of invalid `sample_weight_mode` input.\n  '
    if isinstance(sample_weight_mode, collections.abc.Mapping):
        generic_utils.check_for_unexpected_keys('sample_weight_mode', sample_weight_mode, [e.output_name for e in training_endpoints])
        for end_point in training_endpoints:
            if not end_point.should_skip_target_weights():
                if end_point.output_name not in sample_weight_mode:
                    raise ValueError('Output ' + end_point.output_name + 'missing from `_sample_weight_modes` dictionary')
                else:
                    end_point.sample_weight_mode = sample_weight_mode.get(end_point.output_name)
    elif isinstance(sample_weight_mode, (list, tuple)):
        if len(sample_weight_mode) != len(training_endpoints):
            raise ValueError('When passing a list as sample_weight_mode, it should have one entry per model output. The model has ' + str(len(training_endpoints)) + ' outputs, but you passed ' + str(len(sample_weight_mode)) + '_sample_weight_modes.')
        for (mode, endpoint) in zip(sample_weight_mode, training_endpoints):
            if not endpoint.should_skip_target_weights():
                endpoint.sample_weight_mode = mode
    else:
        for endpoint in training_endpoints:
            if not endpoint.should_skip_target_weights():
                endpoint.sample_weight_mode = sample_weight_mode

def prepare_loss_functions(loss, output_names):
    if False:
        return 10
    'Converts loss to a list of loss functions.\n\n  Args:\n      loss: String (name of objective function), objective function or\n        `tf.losses.Loss` instance. See `tf.losses`. If the model has multiple\n        outputs, you can use a different loss on each output by passing a\n        dictionary or a list of losses. The loss value that will be minimized by\n        the model will then be the sum of all individual losses.\n      output_names: List of model output names.\n\n  Returns:\n      A list of loss objective functions.\n\n  Raises:\n      ValueError: If loss is a dict with keys not in model output names,\n          or if loss is a list with len not equal to model outputs.\n  '
    if isinstance(loss, collections.abc.Mapping):
        generic_utils.check_for_unexpected_keys('loss', loss, output_names)
        loss_functions = []
        for name in output_names:
            if name not in loss:
                logging.warning('Output {0} missing from loss dictionary. We assume this was done on purpose. The fit and evaluate APIs will not be expecting any data to be passed to {0}.'.format(name))
            loss_functions.append(get_loss_function(loss.get(name, None)))
    elif isinstance(loss, str):
        loss_functions = [get_loss_function(loss) for _ in output_names]
    elif isinstance(loss, collections.abc.Sequence):
        if len(loss) != len(output_names):
            raise ValueError('When passing a list as loss, it should have one entry per model outputs. The model has {} outputs, but you passed loss={}'.format(len(output_names), loss))
        loss_functions = nest.map_structure(get_loss_function, loss)
    else:
        loss_functions = [get_loss_function(loss) for _ in range(len(output_names))]
    return loss_functions

def prepare_loss_weights(training_endpoints, loss_weights=None):
    if False:
        return 10
    "Converts loss weights to a list of loss weights.\n\n  The result loss weights will be populated on the training endpoint.\n\n  Args:\n      training_endpoints: List of model training endpoints.\n      loss_weights: Optional list or dictionary specifying scalar coefficients\n        (Python floats) to weight the loss contributions of different model\n        outputs. The loss value that will be minimized by the model will then be\n        the *weighted sum* of all individual losses, weighted by the\n          `loss_weights` coefficients. If a list, it is expected to have a 1:1\n            mapping to the model's outputs. If a dict, it is expected to map\n            output names (strings) to scalar coefficients.\n\n  Raises:\n      ValueError: If loss weight is a dict with key not in model output names,\n          or if loss is a list with len not equal to model outputs.\n  "
    if loss_weights is None:
        for e in training_endpoints:
            e.loss_weight = 1.0
    elif isinstance(loss_weights, collections.abc.Mapping):
        generic_utils.check_for_unexpected_keys('loss_weights', loss_weights, [e.output_name for e in training_endpoints])
        for e in training_endpoints:
            e.loss_weight = loss_weights.get(e.output_name, 1.0)
    elif isinstance(loss_weights, list):
        if len(loss_weights) != len(training_endpoints):
            raise ValueError('When passing a list as loss_weights, it should have one entry per model output. The model has ' + str(len(training_endpoints)) + ' outputs, but you passed loss_weights=' + str(loss_weights))
        for (w, e) in zip(loss_weights, training_endpoints):
            e.loss_weight = w
    else:
        raise TypeError('Could not interpret loss_weights argument: ' + str(loss_weights) + ' - expected a list of dicts.')

def is_feature_layer(layer):
    if False:
        i = 10
        return i + 15
    'Returns whether `layer` is a FeatureLayer or not.'
    return getattr(layer, '_is_feature_layer', False)

def is_eager_dataset_or_iterator(data):
    if False:
        return 10
    return context.executing_eagerly() and isinstance(data, (data_types.DatasetV1, data_types.DatasetV2, iterator_ops.IteratorBase))

def get_dataset_graph_def(dataset):
    if False:
        while True:
            i = 10
    if context.executing_eagerly():
        graph_def_str = dataset._as_serialized_graph().numpy()
    else:
        graph_def_str = backend.get_value(dataset._as_serialized_graph())
    return graph_pb2.GraphDef().FromString(graph_def_str)

def verify_dataset_shuffled(x):
    if False:
        while True:
            i = 10
    'Verifies that the dataset is shuffled.\n\n  Args:\n    x: Dataset passed as an input to the model.\n\n  Returns:\n    boolean, whether the input dataset is shuffled or not.\n  '
    assert isinstance(x, data_types.DatasetV2)
    graph_def = get_dataset_graph_def(x)
    for node in graph_def.node:
        if node.op.startswith('ShuffleDataset'):
            return True
    for function in graph_def.library.function:
        for node in function.node_def:
            if node.op.startswith('ShuffleDataset'):
                return True
    logging.warning('Expected a shuffled dataset but input dataset `x` is not shuffled. Please invoke `shuffle()` on input dataset.')
    return False

def is_dataset_or_iterator(data):
    if False:
        i = 10
        return i + 15
    return isinstance(data, (data_types.DatasetV1, data_types.DatasetV2, iterator_ops.Iterator, iterator_ops.IteratorBase))

def get_iterator(dataset):
    if False:
        print('Hello World!')
    'Create and initialize an iterator from a dataset.'
    if context.executing_eagerly():
        iterator = dataset_ops.make_one_shot_iterator(dataset)
    else:
        iterator = dataset_ops.make_initializable_iterator(dataset)
    initialize_iterator(iterator)
    return iterator

def initialize_iterator(iterator):
    if False:
        i = 10
        return i + 15
    if not context.executing_eagerly():
        init_op = iterator.initializer
        backend.get_session((init_op,)).run(init_op)

def extract_tensors_from_dataset(dataset):
    if False:
        i = 10
        return i + 15
    'Extract a tuple of tensors `inputs, targets, sample_weight` from a dataset.\n\n  Args:\n    dataset: Dataset instance.\n\n  Returns:\n    Tuple of tensors `x, y, weights`. `y` and `weights` entry may be None.\n  '
    iterator = get_iterator(dataset)
    (inputs, targets, sample_weight) = unpack_iterator_input(iterator)
    return (inputs, targets, sample_weight)

def unpack_iterator_input(iterator):
    if False:
        for i in range(10):
            print('nop')
    'Convert a dataset iterator to a tuple of tensors `x, y, sample_weights`.\n\n  Args:\n    iterator: Instance of a dataset iterator.\n\n  Returns:\n    Tuple of tensors `x, y, weights`. `y` and `weights` entry may be None.\n  '
    try:
        next_element = iterator.get_next()
    except errors.OutOfRangeError:
        raise RuntimeError('Your dataset iterator ran out of data; Make sure that your dataset can generate required number of samples.')
    if isinstance(next_element, (list, tuple)):
        if len(next_element) not in [2, 3]:
            raise ValueError('Please provide model inputs as a list or tuple of 2 or 3 elements: (input, target) or (input, target, sample_weights) Received %s' % next_element)
        if len(next_element) == 2:
            (x, y) = next_element
            weights = None
        else:
            (x, y, weights) = next_element
    else:
        x = next_element
        y = None
        weights = None
    return (x, y, weights)

def infer_steps_for_dataset(model, dataset, steps, epochs=1, steps_name='steps'):
    if False:
        i = 10
        return i + 15
    'Infers steps_per_epoch needed to loop through a dataset.\n\n  Args:\n      model: Keras model instance.\n      dataset: Input data of type tf.data.Dataset.\n      steps: Number of steps to draw from the dataset (may be None if unknown).\n      epochs: Number of times to iterate over the dataset.\n      steps_name: The string name of the steps argument, either `steps`,\n        `validation_steps`, or `steps_per_epoch`. Only used for error message\n        formatting.\n\n  Returns:\n    Integer or `None`. Inferred number of steps to loop through the dataset.\n    `None` is returned if 1) the size of the dataset is unknown and `steps` was\n    not specified, or 2) this is multi-worker training and auto sharding is\n    enabled.\n\n  Raises:\n    ValueError: In case of invalid argument values.\n  '
    assert isinstance(dataset, data_types.DatasetV2)
    if model._in_multi_worker_mode() and dataset.options().experimental_distribute.auto_shard_policy != options_lib.AutoShardPolicy.OFF:
        return None
    size = backend.get_value(cardinality.cardinality(dataset))
    if size == cardinality.INFINITE and steps is None:
        raise ValueError('When passing an infinitely repeating dataset, you must specify the `%s` argument.' % (steps_name,))
    if size >= 0:
        if steps is not None and steps * epochs > size:
            if epochs > 1:
                raise ValueError('The dataset you passed contains %s batches, but you passed `epochs=%s` and `%s=%s`, which is a total of %s steps. We cannot draw that many steps from this dataset. We suggest to set `%s=%s`.' % (size, epochs, steps_name, steps, steps * epochs, steps_name, size // epochs))
            else:
                raise ValueError('The dataset you passed contains %s batches, but you passed `%s=%s`. We cannot draw that many steps from this dataset. We suggest to set `%s=%s`.' % (size, steps_name, steps, steps_name, size))
    if steps is None:
        if size >= 0:
            return size
        return None
    return steps

class ModelInputs(object):
    """Encapsulates model inputs.

  Allows for transforming model inputs while keeping the same structure.
  """

    def __init__(self, inputs):
        if False:
            print('Hello World!')
        self._inputs = inputs
        self._is_dict = isinstance(self._inputs, dict)
        self._is_single_input = not isinstance(self._inputs, (list, tuple, dict))
        self._flattened_inputs = []
        self._input_names = []
        if self._is_dict:
            for k in sorted(self._inputs.keys()):
                self._flattened_inputs.append(self._inputs[k])
                self._input_names.append(k)
        else:
            self._flattened_inputs = nest.flatten(self._inputs)
            self._input_names = ['input_%d' % (i + 1) for i in range(len(self._flattened_inputs))]

    def get_input_names(self):
        if False:
            while True:
                i = 10
        "Returns keys to name inputs by.\n\n    In case inputs provided were a list, tuple or single entry, we make up a\n    key 'input_%d'. For dictionary case, we return a sorted list of keys.\n    "
        return self._input_names

    def get_symbolic_inputs(self, return_single_as_list=False):
        if False:
            i = 10
            return i + 15
        'Returns inputs to be set as self.inputs for a model.'
        for (i, (k, v)) in enumerate(zip(self._input_names, self._flattened_inputs)):
            if isinstance(v, (list, float, int)):
                v = np.asarray(v)
                if v.ndim == 1:
                    v = np.expand_dims(v, 1)
            if isinstance(v, np.ndarray):
                shape = (None,) + tuple(v.shape[1:])
                if shape == (None,):
                    shape = (None, 1)
                dtype = dtypes.as_dtype(v.dtype)
                if dtype.is_floating:
                    dtype = backend.floatx()
                v = backend.placeholder(shape=shape, name=k, dtype=dtype)
            elif isinstance(v, tensor_spec.TensorSpec):
                shape = (None,) + tuple(v.shape.as_list()[1:])
                if shape == (None,):
                    shape = (None, 1)
                v = backend.placeholder(shape=shape, name=k, dtype=v.dtype)
            self._flattened_inputs[i] = v
        if self._is_dict:
            return dict(zip(self._input_names, self._flattened_inputs))
        if self._is_single_input and (not return_single_as_list):
            return self._flattened_inputs[0]
        return self._flattened_inputs

    def as_dict(self):
        if False:
            i = 10
            return i + 15
        'An iterable over a dictionary version of inputs.'
        for (k, v) in zip(self._input_names, self._flattened_inputs):
            yield (k, v)

    def as_list(self):
        if False:
            print('Hello World!')
        'Returning the inputs as a list.'
        return self._flattened_inputs

def generic_output_names(outputs_list):
    if False:
        i = 10
        return i + 15
    return ['output_%d' % (i + 1) for i in range(len(outputs_list))]

def should_run_validation(validation_freq, epoch):
    if False:
        print('Hello World!')
    'Checks if validation should be run this epoch.\n\n  Args:\n    validation_freq: Integer or list. If an integer, specifies how many training\n      epochs to run before a new validation run is performed. If a list,\n      specifies the epochs on which to run validation.\n    epoch: Integer, the number of the training epoch just completed.\n\n  Returns:\n    Bool, True if validation should be run.\n\n  Raises:\n    ValueError: if `validation_freq` is an Integer and less than 1, or if\n    it is neither an Integer nor a Sequence.\n  '
    one_indexed_epoch = epoch + 1
    if isinstance(validation_freq, int):
        if validation_freq < 1:
            raise ValueError('`validation_freq` can not be less than 1.')
        return one_indexed_epoch % validation_freq == 0
    if not isinstance(validation_freq, collections.abc.Container):
        raise ValueError('`validation_freq` must be an Integer or `collections.abc.Container` (e.g. list, tuple, etc.)')
    return one_indexed_epoch in validation_freq

def split_training_and_validation_data(x, y, sample_weights, validation_split):
    if False:
        print('Hello World!')
    'Split input data into train/eval section based on validation_split.'
    if has_symbolic_tensors(x):
        raise ValueError('If your data is in the form of symbolic tensors, you cannot use `validation_split`.')
    if hasattr(x[0], 'shape'):
        split_at = int(x[0].shape[0] * (1.0 - validation_split))
    else:
        split_at = int(len(x[0]) * (1.0 - validation_split))
    (x, val_x) = (generic_utils.slice_arrays(x, 0, split_at), generic_utils.slice_arrays(x, split_at))
    (y, val_y) = (generic_utils.slice_arrays(y, 0, split_at), generic_utils.slice_arrays(y, split_at))
    if sample_weights:
        (sample_weights, val_sample_weights) = (generic_utils.slice_arrays(sample_weights, 0, split_at), generic_utils.slice_arrays(sample_weights, split_at))
    else:
        val_sample_weights = None
    return (x, y, sample_weights, val_x, val_y, val_sample_weights)

def unpack_validation_data(validation_data, raise_if_ambiguous=True):
    if False:
        print('Hello World!')
    'Unpack validation data based input type.\n\n  The validation data is not touched if its dataset or dataset iterator.\n  For other type of input (Numpy or tensor), it will be unpacked into tuple of\n  3 which is x, y and sample weights.\n\n  Args:\n    validation_data: dataset, dataset iterator, or numpy, tensor tuple.\n    raise_if_ambiguous: boolean on whether to fail if validation_data cannot be\n      parsed. Otherwise simply return validation_data, None, None and defer the\n      decision to the caller.\n\n  Returns:\n    tuple of 3, (x, y, sample_weights) for numpy and tensor input.\n  '
    if isinstance(validation_data, (iterator_ops.Iterator, iterator_ops.IteratorBase, data_types.DatasetV2, data_utils.Sequence)) or not hasattr(validation_data, '__len__'):
        val_x = validation_data
        val_y = None
        val_sample_weight = None
    elif len(validation_data) == 2:
        try:
            (val_x, val_y) = validation_data
            val_sample_weight = None
        except ValueError:
            (val_x, val_y, val_sample_weight) = (validation_data, None, None)
    elif len(validation_data) == 3:
        try:
            (val_x, val_y, val_sample_weight) = validation_data
        except ValueError:
            (val_x, val_y, val_sample_weight) = (validation_data, None, None)
    else:
        if raise_if_ambiguous:
            raise ValueError('When passing a `validation_data` argument, it must contain either 2 items (x_val, y_val), or 3 items (x_val, y_val, val_sample_weights), or alternatively it could be a dataset or a dataset or a dataset iterator. However we received `validation_data=%s`' % validation_data)
        (val_x, val_y, val_sample_weight) = (validation_data, None, None)
    return (val_x, val_y, val_sample_weight)

class TrainingLoop(object):
    """TrainingLoop is a wrapper class around the training logic.

  This class is trying to encapsulate the different logic of fit/eval/predict
  with regard to different data input and model condition.

  Note that TrainingLoop is stateless, which means it doesn't contain any
  internal field and can be reused with different model and inputs.
  """

    def fit(self, model, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, **kwargs):
        if False:
            while True:
                i = 10
        'Train the model with the inputs and targets.'
        raise NotImplementedError()

    def evaluate(self, model, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Returns the loss value & metrics values for the model in test mode.'
        raise NotImplementedError()

    def predict(self, model, x, batch_size=None, verbose=0, steps=None, callbacks=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()