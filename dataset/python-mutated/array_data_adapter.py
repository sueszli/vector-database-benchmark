import math
import numpy as np
import tree
from keras import backend
from keras.trainers.data_adapters import data_adapter_utils
from keras.trainers.data_adapters.data_adapter import DataAdapter
from keras.utils.nest import lists_to_tuples
try:
    import pandas
except ImportError:
    pandas = None

class ArrayDataAdapter(DataAdapter):
    """Adapter for array-like objects, e.g. TF/JAX Tensors, NumPy arrays."""

    def __init__(self, x, y=None, sample_weight=None, batch_size=None, steps=None, shuffle=False, class_weight=None):
        if False:
            i = 10
            return i + 15
        if not can_convert_arrays((x, y, sample_weight)):
            raise ValueError(f'Expected all elements of `x` to be array-like. Received invalid types: x={x}')
        (x, y, sample_weight) = convert_to_arrays((x, y, sample_weight))
        if sample_weight is not None:
            if class_weight is not None:
                raise ValueError('You cannot `class_weight` and `sample_weight` at the same time.')
            if tree.is_nested(y):
                if isinstance(sample_weight, np.ndarray):
                    is_samplewise = len(sample_weight.shape) == 1 or (len(sample_weight.shape) == 2 and sample_weight.shape[1] == 1)
                    if not is_samplewise:
                        raise ValueError('For a model with multiple outputs, when providing a single `sample_weight` array, it should only have one scalar score per sample (i.e. shape `(num_samples,)`). If you want to use non-scalar sample weights, pass a `sample_weight` argument with one array per model output.')
                    sample_weight = tree.map_structure(lambda _: sample_weight, y)
                else:
                    try:
                        tree.assert_same_structure(y, sample_weight)
                    except ValueError:
                        raise ValueError(f'You should provide one `sample_weight` array per output in `y`. The two structures did not match:\n- y: {y}\n- sample_weight: {sample_weight}\n')
        if class_weight is not None:
            if tree.is_nested(y):
                raise ValueError('`class_weight` is only supported for Models with a single output.')
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(y, class_weight)
        inputs = data_adapter_utils.pack_x_y_sample_weight(x, y, sample_weight)
        data_adapter_utils.check_data_cardinality(inputs)
        num_samples = set((i.shape[0] for i in tree.flatten(inputs))).pop()
        self._num_samples = num_samples
        self._inputs = inputs
        if not batch_size:
            batch_size = int(math.ceil(num_samples / steps)) if steps else 32
        self._size = int(math.ceil(num_samples / batch_size))
        self._batch_size = batch_size
        self._partial_batch_size = num_samples % batch_size
        self._shuffle = shuffle

    def get_numpy_iterator(self):
        if False:
            print('Hello World!')
        inputs = self._inputs
        if self._shuffle:
            inputs = data_adapter_utils.sync_shuffle(inputs, num_samples=self._num_samples)
        for i in range(self._size):
            (start, stop) = (i * self._batch_size, (i + 1) * self._batch_size)
            yield tree.map_structure(lambda x: x[start:stop], inputs)

    def get_tf_dataset(self):
        if False:
            i = 10
            return i + 15
        from keras.utils.module_utils import tensorflow as tf
        inputs = self._inputs
        shuffle = self._shuffle
        batch_size = self._batch_size
        num_samples = self._num_samples
        num_full_batches = int(self._num_samples // batch_size)
        indices_dataset = tf.data.Dataset.range(1)

        def permutation(_):
            if False:
                i = 10
                return i + 15
            indices = tf.range(num_samples, dtype=tf.int64)
            if shuffle and shuffle != 'batch':
                indices = tf.random.shuffle(indices)
            return indices
        indices_dataset = indices_dataset.map(permutation).prefetch(1)

        def slice_batch_indices(indices):
            if False:
                print('Hello World!')
            'Convert a Tensor of indices into a dataset of batched indices.\n\n            This step can be accomplished in several ways. The most natural is\n            to slice the Tensor in a Dataset map. (With a condition on the upper\n            index to handle the partial batch.) However it turns out that\n            coercing the Tensor into a shape which is divisible by the batch\n            size (and handling the last partial batch separately) allows for a\n            much more favorable memory access pattern and improved performance.\n\n            Args:\n                indices: Tensor which determines the data order for an entire\n                    epoch.\n\n            Returns:\n                A Dataset of batched indices.\n            '
            num_in_full_batch = num_full_batches * batch_size
            first_k_indices = tf.slice(indices, [0], [num_in_full_batch])
            first_k_indices = tf.reshape(first_k_indices, [num_full_batches, batch_size])
            flat_dataset = tf.data.Dataset.from_tensor_slices(first_k_indices)
            if self._partial_batch_size:
                index_remainder = tf.data.Dataset.from_tensors(tf.slice(indices, [num_in_full_batch], [self._partial_batch_size]))
                flat_dataset = flat_dataset.concatenate(index_remainder)
            return flat_dataset

        def slice_inputs(indices_dataset, inputs):
            if False:
                for i in range(10):
                    print('nop')
            'Slice inputs into a Dataset of batches.\n\n            Given a Dataset of batch indices and the unsliced inputs,\n            this step slices the inputs in a parallelized fashion\n            and produces a dataset of input batches.\n\n            Args:\n                indices_dataset: A Dataset of batched indices.\n                inputs: A python data structure that contains the inputs,\n                    targets, and possibly sample weights.\n\n            Returns:\n                A Dataset of input batches matching the batch indices.\n            '
            dataset = tf.data.Dataset.zip((indices_dataset, tf.data.Dataset.from_tensors(inputs).repeat()))

            def grab_batch(i, data):
                if False:
                    for i in range(10):
                        print('nop')
                return tree.map_structure(lambda d: tf.gather(d, i, axis=0), data)
            dataset = dataset.map(grab_batch, num_parallel_calls=tf.data.AUTOTUNE)
            options = tf.data.Options()
            options.experimental_optimization.apply_default_optimizations = False
            if self._shuffle:
                options.experimental_external_state_policy = tf.data.experimental.ExternalStatePolicy.IGNORE
            dataset = dataset.with_options(options)
            return dataset
        indices_dataset = indices_dataset.flat_map(slice_batch_indices)
        dataset = slice_inputs(indices_dataset, inputs)
        if shuffle == 'batch':

            def shuffle_batch(*batch):
                if False:
                    while True:
                        i = 10
                return tree.map_structure(tf.random.shuffle, batch)
            dataset = dataset.map(shuffle_batch)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        return dataset.prefetch(tf.data.AUTOTUNE)

    @property
    def num_batches(self):
        if False:
            i = 10
            return i + 15
        return self._size

    @property
    def batch_size(self):
        if False:
            i = 10
            return i + 15
        return self._batch_size

    @property
    def has_partial_batch(self):
        if False:
            while True:
                i = 10
        return self._partial_batch_size > 0

    @property
    def partial_batch_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._partial_batch_size or None

def can_convert_arrays(arrays):
    if False:
        return 10
    'Check if array like-inputs can be handled by `ArrayDataAdapter`\n\n    Args:\n        inputs: Structure of `Tensor`s, NumPy arrays, or tensor-like.\n\n    Returns:\n        `True` if `arrays` can be handled by `ArrayDataAdapter`, `False`\n        otherwise.\n    '

    def can_convert_single_array(x):
        if False:
            for i in range(10):
                print('nop')
        is_none = x is None
        known_type = isinstance(x, data_adapter_utils.ARRAY_TYPES)
        convertable_type = hasattr(x, '__array__')
        return is_none or known_type or convertable_type
    return all(tree.flatten(tree.map_structure(can_convert_single_array, arrays)))

def convert_to_arrays(arrays):
    if False:
        while True:
            i = 10
    'Process array-like inputs.\n\n    This function:\n\n    - Converts tf.Tensors to NumPy arrays.\n    - Converts `pandas.Series` to `np.ndarray`\n    - Converts `list`s to `tuple`s (for `tf.data` support).\n\n    Args:\n        inputs: Structure of `Tensor`s, NumPy arrays, or tensor-like.\n\n    Returns:\n        Structure of NumPy `ndarray`s.\n    '

    def convert_single_array(x):
        if False:
            for i in range(10):
                print('nop')
        if x is None:
            return x
        if pandas is not None:
            if isinstance(x, pandas.Series):
                x = np.expand_dims(x.to_numpy(), axis=-1)
            elif isinstance(x, pandas.DataFrame):
                x = x.to_numpy()
        if is_tf_ragged_tensor(x):
            from keras.utils.module_utils import tensorflow as tf
            if backend.is_float_dtype(x.dtype) and (not backend.standardize_dtype(x.dtype) == backend.floatx()):
                x = tf.cast(x, backend.floatx())
            return x
        if not isinstance(x, np.ndarray):
            if hasattr(x, '__array__'):
                x = backend.convert_to_numpy(x)
            else:
                raise ValueError(f'Expected a NumPy array, tf.Tensor, tf.RaggedTensor, jax.np.ndarray, torch.Tensor, Pandas Dataframe, or Pandas Series. Received invalid input: {x} (of type {type(x)})')
        if x.dtype == object:
            return x
        if backend.is_float_dtype(x.dtype) and (not backend.standardize_dtype(x.dtype) == backend.floatx()):
            x = x.astype(backend.floatx())
        return x
    arrays = tree.map_structure(convert_single_array, arrays)
    return lists_to_tuples(arrays)

def is_tf_ragged_tensor(x):
    if False:
        for i in range(10):
            print('nop')
    return x.__class__.__name__ == 'RaggedTensor'