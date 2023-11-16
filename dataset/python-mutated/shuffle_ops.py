"""Experimental shuffle ops."""
import functools
import numpy as np
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

class _ShuffleAndRepeatDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that fuses `shuffle` and `repeat`."""

    def __init__(self, input_dataset, buffer_size, count=None, seed=None):
        if False:
            while True:
                i = 10
        self._input_dataset = input_dataset
        self._buffer_size = ops.convert_to_tensor(buffer_size, dtype=dtypes.int64, name='buffer_size')
        if count is None:
            self._count = constant_op.constant(-1, dtype=dtypes.int64, name='count')
        else:
            self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name='count')
        (self._seed, self._seed2) = random_seed.get_seed(seed)
        variant_tensor = gen_dataset_ops.shuffle_and_repeat_dataset(self._input_dataset._variant_tensor, buffer_size=self._buffer_size, count=self._count, seed=self._seed, seed2=self._seed2, **self._flat_structure)
        super(_ShuffleAndRepeatDataset, self).__init__(input_dataset, variant_tensor)

@deprecation.deprecated(None, 'Use `tf.data.Dataset.shuffle(buffer_size, seed)` followed by `tf.data.Dataset.repeat(count)`. Static tf.data optimizations will take care of using the fused implementation.')
@tf_export('data.experimental.shuffle_and_repeat')
def shuffle_and_repeat(buffer_size, count=None, seed=None):
    if False:
        while True:
            i = 10
    'Shuffles and repeats a Dataset, reshuffling with each repetition.\n\n  >>> d = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n  >>> d = d.apply(tf.data.experimental.shuffle_and_repeat(2, count=2))\n  >>> [elem.numpy() for elem in d] # doctest: +SKIP\n  [2, 3, 1, 1, 3, 2]\n\n  ```python\n  dataset.apply(\n    tf.data.experimental.shuffle_and_repeat(buffer_size, count, seed))\n  ```\n\n  produces the same output as\n\n  ```python\n  dataset.shuffle(\n    buffer_size, seed=seed, reshuffle_each_iteration=True).repeat(count)\n  ```\n\n  In each repetition, this dataset fills a buffer with `buffer_size` elements,\n  then randomly samples elements from this buffer, replacing the selected\n  elements with new elements. For perfect shuffling, set the buffer size equal\n  to the full size of the dataset.\n\n  For instance, if your dataset contains 10,000 elements but `buffer_size` is\n  set to 1,000, then `shuffle` will initially select a random element from\n  only the first 1,000 elements in the buffer. Once an element is selected,\n  its space in the buffer is replaced by the next (i.e. 1,001-st) element,\n  maintaining the 1,000 element buffer.\n\n  Args:\n    buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum\n      number elements that will be buffered when prefetching.\n    count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the number\n      of times the dataset should be repeated. The default behavior (if `count`\n      is `None` or `-1`) is for the dataset be repeated indefinitely.\n    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random\n      seed that will be used to create the distribution. See\n      `tf.random.set_seed` for behavior.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  '

    def _apply_fn(dataset):
        if False:
            print('Hello World!')
        return _ShuffleAndRepeatDataset(dataset, buffer_size, count, seed)
    return _apply_fn

def _process_file_infos(file_infos):
    if False:
        while True:
            i = 10
    'Computes aggregate information about files to read.\n\n  The method collects information about the files to read, the total number of\n  elements, and arrays that can be used to account for elements to be skipped,\n  which can be specified via the "skip" and "take" keys.\n\n  To account for elements to skip, the range of each file can be divided into\n  three regions:\n  - S (elements to skip)\n  - T (elements to read)\n  - R (remainder of elements that will also be skipped)\n\n  The `thresholds` and `offsets` arrays are initialized as follows:\n  `thresholds = [0, T_1, T_1 + T_2, ...]` and\n  `offsets = [S_1, S_1 + R_1 + S_2, S_1 + R_1 + S_2 + R_2 + S_3, ...]`\n\n  This makes it possible to map an index from a contiguous range\n  `(0...num_elements_to_read)` to an index in the range of all elements,\n  skipping over elements as per the "skip" and "take" keys values. In\n  particular, for a given input index `X`, we find the greatest `thresholds`\n  value that is smaller or equal to `X`. Let `t(X)` denotes such index in the\n  `thresholds` array. The output index is computed as `X + offsets[t(X)]`.\n\n  Args:\n    file_infos: See `file_infos` argument of `index_shuffle` for details.\n\n  Returns:\n    A dictionary containing the following keys:\n      - `files`, the vector of pathnames of files to read\n      - `num_elements`, an integer identifying the total number of elements\n      - `offsets`, the vector of offsets to use for index adjustment (in case\n        any elements should be skipped)\n      - `thresholds`, the vector of thresholds to use for index adjustment (in\n        case any elements should be skipped)\n  '
    files = []
    num_elements = 0
    offsets = np.int64([])
    offset_sum = 0
    thresholds = np.int64([])
    threshold_sum = 0
    adjustment_needed = False
    for file_info in file_infos:
        files.append(file_info['path'])
        skip = 0
        if 'skip' in file_info:
            if file_info['skip'] < -1:
                raise ValueError('`skip` should be greater than `-1` but got {}'.format(file_info['skip']))
            if file_info['skip'] == -1:
                skip = file_info['num_elements']
            else:
                skip = min(file_info['skip'], file_info['num_elements'])
        take = file_info['num_elements'] - skip
        if 'take' in file_info:
            if file_info['take'] < -1:
                raise ValueError('`take` should be greater than `-1` but got {}'.format(file_info['take']))
            if file_info['take'] != -1:
                take = min(file_info['take'], take)
        remainder = file_info['num_elements'] - skip - take
        if take != file_info['num_elements']:
            adjustment_needed = True
        num_elements += take
        offsets = np.append(offsets, offset_sum + skip)
        offset_sum += skip + remainder
        thresholds = np.append(thresholds, threshold_sum)
        threshold_sum += take
    result = {'files': files, 'num_elements': num_elements}
    if adjustment_needed:
        result['offsets'] = offsets
        result['thresholds'] = thresholds
    return result

def _adjust_index(index, thresholds, offsets):
    if False:
        return 10
    'Adjusts index to account for elements to be skipped.'
    t_index = array_ops.shape(array_ops.boolean_mask(thresholds, math_ops.less_equal(thresholds, index)))[0] - 1
    return index + array_ops.gather(offsets, t_index)

def index_shuffle(file_infos, reader_factory, seed=None, reshuffle_each_iteration=False, num_parallel_calls=dataset_ops.AUTOTUNE):
    if False:
        return 10
    'Creates a (globally) shuffled dataset from the given set of files.\n\n  Unlike `tf.data.Dataset.shuffle()`, which uses an in-memory buffer to shuffle\n  elements of input dataset in a streaming fashion,\n  `tf.data.experimental.index_shuffle()` performs a global shuffle of element\n  indices and then reads the data in a shuffled order. The advantage of\n  `index_shuffle()` is that it can perform global shuffle of datasets that do\n  not fit into memory (as long as the array of their indices does) and that the\n  shuffling logic it provides is compatible with symbolic checkpointing. The\n  disadvantage of `index_shuffle()` is that reading data in a shuffled random\n  order will in general not be as efficient as reading data sequentially.\n\n  Args:\n    file_infos: A list of dictionaries that describe each file of the input\n      dataset. Each dictionary is expected to contain the "path" key, which\n      identifies the path of the file and the "num_elements" key, which\n      identifies the number of elements in the file. In addition, the "skip"\n      and "take" keys can be used to identify the number of elements to skip\n      and take respectively. By default, no elements are skipped and all\n      elements are taken.\n    reader_factory: A function that maps a sequence of filenames to an instance\n      of `tf.data.Dataset` that reads data from the files.\n    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random\n      seed that will be used to shuffle the order of elements. Default to\n      non-deterministic seed.\n    reshuffle_each_iteration: (Optional.) A `tf.bool` scalar `tf.Tensor`, that\n      determines whether to change the shuffle order each iteration. Defaults to\n      `False`.\n    num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`, that\n      determines the maximum number of random access operations to perform\n      in parallel. By default, the tf.data runtime uses autotuning to determine\n      the value dynamically.\n\n  Returns:\n    A `tf.data.Dataset` object, representing a globally shuffled dataset of\n    the input data.\n  '
    result = _process_file_infos(file_infos)

    def sequential_index_shuffle(seeds):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(result['num_elements'])

        def read_element(dataset, index):
            if False:
                return 10
            shuffled_index = stateless_random_ops.index_shuffle(index, seeds, result['num_elements'] - 1)
            if 'thresholds' in result and 'offsets' in result:
                shuffled_index = _adjust_index(shuffled_index, result['thresholds'], result['offsets'])
            return random_access.at(dataset, shuffled_index)
        map_func = functools.partial(read_element, reader_factory(result['files']))
        return dataset.map(map_func, num_parallel_calls=num_parallel_calls)
    rng_ds = dataset_ops.Dataset.random(seed=seed, rerandomize_each_iteration=reshuffle_each_iteration)
    rng_ds = rng_ds.take(2).batch(2, drop_remainder=True)
    return rng_ds.flat_map(sequential_index_shuffle)