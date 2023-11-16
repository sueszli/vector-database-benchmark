"""Non-deterministic dataset transformations."""
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@deprecation.deprecated(None, 'Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.deterministic`.')
@tf_export('data.experimental.parallel_interleave')
def parallel_interleave(map_func, cycle_length, block_length=1, sloppy=False, buffer_output_elements=None, prefetch_input_elements=None):
    if False:
        i = 10
        return i + 15
    'A parallel version of the `Dataset.interleave()` transformation.\n\n  `parallel_interleave()` maps `map_func` across its input to produce nested\n  datasets, and outputs their elements interleaved. Unlike\n  `tf.data.Dataset.interleave`, it gets elements from `cycle_length` nested\n  datasets in parallel, which increases the throughput, especially in the\n  presence of stragglers. Furthermore, the `sloppy` argument can be used to\n  improve performance, by relaxing the requirement that the outputs are produced\n  in a deterministic order, and allowing the implementation to skip over nested\n  datasets whose elements are not readily available when requested.\n\n  Example usage:\n\n  ```python\n  # Preprocess 4 files concurrently.\n  filenames = tf.data.Dataset.list_files("/path/to/data/train*.tfrecords")\n  dataset = filenames.apply(\n      tf.data.experimental.parallel_interleave(\n          lambda filename: tf.data.TFRecordDataset(filename),\n          cycle_length=4))\n  ```\n\n  WARNING: If `sloppy` is `True`, the order of produced elements is not\n  deterministic.\n\n  Args:\n    map_func: A function mapping a nested structure of tensors to a `Dataset`.\n    cycle_length: The number of input `Dataset`s to interleave from in parallel.\n    block_length: The number of consecutive elements to pull from an input\n      `Dataset` before advancing to the next input `Dataset`.\n    sloppy: A boolean controlling whether determinism should be traded for\n      performance by allowing elements to be produced out of order.  If `sloppy`\n      is `None`, the `tf.data.Options.deterministic` dataset option (`True` by\n      default) is used to decide whether to enforce a deterministic order.\n    buffer_output_elements: The number of elements each iterator being\n      interleaved should buffer (similar to the `.prefetch()` transformation for\n      each interleaved iterator).\n    prefetch_input_elements: The number of input elements to transform to\n      iterators before they are needed for interleaving.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  '

    def _apply_fn(dataset):
        if False:
            print('Hello World!')
        return readers.ParallelInterleaveDataset(dataset, map_func, cycle_length, block_length, sloppy, buffer_output_elements, prefetch_input_elements)
    return _apply_fn

@deprecation.deprecated(None, 'Use `tf.data.Dataset.sample_from_datasets(...)`.')
@tf_export('data.experimental.sample_from_datasets', v1=[])
def sample_from_datasets_v2(datasets, weights=None, seed=None, stop_on_empty_dataset=False):
    if False:
        while True:
            i = 10
    'Samples elements at random from the datasets in `datasets`.\n\n  Creates a dataset by interleaving elements of `datasets` with `weight[i]`\n  probability of picking an element from dataset `i`. Sampling is done without\n  replacement. For example, suppose we have 2 datasets:\n\n  ```python\n  dataset1 = tf.data.Dataset.range(0, 3)\n  dataset2 = tf.data.Dataset.range(100, 103)\n  ```\n\n  Suppose also that we sample from these 2 datasets with the following weights:\n\n  ```python\n  sample_dataset = tf.data.Dataset.sample_from_datasets(\n      [dataset1, dataset2], weights=[0.5, 0.5])\n  ```\n\n  One possible outcome of elements in sample_dataset is:\n\n  ```\n  print(list(sample_dataset.as_numpy_iterator()))\n  # [100, 0, 1, 101, 2, 102]\n  ```\n\n  Args:\n    datasets: A non-empty list of `tf.data.Dataset` objects with compatible\n      structure.\n    weights: (Optional.) A list or Tensor of `len(datasets)` floating-point\n      values where `weights[i]` represents the probability to sample from\n      `datasets[i]`, or a `tf.data.Dataset` object where each element is such a\n      list. Defaults to a uniform distribution across `datasets`.\n    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random\n      seed that will be used to create the distribution. See\n      `tf.random.set_seed` for behavior.\n    stop_on_empty_dataset: If `True`, sampling stops if it encounters an empty\n      dataset. If `False`, it skips empty datasets. It is recommended to set it\n      to `True`. Otherwise, the distribution of samples starts off as the user\n      intends, but may change as input datasets become empty. This can be\n      difficult to detect since the dataset starts off looking correct. Default\n      to `False` for backward compatibility.\n\n  Returns:\n    A dataset that interleaves elements from `datasets` at random, according to\n    `weights` if provided, otherwise with uniform probability.\n\n  Raises:\n    TypeError: If the `datasets` or `weights` arguments have the wrong type.\n    ValueError:\n      - If `datasets` is empty, or\n      - If `weights` is specified and does not match the length of `datasets`.\n  '
    return dataset_ops.Dataset.sample_from_datasets(datasets=datasets, weights=weights, seed=seed, stop_on_empty_dataset=stop_on_empty_dataset)

@deprecation.deprecated(None, 'Use `tf.data.Dataset.sample_from_datasets(...)`.')
@tf_export(v1=['data.experimental.sample_from_datasets'])
def sample_from_datasets_v1(datasets, weights=None, seed=None, stop_on_empty_dataset=False):
    if False:
        for i in range(10):
            print('nop')
    return dataset_ops.DatasetV1Adapter(sample_from_datasets_v2(datasets, weights, seed, stop_on_empty_dataset))
sample_from_datasets_v1.__doc__ = sample_from_datasets_v2.__doc__

@deprecation.deprecated(None, 'Use `tf.data.Dataset.choose_from_datasets(...)` instead. Note that, unlike the experimental endpoint, the non-experimental endpoint sets `stop_on_empty_dataset=True` by default. You should set this argument explicitly in case you would like to match the behavior of the experimental endpoint.')
@tf_export('data.experimental.choose_from_datasets', v1=[])
def choose_from_datasets_v2(datasets, choice_dataset, stop_on_empty_dataset=False):
    if False:
        while True:
            i = 10
    'Creates a dataset that deterministically chooses elements from `datasets`.\n\n  For example, given the following datasets:\n\n  ```python\n  datasets = [tf.data.Dataset.from_tensors("foo").repeat(),\n              tf.data.Dataset.from_tensors("bar").repeat(),\n              tf.data.Dataset.from_tensors("baz").repeat()]\n\n  # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.\n  choice_dataset = tf.data.Dataset.range(3).repeat(3)\n\n  result = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)\n  ```\n\n  The elements of `result` will be:\n\n  ```\n  "foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"\n  ```\n\n  Args:\n    datasets: A non-empty list of `tf.data.Dataset` objects with compatible\n      structure.\n    choice_dataset: A `tf.data.Dataset` of scalar `tf.int64` tensors between `0`\n      and `len(datasets) - 1`.\n    stop_on_empty_dataset: If `True`, selection stops if it encounters an empty\n      dataset. If `False`, it skips empty datasets. It is recommended to set it\n      to `True`. Otherwise, the selected elements start off as the user intends,\n      but may change as input datasets become empty. This can be difficult to\n      detect since the dataset starts off looking correct. Default to `False`\n      for backward compatibility.\n\n  Returns:\n    A dataset that interleaves elements from `datasets` according to the values\n    of `choice_dataset`.\n\n  Raises:\n    TypeError: If `datasets` or `choice_dataset` has the wrong type.\n    ValueError: If `datasets` is empty.\n  '
    return dataset_ops.Dataset.choose_from_datasets(datasets=datasets, choice_dataset=choice_dataset, stop_on_empty_dataset=stop_on_empty_dataset)

@deprecation.deprecated(None, 'Use `tf.data.Dataset.choose_from_datasets(...)` instead. Note that, unlike the experimental endpoint, the non-experimental endpoint sets `stop_on_empty_dataset=True` by default. You should set this argument explicitly in case you would like to match the behavior of the experimental endpoint.')
@tf_export(v1=['data.experimental.choose_from_datasets'])
def choose_from_datasets_v1(datasets, choice_dataset, stop_on_empty_dataset=False):
    if False:
        return 10
    return dataset_ops.DatasetV1Adapter(choose_from_datasets_v2(datasets, choice_dataset, stop_on_empty_dataset))
choose_from_datasets_v1.__doc__ = choose_from_datasets_v2.__doc__
if tf2.enabled():
    choose_from_datasets = choose_from_datasets_v2
    sample_from_datasets = sample_from_datasets_v2
else:
    choose_from_datasets = choose_from_datasets_v1
    sample_from_datasets = sample_from_datasets_v1