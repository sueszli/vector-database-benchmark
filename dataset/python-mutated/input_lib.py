"""Various classes representing distributed inputs."""
import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
_distributed_dataset_initialization_time_milliseconds = monitoring.Sampler('/tensorflow/api/distribution_strategy/distributed_dataset_initialization_time_milliseconds', monitoring.ExponentialBuckets(scale=1, growth_factor=2, bucket_count=26), 'Track the time (in milliseconds) to initialize distributed datasets.', 'strategy', 'workers')
_distributed_dataset_from_function_initialization_time_milliseconds = monitoring.Sampler('/tensorflow/api/distribution_strategy/distributed_dataset_from_function_initialization_time_milliseconds', monitoring.ExponentialBuckets(scale=1, growth_factor=2, bucket_count=26), 'Track the time (in milliseconds) to initialize distributed datasets from function.', 'strategy', 'workers')

def get_iterator_spec_from_dataset(strategy, dataset):
    if False:
        for i in range(10):
            print('nop')
    'Returns an iterator spec from dataset function.\n\n  This function constructs type spec for iterator obtained from\n  iter(dataset).\n\n  Args:\n    strategy: a `tf.distribute.Strategy` object, used to run all-reduce to\n        handle last partial batch.\n    dataset: A tf.data.Dataset instance. If using a function that returns a\n      tf.data.Dataset instance, pass dataset_fn.structured_outputs.\n\n  Returns:\n    A type_spec for iterator for dataset instance.\n\n  '
    output_element_spec = dataset.element_spec
    if isinstance(dataset._type_spec, (DistributedDatasetSpec, DistributedDatasetsFromFunctionSpec)):
        iterator_type_spec = DistributedIteratorSpec(strategy.extended._input_workers_with_options(), output_element_spec, strategy.extended._container_strategy(), options=None, cardinality=dataset.cardinality, enable_get_next_as_optional=True)
    else:
        if strategy.extended._num_gpus_per_worker:
            logging.warning(f'{strategy.extended._num_gpus_per_worker} GPUs are allocated per worker. Please use DistributedDataset by calling strategy.experimental_distribute_dataset or strategy.distribute_datasets_from_function to make best use of GPU resources')
        iterator_type_spec = iterator_ops.IteratorSpec(output_element_spec)
    return iterator_type_spec

class InputWorkers(object):
    """A 1-to-many mapping from input worker devices to compute devices."""

    def __init__(self, worker_device_pairs, canonicalize_devices=True):
        if False:
            i = 10
            return i + 15
        'Initialize an `InputWorkers` object.\n\n    Args:\n      worker_device_pairs: A sequence of pairs: `(input device, a tuple of\n        compute devices fed by that input device)`.\n      canonicalize_devices: Whether to canonicalize devices for workers fully or\n        partially. If False, it will partially canonicalize devices by removing\n        job and task.\n    '
        self._worker_device_pairs = worker_device_pairs
        self._input_worker_devices = tuple((d for (d, _) in self._worker_device_pairs))
        self._canonicalize_devices = canonicalize_devices
        if canonicalize_devices:
            self._fed_devices = tuple((tuple((device_util.canonicalize(d) for d in f)) for (_, f) in self._worker_device_pairs))
        else:
            self._fed_devices = tuple((tuple((device_util.canonicalize_without_job_and_task(d) for d in f)) for (_, f) in self._worker_device_pairs))

    @property
    def num_workers(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._input_worker_devices)

    @property
    def worker_devices(self):
        if False:
            i = 10
            return i + 15
        return self._input_worker_devices

    def compute_devices_for_worker(self, worker_index):
        if False:
            i = 10
            return i + 15
        return self._fed_devices[worker_index]

    def __repr__(self):
        if False:
            while True:
                i = 10
        devices = self.worker_devices
        debug_repr = ',\n'.join(('  %d %s: %s' % (i, devices[i], self._fed_devices[i]) for i in range(len(devices))))
        return '%s:{\n%s}' % (self.__class__.__name__, debug_repr)

    def serialize(self):
        if False:
            i = 10
            return i + 15
        return (self._worker_device_pairs, self._canonicalize_devices)

    def deserialize(self, serialized):
        if False:
            while True:
                i = 10
        return InputWorkers(serialized)

def _calculate_replicas_with_values(strategy, input_workers, optional_list):
    if False:
        return 10
    'Calcualates the number of replicas that have values.\n\n  Args:\n    strategy: the `tf.distribute.Strategy`.\n    input_workers: the `InputWorkers`.\n    optional_list: a list of lists `tf.experimental.Optional`. The values from\n      each compute device grouped by the input device.\n\n  Returns:\n    A scalar Tensor.\n  '
    worker_has_values = []
    for (worker, optionals) in zip(input_workers.worker_devices, optional_list):
        with ops.device(worker):
            device_has_values = [math_ops.cast(v.has_value(), dtypes.int64) for v in optionals]
            worker_has_values.append(math_ops.reduce_sum(device_has_values, keepdims=True))
    client_has_values = math_ops.reduce_sum(worker_has_values, keepdims=True)
    if strategy.extended._in_multi_worker_mode():
        global_has_values = strategy.reduce(reduce_util.ReduceOp.SUM, client_has_values, axis=None)
        return array_ops.reshape(global_has_values, [])
    else:
        return array_ops.reshape(client_has_values, [])

def _is_statically_shaped(element_spec):
    if False:
        return 10
    'Test if an iterator output is statically shaped.\n\n  For sparse and ragged tensors this only tests the batch dimension.\n\n  Args:\n    element_spec: a nest structure of `tf.TypeSpec`. The element spec of the\n      dataset of the iterator.\n\n  Returns:\n    True if the shape is static, false otherwise.\n  '
    for spec in nest.flatten(element_spec):
        if isinstance(spec, (sparse_tensor.SparseTensorSpec, ragged_tensor.RaggedTensorSpec)):
            if spec.shape.rank > 0 and spec.shape.as_list()[0] is None:
                return False
        else:
            for component in spec._flat_tensor_specs:
                if not component.shape.is_fully_defined():
                    return False
    return True

class DistributedIteratorBase(collections_abc.Iterator, distribute_types.DistributedIteratorInterface):
    """Common implementation for all input iterators."""

    def __init__(self, input_workers, iterators, strategy, cardinality, enable_get_next_as_optional, replica_order=None):
        if False:
            return 10
        assert isinstance(input_workers, InputWorkers)
        if not input_workers.worker_devices:
            raise ValueError('Should have at least one worker for input iterator.')
        self._iterators = iterators
        self._input_workers = input_workers
        self._strategy = strategy
        self._cardinality = cardinality
        self._enable_get_next_as_optional = enable_get_next_as_optional
        self._replica_order = replica_order

    def next(self):
        if False:
            while True:
                i = 10
        return self.__next__()

    def __next__(self):
        if False:
            i = 10
            return i + 15
        try:
            return self.get_next()
        except errors.OutOfRangeError:
            raise StopIteration

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def get_next_as_optional(self):
        if False:
            i = 10
            return i + 15
        if self._cardinality == cardinality_lib.INFINITE:
            return optional_ops.Optional.from_value(self._get_next_no_partial_batch_handling())
        if self._cardinality == 0 and (not self._strategy.extended._in_multi_worker_mode()):
            return optional_ops.Optional.empty(self._element_spec)
        optional_list = []
        for (i, worker) in enumerate(self._input_workers.worker_devices):
            with ops.device(worker):
                optional_list.append(self._iterators[i].get_next_as_optional_list())

        def _create_optional_with_dummy():
            if False:
                print('Hello World!')
            value_list = _get_value_or_dummy(self._input_workers, optional_list, produce_dummy=True)
            if self._replica_order is not None:
                value_list = self._reorder_replicas(value_list)
            per_replica = _create_per_replica(value_list, self._strategy)
            return optional_ops.Optional.from_value(per_replica)

        def _create_empty_optional():
            if False:
                while True:
                    i = 10
            return optional_ops.Optional.empty(self._element_spec)
        num_replicas_with_values = _calculate_replicas_with_values(self._strategy, self._input_workers, optional_list)
        return tf_cond.cond(num_replicas_with_values > 0, _create_optional_with_dummy, _create_empty_optional, strict=True)

    def get_next(self, name=None):
        if False:
            return 10
        'Returns the next input from the iterator for all replicas.'
        with distribute_lib.enter_or_assert_strategy(self._strategy):
            if distribute_lib.get_replica_context() is not None:
                raise ValueError('next(iterator) should be called from outside of replica_fn. e.g. strategy.run(replica_fn, args=(next(iterator),))')
        if not self._enable_get_next_as_optional:
            return self._get_next_no_partial_batch_handling(name)
        optional_list = []
        for (i, worker) in enumerate(self._input_workers.worker_devices):
            with ops.device(worker):
                optional_list.append(self._iterators[i].get_next_as_optional_list())
        num_replicas_with_values = _calculate_replicas_with_values(self._strategy, self._input_workers, optional_list)

        def _value_or_dummy():
            if False:
                return 10
            value_list = _get_value_or_dummy(self._input_workers, optional_list, produce_dummy=True)
            if self._replica_order is not None:
                value_list = self._reorder_replicas(value_list)
            return _create_per_replica(value_list, self._strategy)

        def _eof():
            if False:
                i = 10
                return i + 15
            return self._get_next_no_partial_batch_handling()
        return tf_cond.cond(num_replicas_with_values > 0, _value_or_dummy, _eof, strict=True)

    def _get_next_no_partial_batch_handling(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        replicas = []
        for (i, worker) in enumerate(self._input_workers.worker_devices):
            if name is not None:
                d = tf_device.DeviceSpec.from_string(worker)
                new_name = '%s_%s_%d' % (name, d.job, d.task)
            else:
                new_name = None
            with ops.device(worker):
                replicas.extend(self._iterators[i].get_next_as_list(new_name))
        if self._replica_order is not None:
            replicas = self._reorder_replicas(replicas)
        return _create_per_replica(replicas, self._strategy)

    def _reorder_replicas(self, replicas):
        if False:
            for i in range(10):
                print('nop')
        assert len(self._replica_order) == len(replicas), 'replica order size ({}) != replicas size ({})!'.format(len(self._replica_order), len(replicas))
        return [replicas[i] for i in self._replica_order]

class DistributedDatasetAndIteratorSpec(type_spec.TypeSpec):
    """Common Type specification for `DistributedDataset and DistributedDatasetsFromFunction."""
    __slots__ = ['_input_workers', '_element_spec', '_strategy', '_cardinality', '_enable_get_next_as_optional', '_options', '_canonicalize_devices']

    def __init__(self, input_workers, element_spec, strategy, options, cardinality=cardinality_lib.UNKNOWN, enable_get_next_as_optional=None, replica_order=None):
        if False:
            return 10
        if isinstance(input_workers, tuple):
            raise NotImplementedError('DistributedIteratorSpec does not have support for deserialization.')
        else:
            self._input_workers = input_workers
            self._element_spec = element_spec
            self._strategy = strategy
            self._cardinality = cardinality
            self._enable_get_next_as_optional = enable_get_next_as_optional
            self._options = options
            if self._strategy:
                self._canonicalize_devices = getattr(self._strategy, '_canonicalize_devices', True)
            else:
                self._canonicalize_devices = True
            self._replica_order = replica_order

    def _serialize(self):
        if False:
            i = 10
            return i + 15
        return (self._input_workers.serialize(), self._element_spec, id(self._strategy), id(self._options))

    def _deserialize(self):
        if False:
            return 10
        raise ValueError(f'Deserialization is currently unsupported for {type(self)}.')

    def sanity_check_type(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Returns the most specific TypeSpec compatible with `self` and `other`.\n\n    Args:\n      other: A `TypeSpec`.\n\n    Raises:\n      ValueError: If there is no TypeSpec that is compatible with both `self`\n        and `other`.\n    '
        if type(self) is not type(other):
            raise ValueError('No TypeSpec is compatible with both %s and %s' % (self, other))
        if self._input_workers.serialize() != other._input_workers.serialize():
            raise ValueError('_input_workers is not compatible with both %s and %s' % (self, other))
        if self._strategy is not other._strategy:
            raise ValueError('tf.distribute strategy is not compatible with both %s and %s' % (self, other))

    def is_subtype_of(self, other):
        if False:
            i = 10
            return i + 15
        'Returns True if `self` is subtype of `other`.\n\n    Args:\n      other: A `TypeSpec`.\n    '
        try:
            self.sanity_check_type(other)
            nest.assert_same_structure(self._element_spec, other._element_spec)
        except (TypeError, ValueError):
            return False
        self_elements = nest.flatten(self._element_spec)
        other_elements = nest.flatten(other._element_spec)
        return all((self_element.is_subtype_of(other_element) for (self_element, other_element) in zip(self_elements, other_elements)))

    def most_specific_common_supertype(self, others):
        if False:
            return 10
        'Returns the most specific supertype of `self` and `others`.\n\n    Args:\n      others: A Sequence of `TypeSpec`.\n\n    Returns `None` if a supertype does not exist.\n    '
        try:
            for other in others:
                self.sanity_check_type(other)
                nest.assert_same_structure(self._element_spec, other._element_spec)
        except (TypeError, ValueError):
            return None
        self_elements = nest.flatten(self._element_spec)
        others_elements = [nest.flatten(other._element_spec) for other in others]
        common_elements = [None] * len(self_elements)
        for (i, self_element) in enumerate(self_elements):
            common_elements[i] = self_element.most_specific_common_supertype([other_elements[i] for other_elements in others_elements])
            if common_elements[i] is None:
                return None
        common_element_spec = nest.pack_sequence_as(self._element_spec, common_elements)
        return type(self)(self._input_workers, common_element_spec, self._strategy, self._options, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional)

    def _with_tensor_ranks_only(self):
        if False:
            print('Hello World!')
        element_spec = nest.map_structure(lambda s: s._with_tensor_ranks_only(), self._element_spec)
        return type(self)(self._input_workers, element_spec, self._strategy, self._options, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional)

    def _without_tensor_names(self):
        if False:
            i = 10
            return i + 15
        element_spec = nest.map_structure(lambda s: s._without_tensor_names(), self._element_spec)
        return type(self)(self._input_workers, element_spec, self._strategy, self._options, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional)

class DistributedIteratorSpec(DistributedDatasetAndIteratorSpec):
    """Type specification for `DistributedIterator`."""

    @property
    def value_type(self):
        if False:
            print('Hello World!')
        return DistributedIterator

    @property
    def _component_specs(self):
        if False:
            print('Hello World!')
        specs = []
        worker_device_pairs = self._input_workers._worker_device_pairs
        for (i, (input_device, compute_devices)) in enumerate(worker_device_pairs):
            element_spec = nest.map_structure(functools.partial(_replace_per_replica_spec, i=i), self._element_spec)
            specs.append(_SingleWorkerDatasetIteratorSpec(input_device, compute_devices, element_spec, self._options, self._canonicalize_devices))
        return specs

    def _to_components(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value._iterators

    def _from_components(self, components):
        if False:
            for i in range(10):
                print('nop')
        return DistributedIterator(input_workers=self._input_workers, iterators=None, components=components, element_spec=self._element_spec, strategy=self._strategy, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional, options=self._options, replica_order=self._replica_order)

    @staticmethod
    def from_value(value):
        if False:
            print('Hello World!')
        return DistributedIteratorSpec(value._input_workers, value._element_spec, value._strategy, value._options, cardinality=value._cardinality, enable_get_next_as_optional=value._enable_get_next_as_optional)

class DistributedIterator(DistributedIteratorBase, composite_tensor.CompositeTensor):
    """Input Iterator for a distributed dataset."""

    def __init__(self, input_workers=None, iterators=None, strategy=None, components=None, element_spec=None, cardinality=cardinality_lib.UNKNOWN, enable_get_next_as_optional=False, options=None, replica_order=None):
        if False:
            i = 10
            return i + 15
        if input_workers is None:
            raise ValueError('`input_workers` should be provided.')
        error_message = 'Either `input_workers` or both `components` and `element_spec` need to be provided.'
        self._options = options
        if iterators is None:
            if components is None or element_spec is None:
                raise ValueError(error_message)
            self._element_spec = element_spec
            self._input_workers = input_workers
            self._iterators = components
            self._strategy = strategy
            self._cardinality = cardinality
            self._enable_get_next_as_optional = enable_get_next_as_optional
            self._replica_order = replica_order
        else:
            if components is not None and element_spec is not None:
                raise ValueError(error_message)
            super(DistributedIterator, self).__init__(input_workers, iterators, strategy, cardinality, enable_get_next_as_optional, replica_order)

    @property
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        if self._enable_get_next_as_optional and self._strategy.extended._in_multi_worker_mode():
            return nest.map_structure(_rebatch_as_dynamic, self._element_spec, expand_composites=False)
        return self._element_spec

    @property
    def _type_spec(self):
        if False:
            print('Hello World!')
        return DistributedIteratorSpec(self._input_workers, self._element_spec, self._strategy, self._options, self._cardinality, self._enable_get_next_as_optional, self._replica_order)

class _IterableInput(collections_abc.Iterable, distribute_types.DistributedDatasetInterface):
    """Base class for iterable inputs for distribution strategies."""

    def __init__(self, input_workers):
        if False:
            while True:
                i = 10
        assert isinstance(input_workers, InputWorkers)
        self._input_workers = input_workers

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('must be implemented in descendants')

    def reduce(self, initial_state, reduce_fn):
        if False:
            return 10
        'Execute a `reduce_fn` over all the elements of the input.'
        iterator = iter(self)
        optional_data = iterator.get_next_as_optional()

        def cond(optional_data, state):
            if False:
                i = 10
                return i + 15
            del state
            return optional_data.has_value()

        def loop_body(optional_data, state):
            if False:
                for i in range(10):
                    print('nop')
            'Executes `reduce_fn` in a loop till the dataset is empty.'
            state = reduce_fn(state, optional_data.get_value())
            optional_data = iterator.get_next_as_optional()
            return (optional_data, state)
        (optional_data, final_state) = while_loop.while_loop(cond, loop_body, [optional_data, initial_state], parallel_iterations=1, return_same_structure=True)
        return final_state

class DistributedDatasetSpec(DistributedDatasetAndIteratorSpec):
    """Type specification for `DistributedDataset."""

    @property
    def value_type(self):
        if False:
            i = 10
            return i + 15
        return DistributedDataset

    @property
    def _component_specs(self):
        if False:
            print('Hello World!')
        specs = []
        worker_device_pairs = self._input_workers._worker_device_pairs
        for (i, _) in enumerate(worker_device_pairs):
            element_spec = nest.map_structure(functools.partial(_replace_per_replica_spec, i=i), self._element_spec)
            specs.append(dataset_ops.DatasetSpec(element_spec))
        return specs

    def _to_components(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value._cloned_datasets

    def _from_components(self, components):
        if False:
            i = 10
            return i + 15
        return DistributedDataset(input_workers=self._input_workers, strategy=self._strategy, components=components, element_spec=self._element_spec, enable_get_next_as_optional=self._enable_get_next_as_optional, options=self._options, replica_order=self._replica_order)

    @staticmethod
    def from_value(value):
        if False:
            return 10
        return DistributedDatasetSpec(value._input_workers, value._element_spec, value._strategy, value._options, enable_get_next_as_optional=value._enable_get_next_as_optional)

class DistributedDataset(_IterableInput, composite_tensor.CompositeTensor):
    """Distributed dataset that supports prefetching to multiple devices."""

    def __init__(self, input_workers, strategy, dataset=None, num_replicas_in_sync=None, input_context=None, components=None, element_spec=None, enable_get_next_as_optional=None, build=True, options=None, replica_order=None):
        if False:
            return 10
        "Distribute the dataset on all workers.\n\n    If `num_replicas_in_sync` is not None, we split each batch of the dataset\n    into `num_replicas_in_sync` smaller batches, to be distributed among that\n    worker's replicas, so that the batch size for a global step (across all\n    workers and replicas) is as expected.\n\n    Args:\n      input_workers: an `InputWorkers` object.\n      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to\n        handle last partial batch.\n      dataset: `tf.data.Dataset` that will be used as the input source. Either\n        dataset or components field should be passed when constructing\n        DistributedDataset. Use this when contructing DistributedDataset from a\n        new `tf.data.Dataset`. Use components when constructing using\n        DistributedDatasetSpec.\n      num_replicas_in_sync: Optional integer. If this is not None, the value is\n        used to decide how to rebatch datasets into smaller batches so that the\n        total batch size for each step (across all workers and replicas) adds up\n        to `dataset`'s batch size.\n      input_context: `InputContext` for sharding. Only pass this in for between\n        graph multi-worker cases where there is only one `input_worker`. In\n        these cases, we will shard based on the `input_pipeline_id` and\n        `num_input_pipelines` in the `InputContext`.\n      components: datasets when DistributedDataset is constructed from\n        DistributedDatasetSpec. Either field dataset or components should be\n        passed.\n      element_spec: element spec for DistributedDataset when constructing from\n        DistributedDatasetSpec. This will be used to set the element_spec for\n        DistributedDataset and verified against element_spec from components.\n      enable_get_next_as_optional: this is required when components is passed\n        instead of dataset.\n      build: whether to build underlying datasets when this object is created.\n        This is only useful for `ParameterServerStrategy` now.\n      options: `tf.distribute.InputOptions` used to control options on how this\n        dataset is distributed.\n      replica_order: the order of the replicas, which will be used to reorder\n        the iterators to match the device order.\n    "
        super(DistributedDataset, self).__init__(input_workers=input_workers)
        if input_workers is None or strategy is None:
            raise ValueError('input_workers and strategy are required arguments')
        if dataset is not None and components is not None:
            raise ValueError('Only one of dataset or components should be present')
        if dataset is None and components is None:
            raise ValueError('At least one of dataset or components should be passed')
        self._input_workers = input_workers
        self._strategy = strategy
        self._options = options
        self._input_context = input_context
        self._num_replicas_in_sync = num_replicas_in_sync
        self._replica_order = replica_order
        if dataset is not None:
            self._original_dataset = dataset
            self._built = False
            if build:
                self.build()
        else:
            if not build:
                raise ValueError('When constructing DistributedDataset with components, build should not be False. This is an internal error. Please file a bug.')
            if enable_get_next_as_optional is None:
                raise ValueError('When constructing DistributedDataset with components, ' + 'enable_get_next_as_optional should also be passed')
            self._cloned_datasets = components
            self._cardinality = _cardinality(self._cloned_datasets[0])
            self._enable_get_next_as_optional = enable_get_next_as_optional
            assert element_spec is not None
            if element_spec != _create_distributed_tensor_spec(self._strategy, self._cloned_datasets[0].element_spec):
                raise ValueError('Mismatched element_spec from the passed components')
            self._element_spec = element_spec
            self._built = True

    def build(self, dataset_to_replace=None):
        if False:
            while True:
                i = 10
        assert not self._built
        dataset = dataset_to_replace or self._original_dataset
        self._cardinality = _cardinality(dataset)
        self._enable_get_next_as_optional = _enable_get_next_as_optional(self._strategy, dataset, self._cardinality)
        distribute_start_time_ns = time.time_ns()
        self._create_cloned_datasets_from_dataset(dataset, self._input_context, self._input_workers, self._strategy, self._num_replicas_in_sync)
        if context.executing_eagerly():
            context.async_wait()
            distribute_duration_ms = (time.time_ns() - distribute_start_time_ns) // 1000000
            _distributed_dataset_initialization_time_milliseconds.get_cell(self._strategy.__class__.__name__, str(self._input_workers.num_workers)).add(distribute_duration_ms)
        self._element_spec = _create_distributed_tensor_spec(self._strategy, self._cloned_datasets[0].element_spec)
        self._built = True

    def auto_shard(self, num_shards, shard_ix):
        if False:
            for i in range(10):
                print('nop')
        assert len(self._cloned_datasets) == len(self._input_workers.worker_devices), f'datasets: {len(self._cloned_datasets)}, input workers: {len(self._input_workers.worker_devices)}'
        sharded_datasets = []
        for i in range(len(self._input_workers.worker_devices)):
            with ops.colocate_with(self._cloned_datasets[i]._variant_tensor):
                sharded_datasets.append(input_ops.auto_shard_dataset(self._cloned_datasets[i], num_shards, shard_ix, self._num_replicas_in_sync))
        return DistributedDataset(self._input_workers, self._strategy, components=sharded_datasets, element_spec=self._element_spec, options=self._options, enable_get_next_as_optional=self._enable_get_next_as_optional)

    @property
    def cardinality(self):
        if False:
            while True:
                i = 10
        if not self._built:
            raise ValueError('Cannot get the cardinality of a dataset that is not built')
        return self._cardinality

    def _create_cloned_datasets_from_dataset(self, dataset, input_context, input_workers, strategy, num_replicas_in_sync):
        if False:
            print('Hello World!')
        if num_replicas_in_sync is not None and num_replicas_in_sync > 1:
            num_workers = input_context.num_input_pipelines if input_context else len(input_workers.worker_devices)
            rebatch_fn = self._make_rebatch_fn(dataset, num_workers, num_replicas_in_sync)
        else:
            rebatch_fn = None
        self._cloned_datasets = []
        if input_context:
            assert input_workers.num_workers == 1
            if rebatch_fn is not None:
                dataset = rebatch_fn(dataset, input_context.input_pipeline_id)
            dataset = input_ops.auto_shard_dataset(dataset, input_context.num_input_pipelines, input_context.input_pipeline_id, num_replicas_in_sync)
            self._cloned_datasets.append(dataset)
        else:
            replicated_ds = distribute.replicate(dataset, input_workers.worker_devices)
            for (i, worker) in enumerate(input_workers.worker_devices):
                with ops.device(worker):
                    cloned_dataset = replicated_ds[worker]
                    if rebatch_fn is not None:
                        cloned_dataset = rebatch_fn(cloned_dataset, i)
                    cloned_dataset = input_ops.auto_shard_dataset(cloned_dataset, len(input_workers.worker_devices), i, num_replicas_in_sync)
                    self._cloned_datasets.append(cloned_dataset)

    def _make_rebatch_fn(self, dataset, num_workers, num_replicas_in_sync):
        if False:
            while True:
                i = 10
        'Returns a callable that rebatches the input dataset.\n\n    Args:\n      dataset: A `tf.data.Dataset` representing the dataset to be distributed.\n      num_workers: An integer representing the number of workers to distribute\n        `dataset` among.\n      num_replicas_in_sync: An integer representing the number of replicas in\n        sync across all workers.\n    '
        if num_replicas_in_sync % num_workers:
            raise ValueError('tf.distribute expects every worker to have the same number of replicas. However, encountered `num_replicas_in_sync` ({}) that cannot be divided by `num_workers` ({})'.format(num_replicas_in_sync, num_workers))
        num_replicas_per_worker = num_replicas_in_sync // num_workers
        with ops.colocate_with(dataset._variant_tensor):
            batch_size = distribute.compute_batch_size(dataset)

        def rebatch_fn(dataset, worker_index):
            if False:
                i = 10
                return i + 15
            try:

                def apply_rebatch():
                    if False:
                        while True:
                            i = 10
                    batch_sizes = distribute.batch_sizes_for_worker(batch_size, num_workers, num_replicas_per_worker, worker_index)
                    return dataset.rebatch(batch_sizes).prefetch(num_replicas_per_worker)

                def apply_legacy_rebatch():
                    if False:
                        for i in range(10):
                            print('nop')
                    return distribute._LegacyRebatchDataset(dataset, num_replicas_in_sync).prefetch(num_replicas_per_worker)
                with ops.colocate_with(dataset._variant_tensor):
                    return tf_cond.cond(math_ops.not_equal(batch_size, -1), true_fn=apply_rebatch, false_fn=apply_legacy_rebatch)
            except errors.InvalidArgumentError as e:
                if 'without encountering a batch' in str(e):
                    six.reraise(ValueError, ValueError('Call the `batch` method on the input Dataset in order to be able to split your input across {} replicas.\n Please see the tf.distribute.Strategy guide. {}'.format(num_replicas_in_sync, e)), sys.exc_info()[2])
                else:
                    raise
        return rebatch_fn

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        if not (context.executing_eagerly() or ops.get_default_graph().building_function):
            raise RuntimeError('__iter__() is only supported inside of tf.function or when eager execution is enabled.')
        if not self._built:
            raise ValueError('To use this dataset, you need to pass this dataset to ClusterCoordinator.create_per_worker_dataset.')
        canonicalize_devices = getattr(self._strategy, '_canonicalize_devices', True)
        worker_iterators = _create_iterators_per_worker(self._cloned_datasets, self._input_workers, options=self._options, canonicalize_devices=canonicalize_devices)
        iterator = DistributedIterator(self._input_workers, worker_iterators, self._strategy, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional, options=self._options, replica_order=self._replica_order)
        iterator._element_spec = self._element_spec
        if context.executing_eagerly():
            context.async_wait()
        return iterator

    @property
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        'The type specification of an element of this dataset.'
        if self._enable_get_next_as_optional and self._strategy.extended._in_multi_worker_mode():
            return nest.map_structure(_rebatch_as_dynamic, self._element_spec, expand_composites=False)
        return self._element_spec

    @property
    def _type_spec(self):
        if False:
            for i in range(10):
                print('nop')
        return DistributedDatasetSpec(self._input_workers, self._element_spec, self._strategy, self._options, enable_get_next_as_optional=self._enable_get_next_as_optional)

class DistributedDatasetsFromFunctionSpec(DistributedDatasetAndIteratorSpec):
    """Type specification for `DistributedDatasetsFromFunction."""

    @property
    def value_type(self):
        if False:
            for i in range(10):
                print('nop')
        return DistributedDatasetsFromFunction

    @property
    def _component_specs(self):
        if False:
            for i in range(10):
                print('nop')
        specs = []
        worker_device_pairs = self._input_workers._worker_device_pairs
        for (i, _) in enumerate(worker_device_pairs):
            element_spec = nest.map_structure(functools.partial(_replace_per_replica_spec, i=i), self._element_spec)
            specs.append(dataset_ops.DatasetSpec(element_spec))
        return specs

    def _to_components(self, value):
        if False:
            i = 10
            return i + 15
        return value._datasets

    def _from_components(self, components):
        if False:
            for i in range(10):
                print('nop')
        return DistributedDatasetsFromFunction(input_workers=self._input_workers, strategy=self._strategy, components=components, element_spec=self._element_spec, options=self._options)

    @staticmethod
    def from_value(value):
        if False:
            for i in range(10):
                print('nop')
        return DistributedDatasetsFromFunctionSpec(input_workers=value._input_workers, element_spec=value._element_spec, strategy=value._strategy, options=value._options)

class DistributedDatasetsFromFunction(_IterableInput, composite_tensor.CompositeTensor):
    """Inputs created from dataset function."""

    def __init__(self, input_workers, strategy, input_contexts=None, dataset_fn=None, options=None, components=None, element_spec=None, build=True, replica_order=None):
        if False:
            for i in range(10):
                print('nop')
        'Makes an iterable from datasets created by the given function.\n\n    Args:\n      input_workers: an `InputWorkers` object.\n      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to\n        handle last partial batch.\n      input_contexts: A list of `InputContext` instances to be passed to call(s)\n        to `dataset_fn`. Length and order should match worker order in\n        `worker_device_pairs`.\n      dataset_fn: A function that returns a `Dataset` given an `InputContext`.\n        Either dataset_fn or components should be passed to construct\n        DistributedDatasetsFromFunction. Use this when constructing\n        DistributedDataset using a function. Use components when constructing\n        using DistributedDatasetsFromFunctionSpec.\n      options: `tf.distribute.InputOptions` used to control options on how this\n        dataset is distributed.\n      components: datasets when DistributedDatasetsFromFunction is constructed\n        from DistributedDatasetsFromFunctionSpec. Only one of dataset or\n        components should be passed.\n      element_spec: element spec for DistributedDataset when constructing from\n        DistributedDatasetSpec. This will be used to set the element_spec for\n        DistributedDatasetsFromFunctionSpec and verified against element_spec\n        from components.\n      build: whether to build underlying datasets when this object is created.\n        This is only useful for `ParameterServerStrategy` now.\n      replica_order: the order of the replicas, which will be used to reorder\n        the iterators to match the device order.\n    '
        super(DistributedDatasetsFromFunction, self).__init__(input_workers=input_workers)
        self._input_workers = input_workers
        self._strategy = strategy
        self._options = options
        self._replica_order = replica_order
        if dataset_fn is not None and components is not None:
            raise ValueError('Only one of dataset_fn or components should be set')
        if dataset_fn is None and components is None:
            raise ValueError('At least one of dataset_fn or components should be set')
        if dataset_fn is not None:
            if input_workers.num_workers != len(input_contexts):
                raise ValueError('Number of input workers (%d) is not same as number of input_contexts (%d)' % (input_workers.num_workers, len(input_contexts)))
            self._input_contexts = input_contexts
            self._num_replicas_in_sync = self._input_contexts[0].num_replicas_in_sync
            self._dataset_fn = dataset_fn
            self._built = False
            if build:
                self.build()
        else:
            if element_spec is None:
                raise ValueError('element_spec should also be passed when passing components')
            if not build:
                raise ValueError('When constructing DistributedDatasetFromFunction with components, build should not be False. This is an internal error. Please file a bug.')
            self._element_spec = element_spec
            self._datasets = components
            self._num_replicas_in_sync = None
            self._built = True
            self._cardinality = _cardinality(self._datasets[0])
            self._enable_get_next_as_optional = _enable_get_next_as_optional(self._strategy, self._datasets[0], self._cardinality)

    def build(self):
        if False:
            while True:
                i = 10
        assert not self._built
        distribute_start_time_ns = time.time_ns()
        (self._datasets, element_spec) = _create_datasets_from_function_with_input_context(self._input_contexts, self._input_workers, self._dataset_fn)
        if context.executing_eagerly():
            context.async_wait()
            distribute_duration_ms = (time.time_ns() - distribute_start_time_ns) // 1000000
            _distributed_dataset_from_function_initialization_time_milliseconds.get_cell(self._strategy.__class__.__name__, str(self._input_workers.num_workers)).add(distribute_duration_ms)
        self._element_spec = _create_distributed_tensor_spec(self._strategy, element_spec)
        self._cardinality = _cardinality(self._datasets[0])
        self._enable_get_next_as_optional = _enable_get_next_as_optional(self._strategy, self._datasets[0], self._cardinality)
        self._built = True

    def auto_shard(self, num_shards, shard_ix):
        if False:
            return 10
        assert len(self._datasets) == len(self._input_workers.worker_devices), f'datasets: {len(self._datasets)}, input workers: {len(self._input_workers.worker_devices)}'
        sharded_datasets = []
        for i in range(len(self._input_workers.worker_devices)):
            with ops.colocate_with(self._datasets[i]._variant_tensor):
                sharded_datasets.append(input_ops.auto_shard_dataset(self._datasets[i], num_shards, shard_ix, self._num_replicas_in_sync))
        return DistributedDatasetsFromFunction(self._input_workers, self._strategy, components=sharded_datasets, element_spec=self._element_spec, options=self._options)

    @property
    def cardinality(self):
        if False:
            i = 10
            return i + 15
        if not self._built:
            raise ValueError('Cannot get the cardinality of a dataset that is not built')
        return self._cardinality

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        if not (ops.executing_eagerly_outside_functions() or ops.get_default_graph().building_function):
            raise RuntimeError('__iter__() is only supported inside of tf.function or when eager execution is enabled.')
        if not self._built:
            raise ValueError('You need to use this dataset in ClusterCoordinator.create_per_worker_dataset.')
        canonicalize_devices = getattr(self._strategy, '_canonicalize_devices', True)
        iterators = _create_iterators_per_worker(self._datasets, self._input_workers, options=self._options, canonicalize_devices=canonicalize_devices)
        iterator = DistributedIterator(input_workers=self._input_workers, iterators=iterators, strategy=self._strategy, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional, options=self._options, replica_order=self._replica_order)
        iterator._element_spec = self._element_spec
        if context.executing_eagerly():
            context.async_wait()
        return iterator

    @property
    def element_spec(self):
        if False:
            for i in range(10):
                print('nop')
        'The type specification of an element of this dataset.'
        if self._enable_get_next_as_optional and self._strategy.extended._in_multi_worker_mode():
            return nest.map_structure(_rebatch_as_dynamic, self._element_spec, expand_composites=False)
        return self._element_spec

    @property
    def _type_spec(self):
        if False:
            return 10
        return DistributedDatasetsFromFunctionSpec(self._input_workers, self._element_spec, self._strategy, self._options)

def _dummy_tensor_fn(value_structure):
    if False:
        print('Hello World!')
    'A function to create dummy tensors from `value_structure`.'

    def create_dummy_tensor(spec):
        if False:
            i = 10
            return i + 15
        'Create a dummy tensor with possible batch dimensions set to 0.'
        if hasattr(spec, '_create_empty_value'):
            return spec._create_empty_value()
        if isinstance(spec, ragged_tensor.RaggedTensorSpec):
            feature_shape = spec._shape[:1].concatenate(spec._shape[1 + spec._ragged_rank:])
            feature_type = spec._dtype
        else:
            feature_shape = spec.shape
            feature_type = spec.dtype
        dims = [dim if dim is not None else 0 for dim in feature_shape.as_list()] if feature_shape else []
        if dims and (isinstance(spec, ragged_tensor.RaggedTensorSpec) or feature_shape.is_fully_defined()):
            dims[0] = tensor_shape.Dimension(0)
        if isinstance(spec, sparse_tensor.SparseTensorSpec):
            return sparse_tensor.SparseTensor(values=array_ops.zeros(0, feature_type), indices=array_ops.zeros((0, len(dims)), dtypes.int64), dense_shape=dims)
        dummy_tensor = array_ops.zeros(tensor_shape.TensorShape(dims), feature_type)
        if isinstance(spec, ragged_tensor.RaggedTensorSpec):
            row_splits = array_ops.zeros(1, spec._row_splits_dtype)
            dummy_tensor = ragged_tensor.RaggedTensor.from_nested_row_splits(dummy_tensor, (row_splits,) * spec._ragged_rank, validate=False)
        return dummy_tensor
    return nest.map_structure(create_dummy_tensor, value_structure)

def _get_value_or_dummy(input_workers, optional_list, produce_dummy):
    if False:
        i = 10
        return i + 15
    "Returns the value of the optionals or dummy values.\n\n  Args:\n    input_workers: the `InputWorkers`.\n    optional_list: a list of lists `tf.experimental.Optional`. The values from\n      each compute device grouped by the input device.\n    produce_dummy: a bool. Whether to produce dummy tensors when the optional\n      doesn't have a value.\n\n  Returns:\n    A flatten list of Tensors.\n\n  "
    value_list = []
    for (i, worker) in enumerate(input_workers.worker_devices):
        with ops.device(worker):
            devices = input_workers.compute_devices_for_worker(i)
            for (j, device) in enumerate(devices):
                with ops.device(device):
                    if produce_dummy:
                        value_list.append(tf_cond.cond(optional_list[i][j].has_value(), lambda : optional_list[i][j].get_value(), lambda : _dummy_tensor_fn(optional_list[i][j].element_spec), strict=True))
                    else:
                        value_list.append(optional_list[i][j].get_value())
    return value_list

class _SingleWorkerDatasetIteratorBase(object):
    """Iterator for a single `tf.data.Dataset`."""

    def __init__(self, dataset, worker, devices, options=None):
        if False:
            print('Hello World!')
        "Create iterator for the `dataset` to fetch data to worker's `devices` .\n\n    A `MultiDeviceIterator`  or `OwnedMultiDeviceIterator` is used to prefetch\n    input to the devices on the given worker.\n\n    Args:\n      dataset: A `tf.data.Dataset` instance.\n      worker: Worker on which ops should be created.\n      devices: Distribute data from `dataset` to these devices.\n      options: options.\n    "
        self._dataset = dataset
        self._worker = worker
        self._devices = devices
        self._element_spec = dataset.element_spec
        self._options = options
        self._make_iterator()

    def _make_iterator(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('must be implemented in descendants')

    def _format_data_list_with_options(self, data_list):
        if False:
            while True:
                i = 10
        'Change the data in to a list type if required.\n\n    The OwnedMultiDeviceIterator returns the list data type,\n    while the PER_REPLICA iterator (when used with prefetch disabled)\n    returns without the enclosed list. This is to fix the inconsistency.\n    Args:\n      data_list: data_list\n    Returns:\n      list\n    '
        if self._options and self._options.experimental_replication_mode == InputReplicationMode.PER_REPLICA and (not self._options.experimental_fetch_to_device):
            return [data_list]
        else:
            return data_list

    def get_next(self, device, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Get next element for the given device.'
        del name
        with ops.device(self._worker):
            if _should_use_multi_device_iterator(self._options):
                return self._iterator.get_next(device)
            else:
                return self._iterator.get_next()

    def get_next_as_list(self, name=None):
        if False:
            while True:
                i = 10
        "Get next element from the underlying iterator.\n\n    Runs the iterator get_next() within a device scope. Since this doesn't use\n    get_next_as_optional(), it is considerably faster than get_next_as_list(),\n    but it raises EOFError if any of the device doesn't get any data.\n\n    Args:\n      name: not used.\n\n    Returns:\n      A list consisting of the next data from each device.\n    "
        del name
        with ops.device(self._worker):
            return self._format_data_list_with_options(self._iterator.get_next())

    def get_next_as_optional_list(self):
        if False:
            return 10
        with ops.device(self._worker):
            return self._format_data_list_with_options(self._iterator.get_next_as_optional())

class _SingleWorkerDatasetIteratorSpec(type_spec.TypeSpec):
    """Type specification for `_SingleWorkerOwnedDatasetIterator`."""
    __slots__ = ['_worker', '_devices', '_element_spec', '_options', '_canonicalize_devices']

    def __init__(self, worker, devices, element_spec, options, canonicalize_devices=True):
        if False:
            print('Hello World!')
        self._worker = worker
        if canonicalize_devices:
            self._devices = tuple((device_util.canonicalize(d) for d in devices))
        else:
            self._devices = tuple((device_util.canonicalize_without_job_and_task(d) for d in devices))
        self._element_spec = element_spec
        self._options = options if options is not None else distribute_lib.InputOptions()
        self._canonicalize_devices = canonicalize_devices

    @property
    def value_type(self):
        if False:
            i = 10
            return i + 15
        return _SingleWorkerOwnedDatasetIterator

    def _serialize(self):
        if False:
            i = 10
            return i + 15
        return (self._worker, self._devices, self._element_spec, self._options, self._canonicalize_devices)

    def _get_multi_device_iterator_spec(self, specs):
        if False:
            while True:
                i = 10
        device_scope = device_util.canonicalize(self._worker, device_util.current())
        host_device = device_util.get_host_for_device(device_scope)
        worker = host_device
        specs.append(multi_device_iterator_ops.MultiDeviceIteratorSpec(self._devices, worker, element_spec=self._element_spec))

    @property
    def _component_specs(self):
        if False:
            print('Hello World!')
        specs = []
        if _should_use_multi_device_iterator(self._options):
            self._get_multi_device_iterator_spec(specs)
        else:
            specs.append(iterator_ops.IteratorSpec(element_spec=self._element_spec))
        return specs

    def _to_components(self, value):
        if False:
            while True:
                i = 10
        return [value._iterator]

    def _from_components(self, components):
        if False:
            i = 10
            return i + 15
        return _SingleWorkerOwnedDatasetIterator(dataset=None, worker=self._worker, devices=self._devices, components=components, element_spec=self._element_spec, options=self._options, canonicalize_devices=self._canonicalize_devices)

    @staticmethod
    def from_value(value):
        if False:
            for i in range(10):
                print('nop')
        return _SingleWorkerDatasetIteratorSpec(value._worker, value._devices, value._element_spec, value._options, value._canonicalize_devices)

class _SingleWorkerOwnedDatasetIterator(_SingleWorkerDatasetIteratorBase, composite_tensor.CompositeTensor):
    """Iterator for a DistributedDataset instance."""

    def __init__(self, dataset=None, worker=None, devices=None, components=None, element_spec=None, options=None, canonicalize_devices=None):
        if False:
            i = 10
            return i + 15
        "Create iterator for the `dataset` to fetch data to worker's `devices` .\n\n    `OwnedMultiDeviceIterator` is used to prefetch input to the devices on the\n    given worker. The lifetime of this iterator is tied to the encompassing\n    python object. Once we go out of scope of the python object or return from\n    a tf.function the underlying iterator resource is deleted.\n\n    Args:\n      dataset: A `tf.data.Dataset` instance.\n      worker: Worker on which ops should be created.\n      devices: Distribute data from `dataset` to these devices.\n      components: Tensor components to construct the\n        _SingleWorkerOwnedDatasetIterator from.\n      element_spec: A nested structure of `TypeSpec` objects that represents the\n      type specification of elements of the iterator.\n      options: `tf.distribute.InputOptions` used to control options on how this\n      dataset is distributed.\n      canonicalize_devices: Whether to canonicalize devices for workers fully or\n      partially. If False, it will partially canonicalize devices by removing\n      job and task.\n    "
        if worker is None or devices is None:
            raise ValueError('Both `worker` and `devices` should be provided')
        error_message = 'Either `dataset` or both `components` and `element_spec` need to be provided.'
        self._options = options
        self._canonicalize_devices = canonicalize_devices
        if dataset is None:
            if components is None or element_spec is None:
                raise ValueError(error_message)
            self._element_spec = element_spec
            self._worker = worker
            self._devices = devices
            self._iterator = components[0]
        else:
            if components is not None or element_spec is not None:
                raise ValueError(error_message)
            super(_SingleWorkerOwnedDatasetIterator, self).__init__(dataset, worker, devices, self._options)

    def _create_owned_multi_device_iterator(self):
        if False:
            return 10
        if not ops.inside_function():
            device_scope = device_util.canonicalize(self._worker, device_util.current())
            host_device = device_util.get_host_for_device(device_scope)
        else:
            (device_scope, host_device) = (self._worker, self._worker)
        with ops.device(device_scope):
            if self._options is not None:
                self._iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(self._dataset, self._devices, source_device=host_device, max_buffer_size=self._options.experimental_per_replica_buffer_size, prefetch_buffer_size=self._options.experimental_per_replica_buffer_size)
            else:
                self._iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(self._dataset, self._devices, source_device=host_device)

    def _make_iterator(self):
        if False:
            i = 10
            return i + 15
        'Make appropriate iterator on the dataset.'
        if not self._worker:
            raise ValueError('Worker device must be specified when creating an owned iterator.')
        if _should_use_multi_device_iterator(self._options):
            self._create_owned_multi_device_iterator()
        else:
            with ops.device(self._worker):
                self._iterator = iter(self._dataset)

    @property
    def element_spec(self):
        if False:
            print('Hello World!')
        return self._element_spec

    @property
    def _type_spec(self):
        if False:
            print('Hello World!')
        return _SingleWorkerDatasetIteratorSpec(self._worker, self._devices, self._element_spec, self._options, self._canonicalize_devices)

    @property
    def output_classes(self):
        if False:
            while True:
                i = 10
        'Returns the class of each component of an element of this iterator.\n\n    The expected values are `tf.Tensor` and `tf.SparseTensor`.\n\n    Returns:\n      A nested structure of Python `type` objects corresponding to each\n      component of an element of this dataset.\n    '
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), self._element_spec)

    @property
    def output_shapes(self):
        if False:
            i = 10
            return i + 15
        'Returns the shape of each component of an element of this iterator.\n\n    Returns:\n      A nested structure of `tf.TensorShape` objects corresponding to each\n      component of an element of this dataset.\n    '
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), self._element_spec)

    @property
    def output_types(self):
        if False:
            return 10
        'Returns the type of each component of an element of this iterator.\n\n    Returns:\n      A nested structure of `tf.DType` objects corresponding to each component\n      of an element of this dataset.\n    '
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), self._element_spec)

def _create_iterators_per_worker(worker_datasets, input_workers, options=None, canonicalize_devices=False):
    if False:
        print('Hello World!')
    'Create a multidevice iterator on each of the workers.'
    assert isinstance(input_workers, InputWorkers)
    assert len(worker_datasets) == len(input_workers.worker_devices)
    iterators = []
    for (i, worker) in enumerate(input_workers.worker_devices):
        with ops.device(worker):
            worker_devices = input_workers.compute_devices_for_worker(i)
            iterator = _SingleWorkerOwnedDatasetIterator(dataset=worker_datasets[i], worker=worker, devices=worker_devices, options=options, canonicalize_devices=canonicalize_devices)
            iterators.append(iterator)
    return iterators

def _create_datasets_from_function_with_input_context(input_contexts, input_workers, dataset_fn):
    if False:
        while True:
            i = 10
    'Create device datasets per worker given a dataset function.'
    datasets = []
    for (i, ctx) in enumerate(input_contexts):
        worker = input_workers.worker_devices[i]
        with ops.device(worker):
            dataset = dataset_fn(ctx)
            datasets.append(dataset)
    return (datasets, dataset.element_spec)

def _get_batched_dataset(d):
    if False:
        while True:
            i = 10
    'Get the batched dataset from `d`.'
    if isinstance(d, dataset_ops.DatasetV1Adapter):
        d = d._dataset
    if isinstance(d, (dataset_ops.BatchDataset, batching._MapAndBatchDataset)):
        return d
    elif isinstance(d, (dataset_ops.PrefetchDataset, dataset_ops._OptionsDataset)):
        return _get_batched_dataset(d._input_dataset)
    raise ValueError('Unable to get batched dataset from the input dataset. `batch` `map_and_batch` need to be the last operations on the dataset. The batch operations can be followed by a prefetch.')

def _get_batched_dataset_attributes(d):
    if False:
        i = 10
        return i + 15
    'Get `batch_size`, `drop_remainder` of dataset.'
    assert isinstance(d, (dataset_ops.BatchDataset, batching._MapAndBatchDataset))
    if isinstance(d, dataset_ops.BatchDataset):
        batch_size = d._batch_size
        drop_remainder = d._drop_remainder
    elif isinstance(d, batching._MapAndBatchDataset):
        batch_size = d._batch_size_t
        drop_remainder = d._drop_remainder_t
    if tensor_util.is_tf_type(batch_size):
        batch_size = tensor_util.constant_value(batch_size)
    if tensor_util.is_tf_type(drop_remainder):
        drop_remainder = tensor_util.constant_value(drop_remainder)
    return (batch_size, drop_remainder)

def _get_dataset_attributes(dataset):
    if False:
        i = 10
        return i + 15
    'Get the underlying attributes from the dataset object.'
    batched_dataset = _get_batched_dataset(dataset)
    (batch_size, drop_remainder) = _get_batched_dataset_attributes(batched_dataset)
    prefetch_buffer = None
    if isinstance(dataset, dataset_ops.PrefetchDataset):
        prefetch_buffer = dataset._buffer_size
    elif isinstance(dataset, dataset_ops.DatasetV1Adapter) and isinstance(dataset._dataset, dataset_ops.PrefetchDataset):
        prefetch_buffer = dataset._dataset._buffer_size
    return (batch_size, drop_remainder, prefetch_buffer)

def _should_use_multi_device_iterator(options):
    if False:
        for i in range(10):
            print('nop')
    'Determine whether to use multi_device_iterator_ops.'
    if options is None or options.experimental_replication_mode == InputReplicationMode.PER_WORKER or (options.experimental_replication_mode == InputReplicationMode.PER_REPLICA and options.experimental_fetch_to_device):
        return True
    return False

class MultiStepContext(object):
    """A context object that can be used to capture things when running steps.

  This context object is useful when running multiple steps at a time using the
  `experimental_run_steps_on_iterator` API. For e.g. it allows the user's step
  function to specify which outputs to emit at what frequency. Currently it
  supports capturing output from the last step, as well as capturing non tensor
  outputs.  In the future it will be augmented to support other use cases such
  as output each N steps.
  """

    def __init__(self):
        if False:
            print('Hello World!')
        'Initialize an output context.\n\n    Returns:\n      A context object.\n    '
        self._last_step_outputs = {}
        self._last_step_outputs_reduce_ops = {}
        self._non_tensor_outputs = {}

    @property
    def last_step_outputs(self):
        if False:
            i = 10
            return i + 15
        'A dictionary consisting of outputs to be captured on last step.\n\n    Keys in the dictionary are names of tensors to be captured, as specified\n    when `set_last_step_output` is called.\n    Values in the dictionary are the tensors themselves. If\n    `set_last_step_output` was called with a `reduce_op` for this output,\n    then the value is the reduced value.\n\n    Returns:\n      A dictionary with last step outputs.\n    '
        return self._last_step_outputs

    def _set_last_step_outputs(self, outputs):
        if False:
            return 10
        'Replace the entire dictionary of last step outputs.'
        if not isinstance(outputs, dict):
            raise ValueError('Need a dictionary to set last_step_outputs.')
        self._last_step_outputs = outputs

    def set_last_step_output(self, name, output, reduce_op=None):
        if False:
            i = 10
            return i + 15
        "Set `output` with `name` to be outputted from the last step.\n\n    Args:\n      name: String, name to identify the output. Doesn't need to match tensor\n        name.\n      output: The tensors that should be outputted with `name`. See below for\n        actual types supported.\n      reduce_op: Reduction method to use to reduce outputs from multiple\n        replicas. Required if `set_last_step_output` is called in a replica\n        context. Optional in cross_replica_context.\n        When present, the outputs from all the replicas are reduced using the\n        current distribution strategy's `reduce` method. Hence, the type of\n        `output` must be what's supported by the corresponding `reduce` method.\n        For e.g. if using MirroredStrategy and reduction is set, output\n        must be a `PerReplica` value.\n        The reduce method is also recorded in a dictionary\n        `_last_step_outputs_reduce_ops` for later interpreting of the\n        outputs as already reduced or not.\n    "
        if distribute_lib.in_cross_replica_context():
            self._last_step_outputs_reduce_ops[name] = reduce_op
            if reduce_op is None:
                self._last_step_outputs[name] = output
            else:
                distribution = distribute_lib.get_strategy()
                self._last_step_outputs[name] = distribution.reduce(reduce_op, output, axis=None)
        else:
            assert reduce_op is not None

            def merge_fn(distribution, value):
                if False:
                    print('Hello World!')
                self._last_step_outputs[name] = distribution.reduce(reduce_op, value, axis=None)
                self._last_step_outputs_reduce_ops[name] = reduce_op
            distribute_lib.get_replica_context().merge_call(merge_fn, args=(output,))

    @property
    def non_tensor_outputs(self):
        if False:
            return 10
        'A dictionary consisting of any non tensor outputs to be captured.'
        return self._non_tensor_outputs

    def set_non_tensor_output(self, name, output):
        if False:
            print('Hello World!')
        'Set `output` with `name` to be captured as a non tensor output.'
        if distribute_lib.in_cross_replica_context():
            self._non_tensor_outputs[name] = output
        else:

            def merge_fn(distribution, value):
                if False:
                    while True:
                        i = 10
                self._non_tensor_outputs[name] = distribution.experimental_local_results(value)
            distribute_lib.get_replica_context().merge_call(merge_fn, args=(output,))

def _create_distributed_tensor_spec(strategy, tensor_spec):
    if False:
        return 10
    'Create a `tf.TypeSpec` for a given strategy and input `tensor_spec`.\n\n  Args:\n    strategy: The given `tf.distribute` strategy.\n    tensor_spec: `tf.TensorSpec` of a given value. The batch dimension of the\n      shape should be None if you have partial batches.\n\n  Returns:\n    A `tf.TypeSpec` that matches the values produced by a given strategy. This\n    can be a `tf.TensorSpec` or a `PerRelicaSpec`.\n  '
    num_replicas = len(strategy.extended.worker_devices)
    if not _always_wrap(strategy):
        return tensor_spec

    def _get_value_per_replica(tensor_spec_per_input):
        if False:
            for i in range(10):
                print('nop')
        value_specs = [tensor_spec_per_input for _ in range(num_replicas)]
        return values.PerReplicaSpec(*value_specs)
    return nest.map_structure(_get_value_per_replica, tensor_spec)

def _replace_per_replica_spec(spec, i):
    if False:
        i = 10
        return i + 15
    'If `spec` is a `PerReplicaSpec`, then return its `i`th value_spec.'
    if isinstance(spec, values.PerReplicaSpec):
        return spec._value_specs[i]
    else:
        return spec

def _cardinality(dataset):
    if False:
        print('Hello World!')
    'Returns the cardinality of the dataset.'
    if context.executing_eagerly():
        with ops.device(dataset._variant_tensor.device):
            return dataset.cardinality().numpy()
    return cardinality_lib.UNKNOWN

def _enable_get_next_as_optional(strategy, dataset, cardinality):
    if False:
        return 10
    'Returns whether to enable using partial batch handling.'
    if not getattr(strategy.extended, 'enable_partial_batch_handling', getattr(strategy.extended, 'experimental_enable_get_next_as_optional', False)):
        return False
    if cardinality == cardinality_lib.INFINITE:
        return False
    return not _is_statically_shaped(dataset.element_spec) or strategy.extended._in_multi_worker_mode()

def _create_per_replica(value_list, strategy):
    if False:
        while True:
            i = 10
    'Creates a PerReplica.\n\n  For strategies other than OneDeviceStrategy, it creates a PerReplica whose\n  type spec is set to the element spec of the dataset. This helps avoid\n  retracing for partial batches. Retracing is problematic for multi client when\n  different client retraces different time, since retracing changes the\n  collective keys in the tf.function, and causes mismatches among clients.\n\n  For single client strategies, this simply calls distribute_utils.regroup().\n\n  Args:\n    value_list: a list of values, one for each replica.\n    strategy: the `tf.distribute.Strategy`.\n\n  Returns:\n    a structure of PerReplica.\n\n  '
    always_wrap = _always_wrap(strategy)
    per_replicas = distribute_utils.regroup(value_list, always_wrap=always_wrap)
    return per_replicas

def _always_wrap(strategy):
    if False:
        return 10
    'Returns whether to always wrap the values in a DistributedValues.'
    return strategy.extended._in_multi_worker_mode() or len(strategy.extended.worker_devices) > 1

def _rebatch_as_dynamic(per_replica_spec):
    if False:
        while True:
            i = 10
    'Rebatch the spec to have a dynamic batch dimension.'
    assert isinstance(per_replica_spec, values.PerReplicaSpec), per_replica_spec

    def _rebatch(spec):
        if False:
            i = 10
            return i + 15
        try:
            return spec._unbatch()._batch(None)
        except ValueError:
            pass
        return spec
    return values.PerReplicaSpec(*nest.map_structure(_rebatch, per_replica_spec._value_specs))

def _ag_enumerate_not_implemented(s, unused_start):
    if False:
        while True:
            i = 10
    msg = f'enumerate not supported with {s.__class__.__name__} types within tf.functions. Use a for loop over the dataset and keep a separate counter instead.'
    raise NotImplementedError(msg)
py_builtins.enumerate_registry.register(DistributedIterator, _ag_enumerate_not_implemented)
py_builtins.enumerate_registry.register(DistributedDataset, _ag_enumerate_not_implemented)