"""Python wrapper for prefetching_ops."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import prefetch_op
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import resource_variable_ops

class _PerDeviceGenerator(dataset_ops.DatasetV2):
    """A `dummy` generator dataset."""

    def __init__(self, shard_num, multi_device_iterator_resource, incarnation_id, source_device, element_spec, iterator_is_anonymous):
        if False:
            print('Hello World!')
        self._element_spec = element_spec
        multi_device_iterator_string_handle = gen_dataset_ops.multi_device_iterator_to_string_handle(multi_device_iterator_resource)

        @def_function.function(autograph=False)
        def _init_func():
            if False:
                return 10
            return multi_device_iterator_string_handle
        init_func_concrete = _init_func.get_concrete_function()

        @def_function.function(autograph=False)
        def _remote_init_func():
            if False:
                i = 10
                return i + 15
            return functional_ops.remote_call(target=source_device, args=init_func_concrete.captured_inputs, Tout=[dtypes.string], f=init_func_concrete)
        self._init_func = _remote_init_func.get_concrete_function()
        self._init_captured_args = self._init_func.captured_inputs

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)], autograph=False)
        def _next_func(string_handle):
            if False:
                print('Hello World!')
            multi_device_iterator = gen_dataset_ops.multi_device_iterator_from_string_handle(string_handle=string_handle, output_types=structure.get_flat_tensor_types(self._element_spec), output_shapes=structure.get_flat_tensor_shapes(self._element_spec))
            return gen_dataset_ops.multi_device_iterator_get_next_from_shard(multi_device_iterator=multi_device_iterator, shard_num=shard_num, incarnation_id=incarnation_id, output_types=structure.get_flat_tensor_types(self._element_spec), output_shapes=structure.get_flat_tensor_shapes(self._element_spec))
        next_func_concrete = _next_func.get_concrete_function()

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)], experimental_attributes={'experimental_ints_on_device': True}, autograph=False)
        def _remote_next_func(string_handle):
            if False:
                print('Hello World!')
            return_values = functional_ops.remote_call(target=source_device, args=[string_handle] + next_func_concrete.captured_inputs, Tout=structure.get_flat_tensor_types(self._element_spec), f=next_func_concrete)
            fulltype_list = type_utils.fulltypes_for_flat_tensors(self._element_spec)
            fulltype = type_utils.fulltype_list_to_product(fulltype_list)
            for return_value in return_values:
                return_value.op.experimental_set_type(fulltype)
            return return_values
        self._next_func = _remote_next_func.get_concrete_function()
        self._next_captured_args = self._next_func.captured_inputs
        if iterator_is_anonymous:
            self._next_captured_args = self._next_captured_args + [multi_device_iterator_resource]
        self._incarnation_id_index = -1
        for (i, arg) in enumerate(self._next_captured_args):
            if arg is incarnation_id:
                self._incarnation_id_index = i

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)], autograph=False)
        def _finalize_func(unused_string_handle):
            if False:
                while True:
                    i = 10
            return array_ops.constant(0, dtypes.int64)
        finalize_func_concrete = _finalize_func.get_concrete_function()

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)], autograph=False)
        def _remote_finalize_func(string_handle):
            if False:
                i = 10
                return i + 15
            return functional_ops.remote_call(target=source_device, args=[string_handle] + finalize_func_concrete.captured_inputs, Tout=[dtypes.int64], f=finalize_func_concrete)
        self._finalize_func = _remote_finalize_func.get_concrete_function()
        self._finalize_captured_args = self._finalize_func.captured_inputs
        variant_tensor = gen_dataset_ops.generator_dataset(self._init_captured_args, self._next_captured_args, self._finalize_captured_args, init_func=self._init_func, next_func=self._next_func, finalize_func=self._finalize_func, **self._flat_structure)
        super(_PerDeviceGenerator, self).__init__(variant_tensor)

    def _inputs(self):
        if False:
            print('Hello World!')
        return []

    @property
    def element_spec(self):
        if False:
            for i in range(10):
                print('nop')
        return self._element_spec

class _ReincarnatedPerDeviceGenerator(dataset_ops.DatasetV2):
    """Creates a _PerDeviceGenerator-like dataset with a new incarnation_id.

  Re-uses the functions from the provided per_device_dataset and just switches
  out the function argument corresponding to the incarnation_id.
  """

    def __init__(self, per_device_dataset, incarnation_id):
        if False:
            return 10
        self._element_spec = per_device_dataset.element_spec
        self._init_func = per_device_dataset._init_func
        self._init_captured_args = self._init_func.captured_inputs
        self._next_func = per_device_dataset._next_func
        self._next_captured_args = per_device_dataset._next_captured_args
        self._next_captured_args[per_device_dataset._incarnation_id_index] = incarnation_id
        self._finalize_func = per_device_dataset._finalize_func
        self._finalize_captured_args = per_device_dataset._finalize_captured_args
        variant_tensor = gen_dataset_ops.generator_dataset(self._init_captured_args, self._next_captured_args, self._finalize_captured_args, init_func=self._init_func, next_func=self._next_func, finalize_func=self._finalize_func, **self._flat_structure)
        super(_ReincarnatedPerDeviceGenerator, self).__init__(variant_tensor)

    def _inputs(self):
        if False:
            while True:
                i = 10
        return []

    @property
    def element_spec(self):
        if False:
            print('Hello World!')
        return self._element_spec

def _create_device_dataset(prototype_ds, incarnation_id, prefetch_buffer_size, experimental_slack):
    if False:
        i = 10
        return i + 15
    'Uses _prototype_device_datasets[i] to build a dataset for the device.'
    ds = _ReincarnatedPerDeviceGenerator(prototype_ds, incarnation_id)
    if prefetch_buffer_size > 0:
        if experimental_slack:
            ds = prefetch_op._PrefetchDataset(ds, prefetch_buffer_size, slack_period=1)
        else:
            ds = ds.prefetch(prefetch_buffer_size)
    return ds

class MultiDeviceIterator:
    """An iterator over multiple devices."""

    def __init__(self, dataset, devices, max_buffer_size=1, prefetch_buffer_size=1, source_device='/cpu:0'):
        if False:
            for i in range(10):
                print('nop')
        'Constructs a MultiDeviceIterator.\n\n    Args:\n      dataset: The input dataset to be iterated over.\n      devices: The list of devices to fetch data to.\n      max_buffer_size: Maximum size of the host side per device buffer to keep.\n      prefetch_buffer_size: if > 0, then we setup a buffer on each device to\n        prefetch into.\n      source_device: The host device to place the `dataset` on.  In order to\n        prevent deadlocks, if the prefetch_buffer_size is greater than the\n        max_buffer_size, we set the max_buffer_size to prefetch_buffer_size.\n    '
        options = options_lib.Options()
        options.experimental_distribute.num_devices = len(devices)
        if prefetch_buffer_size == 0:
            options.experimental_optimization.inject_prefetch = False
        dataset = dataset.with_options(options)
        self._dataset = dataset._apply_debug_options()
        self._experimental_slack = dataset.options().experimental_slack
        self._devices = devices
        self._source_device = source_device
        self._source_device_tensor = ops.convert_to_tensor(source_device)
        self._max_buffer_size = max_buffer_size
        self._prefetch_buffer_size = prefetch_buffer_size
        if self._prefetch_buffer_size > self._max_buffer_size:
            self._max_buffer_size = self._prefetch_buffer_size
        with ops.device(self._source_device):
            shared_name = ''
            if context.executing_eagerly():
                shared_name = context.anonymous_name()
            self._multi_device_iterator_resource = gen_dataset_ops.multi_device_iterator(devices=self._devices, shared_name=shared_name, container='', **self._dataset._flat_structure)
            if context.executing_eagerly():
                self._resource_deleter = resource_variable_ops.EagerResourceDeleter(handle=self._multi_device_iterator_resource, handle_device=self._source_device)
            self._incarnation_id = gen_dataset_ops.multi_device_iterator_init(self._dataset._variant_tensor, self._multi_device_iterator_resource, max_buffer_size=self._max_buffer_size)
        self._prototype_device_datasets = []
        for (i, device) in enumerate(self._devices):
            with ops.device(device):
                ds = _PerDeviceGenerator(i, self._multi_device_iterator_resource, self._incarnation_id, self._source_device_tensor, self._dataset.element_spec, iterator_is_anonymous=False)
                self._prototype_device_datasets.append(ds)
        self._device_iterators = []
        for (i, device) in enumerate(self._devices):
            with ops.device(device):
                ds = _create_device_dataset(self._prototype_device_datasets[i], self._incarnation_id, self._prefetch_buffer_size, self._experimental_slack)
                if context.executing_eagerly():
                    self._device_iterators.append(dataset_ops.make_one_shot_iterator(ds))
                else:
                    self._device_iterators.append(dataset_ops.make_initializable_iterator(ds))
        if not context.executing_eagerly():
            device_iterator_initializers = [iterator.initializer for iterator in self._device_iterators]
            self._initializer = control_flow_ops.group(*device_iterator_initializers)

    def get_next(self, device=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the next element given a `device`, else returns all in a list.'
        if device is not None:
            index = self._devices.index(device)
            return self._device_iterators[index].get_next()
        result = []
        for (i, device) in enumerate(self._devices):
            with ops.device(device):
                result.append(self._device_iterators[i].get_next())
        return result

    def get_next_as_optional(self):
        if False:
            i = 10
            return i + 15
        result = []
        for (i, device) in enumerate(self._devices):
            with ops.device(device):
                result.append(self._device_iterators[i].get_next_as_optional())
        return result

    @property
    def initializer(self):
        if False:
            i = 10
            return i + 15
        if context.executing_eagerly():
            return control_flow_ops.no_op()
        return self._initializer

    def _eager_reset(self):
        if False:
            return 10
        'Resets the MultiDeviceIterator in eager mode.'
        if not ops.executing_eagerly_outside_functions():
            raise ValueError('Resetting a multi-device iterator is only supported in the eager mode.')
        self._incarnation_id = gen_dataset_ops.multi_device_iterator_init(self._dataset._variant_tensor, self._multi_device_iterator_resource, max_buffer_size=self._max_buffer_size)
        for (i, device) in enumerate(self._devices):
            with ops.device(device):
                ds = _create_device_dataset(self._prototype_device_datasets[i], self._incarnation_id, self._prefetch_buffer_size, self._experimental_slack)
                ds_variant = ds._variant_tensor
                gen_dataset_ops.make_iterator(ds_variant, self._device_iterators[i]._iterator_resource)

    @property
    def element_spec(self):
        if False:
            return 10
        return self._dataset.element_spec

class MultiDeviceIteratorSpec(type_spec.TypeSpec):
    """Type specification for `OwnedMultiDeviceIterator`."""
    __slots__ = ['_devices', '_source_device', '_element_spec']

    def __init__(self, devices, source_device, element_spec):
        if False:
            print('Hello World!')
        self._devices = devices
        self._source_device = source_device
        self._element_spec = element_spec

    @property
    def value_type(self):
        if False:
            while True:
                i = 10
        return OwnedMultiDeviceIterator

    def _serialize(self):
        if False:
            return 10
        return (tuple(self._devices), self._source_device, self._element_spec)

    @property
    def _component_specs(self):
        if False:
            print('Hello World!')
        specs = [tensor_spec.TensorSpec([], dtypes.resource)]
        for _ in range(len(self._devices)):
            specs.append(iterator_ops.IteratorSpec(self._element_spec))
        return specs

    def _to_components(self, value):
        if False:
            print('Hello World!')
        c = [value._multi_device_iterator_resource]
        c.extend(value._device_iterators)
        return c

    def _from_components(self, components):
        if False:
            print('Hello World!')
        return OwnedMultiDeviceIterator(dataset=None, devices=self._devices, source_device=self._source_device, components=components, element_spec=self._element_spec)

    @staticmethod
    def from_value(value):
        if False:
            while True:
                i = 10
        return MultiDeviceIteratorSpec(value._devices, value._source_device, value.element_spec)

class OwnedMultiDeviceIterator(composite_tensor.CompositeTensor):
    """An iterator over multiple devices.

  The multi-device iterator resource created through `OwnedMultiDeviceIterator`
  is owned by the Python object and the life time of the underlying resource is
  tied to the life time of the `OwnedMultiDeviceIterator` object. This makes
  `OwnedMultiDeviceIterator` appropriate for use in eager mode and inside of
  tf.functions.
  """

    def __init__(self, dataset=None, devices=None, max_buffer_size=1, prefetch_buffer_size=1, source_device='/cpu:0', components=None, element_spec=None):
        if False:
            while True:
                i = 10
        'Constructs an owned MultiDeviceIterator object.\n\n    Args:\n      dataset: The input dataset to be iterated over.\n      devices: (Required.) The list of devices to fetch data to.\n      max_buffer_size: Maximum size of the host side per device buffer to keep.\n      prefetch_buffer_size: if > 0, then we setup a buffer on each device to\n        prefetch into.\n      source_device: The host device to place the `dataset` on.  In order to\n        prevent deadlocks, if the prefetch_buffer_size is greater than the\n        max_buffer_size, we set the max_buffer_size to prefetch_buffer_size.\n      components: Tensor components to construct the MultiDeviceIterator from.\n      element_spec: A (nested) structure of `tf.TypeSpec` objects that\n        represents the type specification of elements of the iterator.\n\n    Raises:\n      RuntimeError: If executed in graph mode or outside of function building\n        mode.\n      ValueError: If any of the following happens:\n        - `devices` is `None`\n        - `dataset` is `None` and either `components` or `element_spec` is\n          `None`\n        - `dataset` is not None and either `components` or `element_spec` is\n          provided\n    '
        if not context.executing_eagerly() and (not ops.inside_function()):
            raise RuntimeError('OwnedMultiDeviceIterator is only supported inside of tf.function or when eager execution is enabled.')
        if devices is None:
            raise ValueError('`devices` must be provided.')
        if dataset is None:
            if components is None or element_spec is None:
                raise ValueError('When `dataset` is not provided, both `components` and `element_spec` must be specified.')
            self._element_spec = element_spec
            self._devices = devices
            self._source_device = source_device
            self._multi_device_iterator_resource = components[0]
            self._device_iterators = components[1:]
        else:
            if components is not None or element_spec is not None:
                raise ValueError('When `dataset` is provided, `element_spec` and `components` must not be specified.')
            options = options_lib.Options()
            options.experimental_distribute.num_devices = len(devices)
            if prefetch_buffer_size == 0:
                options.experimental_optimization.inject_prefetch = False
            dataset = dataset.with_options(options)
            dataset = dataset._apply_debug_options()
            self._element_spec = dataset.element_spec
            experimental_slack = dataset.options().experimental_slack
            self._devices = devices
            self._source_device = source_device
            source_device_tensor = ops.convert_to_tensor(self._source_device)
            if prefetch_buffer_size > max_buffer_size:
                max_buffer_size = prefetch_buffer_size
            with ops.device(self._source_device):
                self._multi_device_iterator_resource = gen_dataset_ops.anonymous_multi_device_iterator_v3(devices=self._devices, **dataset._flat_structure)
                incarnation_id = gen_dataset_ops.multi_device_iterator_init(dataset._variant_tensor, self._multi_device_iterator_resource, max_buffer_size=max_buffer_size)
            prototype_device_datasets = []
            for (i, device) in enumerate(self._devices):
                with ops.device(device):
                    ds = _PerDeviceGenerator(i, self._multi_device_iterator_resource, incarnation_id, source_device_tensor, dataset.element_spec, iterator_is_anonymous=True)
                    prototype_device_datasets.append(ds)
            self._device_iterators = []
            for (i, device) in enumerate(self._devices):
                with ops.device(device):
                    ds = _create_device_dataset(prototype_device_datasets[i], incarnation_id, prefetch_buffer_size, experimental_slack)
                    iterator = iter(ds)
                    self._device_iterators.append(iterator)

    def get_next(self, device=None):
        if False:
            i = 10
            return i + 15
        'Returns the next element given a `device`, else returns all in a list.'
        if device is not None:
            index = self._devices.index(device)
            return self._device_iterators[index].get_next()
        result = []
        for (i, device) in enumerate(self._devices):
            with ops.device(device):
                result.append(self._device_iterators[i].get_next())
        return result

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def next(self):
        if False:
            while True:
                i = 10
        return self.__next__()

    def __next__(self):
        if False:
            return 10
        try:
            return self.get_next()
        except errors.OutOfRangeError:
            raise StopIteration

    def get_next_as_optional(self):
        if False:
            return 10
        result = []
        for (i, device) in enumerate(self._devices):
            with ops.device(device):
                result.append(self._device_iterators[i].get_next_as_optional())
        return result

    @property
    def element_spec(self):
        if False:
            return 10
        return self._element_spec

    @property
    def _type_spec(self):
        if False:
            while True:
                i = 10
        return MultiDeviceIteratorSpec(self._devices, self._source_device, self._element_spec)