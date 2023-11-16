"""Python wrapper for prefetching_ops."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as framework_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util.tf_export import tf_export

@tf_export('data.experimental.prefetch_to_device')
def prefetch_to_device(device, buffer_size=None):
    if False:
        i = 10
        return i + 15
    'A transformation that prefetches dataset values to the given `device`.\n\n  NOTE: Although the transformation creates a `tf.data.Dataset`, the\n  transformation must be the final `Dataset` in the input pipeline.\n\n  For example,\n  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n  >>> dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/cpu:0"))\n  >>> for element in dataset:\n  ...   print(f\'Tensor {element} is on device {element.device}\')\n  Tensor 1 is on device /job:localhost/replica:0/task:0/device:CPU:0\n  Tensor 2 is on device /job:localhost/replica:0/task:0/device:CPU:0\n  Tensor 3 is on device /job:localhost/replica:0/task:0/device:CPU:0\n\n  Args:\n    device: A string. The name of a device to which elements will be prefetched.\n    buffer_size: (Optional.) The number of elements to buffer on `device`.\n      Defaults to an automatically chosen value.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  '

    def _apply_fn(dataset):
        if False:
            i = 10
            return i + 15
        return dataset.apply(copy_to_device(target_device=device)).prefetch(buffer_size)
    return _apply_fn

@tf_export('data.experimental.copy_to_device')
def copy_to_device(target_device, source_device='/cpu:0'):
    if False:
        while True:
            i = 10
    'A transformation that copies dataset elements to the given `target_device`.\n\n  Args:\n    target_device: The name of a device to which elements will be copied.\n    source_device: The original device on which `input_dataset` will be placed.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  '

    def _apply_fn(dataset):
        if False:
            return 10
        return _CopyToDeviceDataset(dataset, target_device=target_device, source_device=source_device)
    return _apply_fn

class _CopyToDeviceDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that copies elements to another device."""

    def __init__(self, input_dataset, target_device, source_device='/cpu:0'):
        if False:
            print('Hello World!')
        'Constructs a _CopyToDeviceDataset.\n\n    Args:\n      input_dataset: `Dataset` to be copied\n      target_device: The name of the device to which elements would be copied.\n      source_device: Device where input_dataset would be placed.\n    '
        self._input_dataset = input_dataset._apply_debug_options()
        self._target_device = target_device
        spec = framework_device.DeviceSpec().from_string(self._target_device)
        self._is_gpu_target = spec.device_type == 'GPU'
        self._source_device_string = source_device
        self._source_device = ops.convert_to_tensor(source_device)
        wrap_ds_variant = gen_dataset_ops.wrap_dataset_variant(self._input_dataset._variant_tensor)

        @def_function.function()
        def _init_func():
            if False:
                for i in range(10):
                    print('nop')
            'Creates an iterator for the input dataset.\n\n      Returns:\n        A `string` tensor that encapsulates the iterator created.\n      '
            ds_variant = gen_dataset_ops.unwrap_dataset_variant(wrap_ds_variant)
            resource = gen_dataset_ops.anonymous_iterator(**self._input_dataset._flat_structure)
            with ops.control_dependencies([gen_dataset_ops.make_iterator(ds_variant, resource)]):
                return gen_dataset_ops.iterator_to_string_handle(resource)
        init_func_concrete = _init_func.get_concrete_function()

        @def_function.function()
        def _remote_init_func():
            if False:
                i = 10
                return i + 15
            return functional_ops.remote_call(target=self._source_device, args=init_func_concrete.captured_inputs, Tout=[dtypes.string], f=init_func_concrete)
        self._init_func = _remote_init_func.get_concrete_function()
        self._init_captured_args = self._init_func.captured_inputs

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
        def _next_func(string_handle):
            if False:
                return 10
            'Calls get_next for created iterator.\n\n      Args:\n        string_handle: An iterator string handle created by _init_func\n      Returns:\n        The elements generated from `input_dataset`\n      '
            with ops.device(self._source_device_string):
                iterator = iterator_ops.Iterator.from_string_handle(string_handle, dataset_ops.get_legacy_output_types(self), dataset_ops.get_legacy_output_shapes(self), dataset_ops.get_legacy_output_classes(self))
            return structure.to_tensor_list(self.element_spec, iterator.get_next())
        next_func_concrete = _next_func.get_concrete_function()

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)], experimental_attributes={'experimental_ints_on_device': True})
        def _remote_next_func(string_handle):
            if False:
                print('Hello World!')
            return functional_ops.remote_call(target=self._source_device, args=[string_handle] + next_func_concrete.captured_inputs, Tout=self._input_dataset._flat_types, f=next_func_concrete)
        self._next_func = _remote_next_func.get_concrete_function()
        self._next_captured_args = self._next_func.captured_inputs

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
        def _finalize_func(string_handle):
            if False:
                for i in range(10):
                    print('nop')
            'Destroys the iterator resource created.\n\n      Args:\n        string_handle: An iterator string handle created by _init_func\n      Returns:\n        Tensor constant 0\n      '
            iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(string_handle, **self._input_dataset._flat_structure)
            with ops.control_dependencies([resource_variable_ops.destroy_resource_op(iterator_resource, ignore_lookup_error=True)]):
                return array_ops.constant(0, dtypes.int64)
        finalize_func_concrete = _finalize_func.get_concrete_function()

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
        def _remote_finalize_func(string_handle):
            if False:
                print('Hello World!')
            return functional_ops.remote_call(target=self._source_device, args=[string_handle] + finalize_func_concrete.captured_inputs, Tout=[dtypes.int64], f=finalize_func_concrete)
        self._finalize_func = _remote_finalize_func.get_concrete_function()
        self._finalize_captured_args = self._finalize_func.captured_inputs
        g = ops.get_default_graph()
        self._init_func.add_to_graph(g)
        self._next_func.add_to_graph(g)
        self._finalize_func.add_to_graph(g)
        with ops.device(self._target_device):
            variant_tensor = gen_dataset_ops.generator_dataset(self._init_captured_args, self._next_captured_args, self._finalize_captured_args, init_func=self._init_func, next_func=self._next_func, finalize_func=self._finalize_func, **self._input_dataset._flat_structure)
        super(_CopyToDeviceDataset, self).__init__(input_dataset, variant_tensor)

    def make_one_shot_iterator(self):
        if False:
            print('Hello World!')
        if self._is_gpu_target:
            raise ValueError('`make_one_shot_iterator` is not compatible with GPU execution. Please use `Dataset.make_initializable_iterator()` instead.')
        else:
            return super(_CopyToDeviceDataset, self).make_one_shot_iterator()

class _MapOnGpuDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that maps a function over elements in its using a GPU."""

    def __init__(self, input_dataset, map_func, use_inter_op_parallelism=True):
        if False:
            for i in range(10):
                print('nop')
        'See `Dataset.map()` for details.'
        self._input_dataset = input_dataset
        self._use_inter_op_parallelism = use_inter_op_parallelism
        self._map_func = structured_function.StructuredFunctionWrapper(map_func, self._transformation_name(), dataset=input_dataset, defun_kwargs={'experimental_ints_on_device': True})
        variant_tensor = ged_ops.experimental_map_dataset(self._input_dataset._variant_tensor, self._map_func.function.captured_inputs, f=self._map_func.function, use_inter_op_parallelism=self._use_inter_op_parallelism, **self._flat_structure)
        super(_MapOnGpuDataset, self).__init__(input_dataset, variant_tensor)

    def _functions(self):
        if False:
            print('Hello World!')
        return [self._map_func]

    @property
    def element_spec(self):
        if False:
            for i in range(10):
                print('nop')
        return self._map_func.output_structure

    def _transformation_name(self):
        if False:
            print('Hello World!')
        return 'map_on_gpu()'

def map_on_gpu(map_func):
    if False:
        print('Hello World!')
    'Maps `map_func` across the elements of this dataset.\n\n  NOTE: This is a highly experimental version of `tf.data.Dataset.map` that runs\n  `map_func` on GPU. It must be used after applying the\n  `tf.data.experimental.copy_to_device` transformation with a GPU device\n  argument.\n\n  Args:\n    map_func: A function mapping a nested structure of tensors (having shapes\n      and types defined by `self.output_shapes` and `self.output_types`) to\n      another nested structure of tensors.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  '

    def _apply_fn(dataset):
        if False:
            for i in range(10):
                print('nop')
        return _MapOnGpuDataset(dataset, map_func)
    return _apply_fn