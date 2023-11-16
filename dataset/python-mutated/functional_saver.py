"""Saves and restore variables inside traced @tf.functions."""
from typing import Callable, Mapping, Sequence
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
RegisteredSaversDict = Mapping[registration.RegisteredSaver, Mapping[str, base.Trackable]]
MappedCapturesCallable = Callable[[core.ConcreteFunction, Sequence[tensor_lib.Tensor]], tensor_lib.Tensor]

def _single_task_save(file_prefix: tensor_lib.Tensor, tensor_slice_dict: sharding_util.TensorSliceDict, options: 'checkpoint_options.CheckpointOptions | None'=None) -> ops.Operation:
    if False:
        for i in range(10):
            print('nop')
    'Save the saveable objects to a checkpoint with `file_prefix`.\n\n  Args:\n    file_prefix: A string or scalar string Tensor containing the prefix to\n      save under.\n    tensor_slice_dict: A dict mapping checkpoint key -> slice_spec -> tensor.\n      Tensors in this structure must belong to the same task, but may belong to\n      different devices within that task.\n    options: Optional `CheckpointOptions` object.\n\n  Returns:\n    An `Operation`, or None when executing eagerly.\n  '
    options = options or checkpoint_options.CheckpointOptions()
    tensor_names = []
    tensors = []
    slice_specs = []
    for (checkpoint_key, tensor_slices) in tensor_slice_dict.items():
        for (slice_spec, tensor) in tensor_slices.items():
            if isinstance(tensor, saveable_object.SaveSpec):
                tensor_value = tensor.tensor
                if tensor_value is not None:
                    tensor_names.append(tensor.name)
                    tensors.append(tensor_value)
                    slice_specs.append(tensor.slice_spec)
            else:
                tensor_names.append(checkpoint_key)
                tensors.append(tensor)
                slice_specs.append(slice_spec)
    save_device_spec = options.experimental_io_device or (len(tensors) and saveable_object_util.set_cpu0(tensors[0].device))
    save_device_spec = save_device_spec or 'cpu:0'
    with ops.device(save_device_spec):
        return io_ops.save_v2(file_prefix, tensor_names, slice_specs, tensors)

def _single_task_restore(file_prefix: tensor_lib.Tensor, tensor_slice_dict: sharding_util.TensorSliceDict, options: 'checkpoint_options.CheckpointOptions | None'=None) -> sharding_util.TensorSliceDict:
    if False:
        print('Hello World!')
    'Restore the saveable objects from a checkpoint with `file_prefix`.\n\n  Args:\n    file_prefix: A string or scalar string Tensor containing the prefix for\n      files to read from.\n    tensor_slice_dict: A dict mapping checkpoint key -> slice_spec -> tensor.\n    options: Optional `CheckpointOptions` object.\n\n  Returns:\n    A restored tensor dict (maps checkpoint_key -> slice_spec -> tensor).\n  '
    options = options or checkpoint_options.CheckpointOptions()
    tensor_names = []
    tensor_dtypes = []
    slice_specs = []
    for (checkpoint_key, tensor_slices) in tensor_slice_dict.items():
        for (slice_spec, tensor) in tensor_slices.items():
            tensor_dtypes.append(tensor.dtype)
            if isinstance(tensor, saveable_object.SaveSpec):
                slice_specs.append(tensor.slice_spec)
                tensor_names.append(tensor.name)
            else:
                slice_specs.append(slice_spec)
                tensor_names.append(checkpoint_key)
    restore_device_spec = options.experimental_io_device or 'cpu:0'
    with ops.device(restore_device_spec):
        restored_tensors = io_ops.restore_v2(file_prefix, tensor_names, slice_specs, tensor_dtypes)
    restored_tensor_dict = {}
    for (checkpoint_key, tensor_slices) in tensor_slice_dict.items():
        for slice_spec in tensor_slices:
            restored_tensor = restored_tensors.pop(0)
            restored_tensor_dict.setdefault(checkpoint_key, {})[slice_spec] = restored_tensor
    return restored_tensor_dict

def sharded_filename(filename_tensor: tensor_lib.Tensor, shard: int, num_shards: tensor_lib.Tensor) -> tensor_lib.Tensor:
    if False:
        while True:
            i = 10
    'Append sharding information to a filename.\n\n  Args:\n    filename_tensor: A string tensor.\n    shard: Integer.  The shard for the filename.\n    num_shards: An int Tensor for the number of shards.\n\n  Returns:\n    A string tensor.\n  '
    return gen_io_ops.sharded_filename(filename_tensor, shard, num_shards)

def registered_saver_filename(filename_tensor: tensor_lib.Tensor, saver_name: registration.RegisteredSaver) -> tensor_lib.Tensor:
    if False:
        for i in range(10):
            print('nop')
    return string_ops.string_join([filename_tensor, constant_op.constant(f'-{saver_name}')])

def _get_mapped_registered_save_fn(fn: Callable[..., tensor_lib.Tensor], trackables: Sequence[base.Trackable], call_with_mapped_captures: MappedCapturesCallable) -> Callable[[tensor_lib.Tensor], MappedCapturesCallable]:
    if False:
        return 10
    'Converts the function to a python or tf.function with a single file arg.'

    def save_fn(file_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
        if False:
            while True:
                i = 10
        return fn(trackables=trackables, file_prefix=file_prefix)
    if call_with_mapped_captures is None:
        return save_fn
    else:
        tf_fn = def_function.function(save_fn, autograph=False)
        concrete = tf_fn.get_concrete_function(file_prefix=tensor_spec.TensorSpec(shape=(), dtype=dtypes.string))

        def save_fn_with_replaced_captures(file_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
            if False:
                i = 10
                return i + 15
            return call_with_mapped_captures(concrete, [file_prefix])
        return save_fn_with_replaced_captures

def _get_mapped_registered_restore_fn(fn: Callable[..., tensor_lib.Tensor], trackables: Sequence[base.Trackable], call_with_mapped_captures: MappedCapturesCallable) -> Callable[..., tensor_lib.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    'Converts the function to a python or tf.function with a single file arg.'

    def restore_fn(merged_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
        if False:
            print('Hello World!')
        return fn(trackables=trackables, merged_prefix=merged_prefix)
    if call_with_mapped_captures is None:
        return restore_fn
    else:
        tf_fn = def_function.function(restore_fn, autograph=False)
        concrete = tf_fn.get_concrete_function(merged_prefix=tensor_spec.TensorSpec(shape=(), dtype=dtypes.string))

        def restore_fn_with_replaced_captures(merged_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
            if False:
                return 10
            return call_with_mapped_captures(concrete, [merged_prefix])
        return restore_fn_with_replaced_captures
_restore_noop = lambda *args, **kwargs: None

class MultiDeviceSaver:
    """Saves checkpoints directly from multiple devices.

  Note that this is a low-level utility which stores Tensors in the keys
  specified by `SaveableObject`s. Higher-level utilities for object-based
  checkpointing are built on top of it.
  """

    def __init__(self, serialized_tensors: Mapping[base.Trackable, sharding_util.TensorSliceDict], registered_savers: 'RegisteredSaversDict | None'=None, call_with_mapped_captures: 'MappedCapturesCallable | None'=None):
        if False:
            while True:
                i = 10
        'Specify a list of `SaveableObject`s to save and restore.\n\n    Args:\n      serialized_tensors: A dictionary mapping `Trackable` to a tensor dict,\n        which maps checkpoint_key -> (slice_spec ->) -> Tensor/SaveSpec. The\n        `Trackable` key is used to get the `restore_from_tensors` function,\n        and may be `None` if the tensor is not meant to be restored.\n      registered_savers: A dictionary mapping `registration.RegisteredSaver`\n        namedtuples to a dictionary of named Trackables. The keys of the\n        Trackable dictionary are string names that uniquely identify the\n        Trackable in the checkpoint.\n      call_with_mapped_captures: TODO\n    '
        self._keys_to_restore_fn = {}
        self._restore_fn_to_keys = {}
        self._tensors_by_task = {}
        for (obj, tensor_dict) in serialized_tensors.items():
            restore_fn = _restore_noop if obj is None else obj._restore_from_tensors
            for (checkpoint_key, maybe_tensor) in tensor_dict.items():
                if not isinstance(maybe_tensor, dict):
                    maybe_tensor = {'': maybe_tensor}
                for (slice_spec, tensor) in maybe_tensor.items():
                    if (checkpoint_key, slice_spec) in self._keys_to_restore_fn:
                        raise ValueError('Recieved multiple tensors with the same checkpoint key and slice spec. This is invalid because one will overwrite the other in the checkpoint. This indicates a bug in the Checkpoint key-generation.')
                    self._keys_to_restore_fn[checkpoint_key, slice_spec] = restore_fn
                    self._restore_fn_to_keys.setdefault(restore_fn, []).append((checkpoint_key, slice_spec))
                    tensor_task = saveable_object_util.set_cpu0(tensor.device)
                    self._tensors_by_task.setdefault(tensor_task, {}).setdefault(checkpoint_key, {})[slice_spec] = tensor
        self._registered_savers = {}
        if registered_savers:
            for (registered_name, trackables) in registered_savers.items():
                save_fn = _get_mapped_registered_save_fn(registration.get_save_function(registered_name), trackables, call_with_mapped_captures)
                restore_fn = _get_mapped_registered_restore_fn(registration.get_restore_function(registered_name), trackables, call_with_mapped_captures)
                self._registered_savers[registered_name] = (save_fn, restore_fn)

    @classmethod
    def from_saveables(cls, saveables: Sequence[base.Trackable], registered_savers: 'RegisteredSaversDict | None'=None, call_with_mapped_captures: 'MappedCapturesCallable | None'=None) -> 'MultiDeviceSaver':
        if False:
            while True:
                i = 10
        'Constructs a MultiDeviceSaver from a list of `SaveableObject`s.'
        serialized_tensors = object_identity.ObjectIdentityDictionary()
        for saveable in saveables:
            trackable = saveable_object_util.SaveableCompatibilityConverter(saveable, saveables=[saveable])
            serialized_tensors[trackable] = trackable._serialize_to_tensors()
        return cls(serialized_tensors, registered_savers, call_with_mapped_captures)

    def to_proto(self) -> saver_pb2.SaverDef:
        if False:
            while True:
                i = 10
        'Serializes to a SaverDef referencing the current graph.'
        filename_tensor = array_ops.placeholder(shape=[], dtype=dtypes.string, name='saver_filename')
        save_tensor = self._traced_save(filename_tensor)
        restore_op = self._traced_restore(filename_tensor).op
        return saver_pb2.SaverDef(filename_tensor_name=filename_tensor.name, save_tensor_name=save_tensor.name, restore_op_name=restore_op.name, version=saver_pb2.SaverDef.V2)

    @def_function.function(input_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.string),), autograph=False)
    def _traced_save(self, file_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
        if False:
            i = 10
            return i + 15
        save_op = self.save(file_prefix)
        with ops.device('cpu:0'):
            with ops.control_dependencies([save_op]):
                return array_ops.identity(file_prefix)

    @def_function.function(input_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.string),), autograph=False)
    def _traced_restore(self, file_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
        if False:
            i = 10
            return i + 15
        restore_ops = self.restore(file_prefix)
        with ops.device('cpu:0'):
            with ops.control_dependencies(restore_ops.values()):
                return array_ops.identity(file_prefix)

    def save(self, file_prefix: tensor_lib.Tensor, options: 'checkpoint_options.CheckpointOptions | None'=None) -> ops.Operation:
        if False:
            return 10
        'Save the saveable objects to a checkpoint with `file_prefix`.\n\n    Args:\n      file_prefix: A string or scalar string Tensor containing the prefix to\n        save under.\n      options: Optional `CheckpointOptions` object.\n    Returns:\n      An `Operation`, or None when executing eagerly.\n    '
        options = options or checkpoint_options.CheckpointOptions()
        with ops.device('CPU'):
            sharded_suffix = array_ops.where(string_ops.regex_full_match(file_prefix, '^s3://.*'), constant_op.constant('.part'), constant_op.constant('_temp/part'))
            tmp_checkpoint_prefix = string_ops.string_join([file_prefix, sharded_suffix])
            registered_paths = {saver_name: registered_saver_filename(file_prefix, saver_name) for saver_name in self._registered_savers}

        def save_fn() -> ops.Operation:
            if False:
                i = 10
                return i + 15
            saved_prefixes = []
            for (saver_name, (save_fn, _)) in self._registered_savers.items():
                maybe_saved_prefixes = save_fn(registered_paths[saver_name])
                if maybe_saved_prefixes is not None:
                    flattened_saved_prefixes = nest.flatten(maybe_saved_prefixes)
                    if not all((tensor_util.is_tf_type(x) and x.dtype == dtypes.string for x in flattened_saved_prefixes)):
                        raise ValueError(f'Registered saver must return a (maybe empty) list of string type tensors. Got {maybe_saved_prefixes}.')
                    saved_prefixes.extend(flattened_saved_prefixes)
            num_shards = len(self._tensors_by_task)
            sharded_saves = []
            num_shards_tensor = constant_op.constant(num_shards, name='num_shards')
            last_device_spec = None
            for (shard, (device_spec, tensor_slice_dict)) in enumerate(sorted(self._tensors_by_task.items())):
                last_device_spec = device_spec
                with ops.device(saveable_object_util.set_cpu0(device_spec)):
                    shard_prefix = sharded_filename(tmp_checkpoint_prefix, shard, num_shards_tensor)
                saved_prefixes.append(shard_prefix)
                with ops.device(device_spec):
                    sharded_saves.append(_single_task_save(shard_prefix, tensor_slice_dict, options))
            with ops.control_dependencies(sharded_saves):
                merge_device_spec = options.experimental_io_device or saveable_object_util.set_cpu0(last_device_spec)
                with ops.device(merge_device_spec):
                    return gen_io_ops.merge_v2_checkpoints(saved_prefixes, file_prefix, delete_old_dirs=True)
        if context.executing_eagerly() and len(self._tensors_by_task) > 1:

            @def_function.function(jit_compile=False)
            def tf_function_save() -> None:
                if False:
                    while True:
                        i = 10
                save_fn()
            tf_function_save()
        else:
            return save_fn()

    def restore(self, file_prefix: tensor_lib.Tensor, options: 'checkpoint_options.CheckpointOptions | None'=None) -> Mapping[str, ops.Operation]:
        if False:
            while True:
                i = 10
        'Restore the saveable objects from a checkpoint with `file_prefix`.\n\n    Args:\n      file_prefix: A string or scalar string Tensor containing the prefix for\n        files to read from.\n      options: Optional `CheckpointOptions` object.\n\n    Returns:\n      When not run eagerly or when saving on a single device, returns a\n      dictionary mapping from SaveableObject names to restore operations;\n      otherwise, returns an empty dict.\n    '
        options = options or checkpoint_options.CheckpointOptions()

        def restore_fn() -> Mapping[str, ops.Operation]:
            if False:
                return 10
            restore_fn_inputs = {}
            restore_fn_input_count = {fn: len(keys) for (fn, keys) in self._restore_fn_to_keys.items()}
            restore_ops = {}
            for (device_spec, tensor_slice_dict) in sorted(self._tensors_by_task.items()):
                with ops.device(device_spec):
                    restored_tensor_dict = _single_task_restore(file_prefix, tensor_slice_dict, options)
                    for (checkpoint_key, slice_and_tensor) in restored_tensor_dict.items():
                        for (slice_spec, tensor) in slice_and_tensor.items():
                            restore_fn = self._keys_to_restore_fn[checkpoint_key, slice_spec]
                            if slice_spec:
                                restore_fn_inputs.setdefault(restore_fn, {}).setdefault(checkpoint_key, {})[slice_spec] = tensor
                            else:
                                restore_fn_inputs.setdefault(restore_fn, {})[checkpoint_key] = tensor
                            restore_fn_input_count[restore_fn] -= 1
                            if restore_fn_input_count[restore_fn] == 0:
                                restored_tensors = {}
                                for (ckpt_key, tensor) in restore_fn_inputs[restore_fn].items():
                                    restored_tensors[trackable_utils.extract_local_name(ckpt_key)] = tensor
                                ret = restore_fn(restored_tensors)
                                if isinstance(ret, dict):
                                    restore_ops.update(ret)
            for (_, (_, restore_fn)) in self._registered_savers.items():
                restore_fn(file_prefix)
            return restore_ops
        has_custom_device_saver = any([context.is_custom_device(ds) for ds in self._tensors_by_task])
        if context.executing_eagerly() and (len(self._tensors_by_task) > 1 or has_custom_device_saver):

            @def_function.function(jit_compile=False, autograph=False)
            def tf_function_restore() -> Mapping[str, ops.Operation]:
                if False:
                    while True:
                        i = 10
                restore_fn()
                return {}
            restore_ops = tf_function_restore()
        else:
            restore_ops = restore_fn()
        return restore_ops