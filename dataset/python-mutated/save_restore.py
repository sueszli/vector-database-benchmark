"""Contains functionaility for Checkpoint/SavedModel in DTensor."""
import collections
from typing import Dict, List, Union
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.eager import context
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import tf_export

@tf_export('experimental.dtensor.sharded_save', v1=[])
def sharded_save(mesh: layout_lib.Mesh, file_prefix: Union[str, tensor_lib.Tensor], tensor_names: Union[List[str], tensor_lib.Tensor], shape_and_slices: Union[List[str], tensor_lib.Tensor], tensors: List[Union[tensor_lib.Tensor, tf_variables.Variable]]):
    if False:
        i = 10
        return i + 15
    'Saves given named tensor slices in a sharded, multi-client safe fashion.\n\n  The method makes sure the checkpoint directory state is correct in a sharded\n  mutli-client saving. Namely, we place a barrier after SaveV2 to make sure\n  every client has done writing the files. And another one after\n  MergeV2Checkpoints to make sure all Metadata is properly merged.\n\n  Upon existing, the checkpoint is completed and the all directory operations\n  are done.\n\n  Args:\n    mesh: The Mesh that contains the Tensors to save.\n    file_prefix: The prefix of checkpoint.\n    tensor_names: a list of tensor names used in save op.\n    shape_and_slices: a list of shape and slice specification used in save op.\n      The only supported value is "" as we don\'t support distributed saving with\n      slices yet.\n    tensors: a list of tensors used in save op. The order should match\n      tensor_names.\n\n  Returns:\n    A MergeV2Checkpoints op that merged all Metadata.\n  '
    with ops.device(api.device_name()):
        io_ops.save_v2(file_prefix, tensor_names, shape_and_slices, tensors)
    mesh_util.barrier(mesh.host_mesh(), 'SaveV2')
    with api.default_mesh(mesh.host_mesh()):
        merge_op = io_ops.MergeV2Checkpoints(checkpoint_prefixes=[file_prefix], destination_prefix=file_prefix, delete_old_dirs=True)
    mesh_util.barrier(mesh.host_mesh(), 'MergeV2Checkpoints')
    return merge_op

@tf_export('experimental.dtensor.enable_save_as_bf16', v1=[])
def enable_save_as_bf16(variables: List[tf_variables.Variable]):
    if False:
        i = 10
        return i + 15
    'Allows float32 DVariables to be checkpointed and restored as bfloat16.\n\n  The method only affects the DVariable part inside the model and leaves\n  non-DTensor Variables/Tensors untouched.\n\n  Args:\n    variables: A list of tf.Variable to be enabled with bfloat16 save/restore.\n      Only has effect on DTensor Variables as they go through d_variables with\n      DTensor Specific logis.\n  '
    for v in variables:
        if isinstance(v, d_variable.DVariable):
            v.save_as_bf16 = True

@tf_export('experimental.dtensor.name_based_restore', v1=[])
def name_based_restore(mesh: layout_lib.Mesh, checkpoint_prefix: str, name_tensor_dict: Dict[str, Union[tensor_lib.Tensor, tf_variables.Variable]]):
    if False:
        while True:
            i = 10
    'Restores from checkpoint_prefix to name based DTensors.\n\n  It is required to have already-initialized DTensor variables that have same\n  shape/dtype for the tensors being restored.\n\n  Also, we currently only support a named based restore on a single mesh.\n\n  Args:\n    mesh: The single mesh that all Tensors would be restored to.\n    checkpoint_prefix : The prefix of checkpoint to be restored.\n    name_tensor_dict: A ordered dictionary of tensor_names to a DTensor. The\n      DTensor shape/dtype must match the tensors being saved/restored for now.\n\n  Returns:\n    A dictionary of name to its restored DTensor value.\n  '
    if not context.executing_eagerly():
        raise ValueError('name based restore must run eagerly.')
    ordered_name_tensor_dict = name_tensor_dict
    if not isinstance(name_tensor_dict, collections.OrderedDict):
        ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)
    for (name, tensor) in ordered_name_tensor_dict.items():
        try:
            if api.fetch_layout(tensor).mesh.device_type().upper() != 'CPU':
                raise ValueError('Restoring a non CPU Tensor is not supported currently. Offending tensor name : {tensor_name}'.format(tensor_name=name))
        except errors_impl.OpError as op_error:
            raise ValueError('Saving/Restoring tensor must be a DTensor') from op_error
    checkpoint_prefix = api.pack([checkpoint_prefix] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=0))
    tensor_names = api.pack([list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
    shape_and_slices = api.pack([[''] * len(ordered_name_tensor_dict)] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
    input_shapes = [tensor.shape for tensor in ordered_name_tensor_dict.values()]
    input_layouts = [api.fetch_layout(tensor).to_string() for tensor in ordered_name_tensor_dict.values()]
    with ops.device(api.device_name()):
        restored_cpu_tensors = gen_dtensor_ops.d_tensor_restore_v2(prefix=checkpoint_prefix, tensor_names=tensor_names, shape_and_slices=shape_and_slices, input_shapes=input_shapes, input_layouts=input_layouts, dtypes=[tensor.dtype for tensor in ordered_name_tensor_dict.values()])
    return collections.OrderedDict(zip(ordered_name_tensor_dict.keys(), restored_cpu_tensors))

@tf_export('experimental.dtensor.name_based_save', v1=[])
def name_based_save(mesh: layout_lib.Mesh, checkpoint_prefix: Union[str, tensor_lib.Tensor], name_tensor_dict: Dict[str, Union[tensor_lib.Tensor, tf_variables.Variable]]):
    if False:
        for i in range(10):
            print('nop')
    'Saves name based Tensor into a Checkpoint.\n\n  The function prepares the input dictionary to the format of a `sharded_save`,\n  so that it can take advantage of DTensor SPMD based distributed save.\n\n  Same as restore, the function only supports saving on the single mesh.\n\n  Args:\n    mesh: The single mesh that all Tensors would be restored to.\n    checkpoint_prefix : The prefix of checkpoint to be restored.\n    name_tensor_dict: A ordered dictionary of tensor_names to a DTensor. The\n      DTensor shape/dtype must match the tensors being saved/restored for now.\n  '
    if not context.executing_eagerly():
        raise ValueError('name based save must run eagerly.')
    ordered_name_tensor_dict = name_tensor_dict
    if not isinstance(name_tensor_dict, collections.OrderedDict):
        ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)
    checkpoint_prefix = api.pack([checkpoint_prefix] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=0))
    tensor_names = api.pack([list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
    sharded_save(mesh, file_prefix=checkpoint_prefix, tensor_names=tensor_names, shape_and_slices=[''] * len(ordered_name_tensor_dict), tensors=list(ordered_name_tensor_dict.values()))