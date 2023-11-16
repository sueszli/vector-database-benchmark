"""Utils for Sparsecore Checkpoints."""
import functools
from typing import Any, Dict
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework.constant_op import constant as tf_constant
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.trackable import base as trackable_base
SPARSECORE_LAYOUTS_CHECKPOINT_KEY = '_sparse_core_table_layouts'

def unshuffle_from_sc_to_cpu(t: tensor.Tensor, num_sparse_cores: int, offset_in_shard: int, size_in_shard: int, shard_rotation: int=0) -> tensor.Tensor:
    if False:
        return 10
    "Unshuffles the sparse core sharded embedding tables to unsharded.\n\n  This converts an input tensor respresenting stacked and sharded embedding\n  table into a specific embedding table variable by using the provided\n  metadata about the said table within the stacked, sharded embedding table.\n  Args:\n    t: The input stacked and sharded embedding table from sparsecore.\n    num_sparse_cores: The number of sparsecores, this determines the number of\n      shards that are present in the input t.\n    offset_in_shard: Offset within a shard where the queried table starts.\n    size_in_shard: size (number of rows) of this queried table within each shard\n      of the input t.\n    shard_rotation: The rotation of this table's shards.\n\n  Returns:\n    An embedding table which is part of the stacked embedding table t.\n  "
    old_shape = t.shape
    if t.shape[0] % num_sparse_cores != 0:
        raise ValueError('The dim of table ({}) should be multiple of number of sparse cores ({})'.format(t.shape[1], num_sparse_cores))
    shards_t = array_ops.reshape(t, (num_sparse_cores, t.shape[0] // num_sparse_cores, t.shape[1]))
    shards = shards_t[:, offset_in_shard:offset_in_shard + size_in_shard, :]
    shards = manip_ops.roll(shards, -shard_rotation, axis=0)
    intermediate_tensor = array_ops.transpose(shards, (1, 0, 2))
    new_shape = (size_in_shard * num_sparse_cores, old_shape[1])
    return array_ops.reshape(intermediate_tensor, new_shape)

def remove_padding_from_sc(value_in_checkpoint: tensor.Tensor, variable_shape: tuple[int, int]) -> tensor.Tensor:
    if False:
        print('Hello World!')
    'Removes padding, if any, from sparsecore checkpoint.\n\n  Args:\n    value_in_checkpoint: input tensor value, usually from checkpoint.\n    variable_shape: Expected shape of tensor after removing padding.\n\n  Returns:\n    A slice of the input tensor to match the variable_shape if the\n    variable shape is a valid slice if the input tensor.\n  '
    checkpoint_value_shape = value_in_checkpoint.shape.as_list()
    is_init_value_padded = all([i >= j for (i, j) in zip(checkpoint_value_shape, variable_shape)])
    if not is_init_value_padded:
        return value_in_checkpoint
    begin = [0] * len(checkpoint_value_shape)
    return array_ops.slice(value_in_checkpoint, begin=begin, size=variable_shape)

def map_indices_in_shard(num_sparse_cores: int, offset_in_shard: int, shard_rotation: int, row_indices: tensor.Tensor) -> tuple[tensor.Tensor, tensor.Tensor]:
    if False:
        i = 10
        return i + 15
    "Maps a row of a given table to its sparse core shard and position.\n\n  Maps a given a row index of a logical table and its layout in sparse core,\n  returns the index of the shard where the row is placed and its relative\n  position within\n  that sparse core shard.\n  Args:\n    num_sparse_cores: The number of sparsecores, this determines the number of\n      shards present.\n    offset_in_shard: Offset within a shard where the queried table starts.\n    shard_rotation: The rotation of this table's shards.\n    row_indices: row indices of the embedding table being looked up.\n\n  Returns:\n    A Tuple representing shard_index and position of the row in that shard.\n  "
    shard_index = (row_indices % num_sparse_cores + shard_rotation) % num_sparse_cores
    position_in_shard = offset_in_shard + row_indices // num_sparse_cores
    return (shard_index, position_in_shard)

class SparseCoreLayoutsTrackable(trackable_base.Trackable):
    """Trackable for sparsecore layouts used in training."""

    def __init__(self, proto_str_tensor: tensor.Tensor):
        if False:
            while True:
                i = 10
        self.value = proto_str_tensor

    def _serialize_to_tensors(self) -> Dict[str, tensor.Tensor]:
        if False:
            return 10
        return {trackable_base.VARIABLE_VALUE_KEY: self.value}

    def _restore_from_tensors(self, restored_tensors: Dict[str, tensor.Tensor]) -> None:
        if False:
            i = 10
            return i + 15
        self.value = restored_tensors[trackable_base.VARIABLE_VALUE_KEY]

class SparseCoreStackedTableTrackable(trackable_base.Trackable):
    """Trackable for stacked tables generated from sparse core."""

    def __init__(self, stacked_layouts, table_to_config):
        if False:
            return 10
        self.vars = {}
        self._stacked_layouts = stacked_layouts
        for table_layout in stacked_layouts:
            variable_shape = tuple(table_layout.unsharded_shape)
            self.vars[table_layout.table_name] = tf_variables.Variable(name=table_layout.table_name, initial_value=functools.partial(table_to_config[table_layout.table_name].initializer, variable_shape, dtype=dtypes.float32), shape=variable_shape, dtype=dtypes.float32)

    def _serialize_to_tensors(self) -> Any:
        if False:
            while True:
                i = 10
        return {trackable_base.VARIABLE_VALUE_KEY: tf_constant(0.0, dtype=dtypes.float32)}

    def _restore_from_tensors(self, restored_tensors: Dict[str, tensor.Tensor]):
        if False:
            i = 10
            return i + 15

        def fn(restored_tensors):
            if False:
                print('Hello World!')
            value_from_checkpoint = restored_tensors[trackable_base.VARIABLE_VALUE_KEY]
            for layout in self._stacked_layouts:
                variable_shape = (layout.unsharded_shape[0], layout.unsharded_shape[1])
                t_part = unshuffle_from_sc_to_cpu(t=value_from_checkpoint, num_sparse_cores=layout.num_sparse_cores, offset_in_shard=layout.sparse_core_shard_row_offset, size_in_shard=layout.unsharded_padded_shape[0] // layout.num_sparse_cores, shard_rotation=layout.sparse_core_shard_rotation)
                t_part = remove_padding_from_sc(t_part, variable_shape)
                self.vars[layout.table_name].assign(t_part)
        return fn(restored_tensors)

    def get_var(self, name: str) -> tf_variables.Variable:
        if False:
            for i in range(10):
                print('nop')
        return self.vars[name]

    def get_vars(self) -> Dict[str, tf_variables.Variable]:
        if False:
            while True:
                i = 10
        return self.vars

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'SparseCoreStackedTableTrackable({})'.format(self.vars.keys())