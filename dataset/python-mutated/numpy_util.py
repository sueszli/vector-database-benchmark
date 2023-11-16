"""Utilities to convert data buffers to/from DTensor tensors."""
from typing import List
import numpy as np
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.types.core import Tensor, TensorLike

def _split(value, splits, axis=0, split_fn=np.split, stack_fn=np.stack):
    if False:
        while True:
            i = 10
    'Split `value` into a sharded nparray/tf tensor based on the number of splits.\n  '
    children = split_fn(value, splits[0], axis=axis)
    if len(splits) > 1:
        splits = splits[1:]
        children = [_split(child, splits, axis + 1) for child in children]
    return stack_fn(children)

def to_numpy(tensor: TensorLike) -> np.ndarray:
    if False:
        return 10
    'Copy `input` DTensor to an equivalent local numpy array.'
    layout = api.fetch_layout(tensor)
    if layout.mesh.is_remote():
        return np.array([None])
    unpacked = [tensor.numpy() for tensor in api.unpack(tensor)]
    return unpacked_to_numpy(unpacked, layout)

def unpacked_to_numpy(unpacked: List[TensorLike], layout: layout_lib.Layout) -> np.ndarray:
    if False:
        print('Hello World!')
    'Heals local Tensor components to a numpy array.'
    if len(unpacked) != len(layout.offset_to_shard()):
        raise ValueError('Wrong number of component Tensors.')
    unravelled = np.ndarray([layout.num_shards(i) for i in range(layout.rank)], dtype=object)
    for (offset, loc) in enumerate(layout.offset_to_shard()):
        unravelled[loc] = unpacked[offset]
    concat_tensor = np.block(unravelled.tolist())
    while concat_tensor.ndim > unpacked[0].ndim:
        concat_tensor = np.squeeze(concat_tensor, axis=0)
    return concat_tensor

def unpack(t: TensorLike, layout: layout_lib.Layout, split_fn=np.split, stack_fn=np.stack) -> List[TensorLike]:
    if False:
        return 10
    'Slice `t` into a flattened list of tensors suitable for `pack`.'
    if not layout.rank:
        return [t] * layout.mesh.size
    sharded_tensor = _split(t, [layout.num_shards(i) for i in range(layout.rank)], split_fn=split_fn, stack_fn=stack_fn)
    flattened = [np.ndarray([])] * layout.mesh.size
    for (offset, shard) in enumerate(layout.offset_to_shard()):
        flattened[offset] = sharded_tensor[tuple(shard)]
    return flattened

def pack_numpy(value: np.ndarray, layout: layout_lib.Layout, make_sparse: bool=False) -> Tensor:
    if False:
        i = 10
        return i + 15
    assert value is not None
    unpacked = unpack(value, layout)
    if make_sparse:
        return api.pack([sparse_ops.from_dense(t) for t in unpacked], layout)
    return api.pack(unpacked, layout)

def pack_tf_tensor(value: Tensor, layout: layout_lib.Layout) -> Tensor:
    if False:
        return 10
    if value is None:
        raise ValueError('pack requires values to be passed in')
    unpacked = unpack(value, layout, split_fn=array_ops.split, stack_fn=array_ops_stack.stack)
    return api.pack(unpacked, layout)

@polymorphic_function.function
def stateless_random_uniform(shape, seed, layout):
    if False:
        return 10
    'Creates uniform random tensor with the given layout.'
    return api.relayout(stateless_random_ops.stateless_random_uniform(shape=shape, seed=seed), layout=layout)