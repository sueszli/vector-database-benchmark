"""Checkpoint policies that determine how tensors are split into shards."""
from typing import Sequence
from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.training.saving import saveable_object_util

class ShardByDevicePolicy(sharding_util.ShardingCallback):
    """Policy that splits tensors into shards based on their device spec."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(self._device_callback_impl, 'Split tensors into shards based on their device spec.')

    def _device_callback_impl(self, shardable_tensors: Sequence[sharding_util.ShardableTensor]) -> Sequence[sharding_util.TensorSlice]:
        if False:
            i = 10
            return i + 15
        'Callback to split tensors into shards based on their device spec.\n\n    Args:\n      shardable_tensors: A list of ShardableTensors.\n\n    Returns:\n      List of shard dicts containing tensors.\n        [ {checkpoint key: {slice_spec: tensor} } ]\n    '
        tensors_by_device = {}
        for shardable_tensor in shardable_tensors:
            tensor = shardable_tensor.tensor
            checkpoint_key = shardable_tensor.checkpoint_key
            slice_spec = shardable_tensor.slice_spec
            device = saveable_object_util.set_cpu0(shardable_tensor.device)
            tensors_by_device.setdefault(device, {}).setdefault(checkpoint_key, {})[slice_spec] = tensor
        return list(tensors_by_device.values())

    def __call__(self, shardable_tensors: Sequence[sharding_util.ShardableTensor]) -> Sequence[sharding_util.TensorSlice]:
        if False:
            return 10
        return self.callback(shardable_tensors)