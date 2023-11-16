"""Helper library for sharding during TPU compilation."""
from tensorflow.python.framework import tensor_shape
_DEFAULT_NUMBER_OF_SHARDS = 1
_DEFAULT_SHARD_DIMENSION = 0

class ShardingPolicy(object):
    """An object use to hold the sharding policy for a Tensor."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._number_of_shards = None
        self._number_of_partitions = 1
        self._shard_dimension = None
        self._frozen = False

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if self.number_of_shards is None or self.shard_dimension is None:
            return 'ShardingPolicy(unset)'
        else:
            return 'ShardingPolicy(%d shards dimension %d)' % (self.number_of_shards, self.shard_dimension)

    def _fill_default_values(self):
        if False:
            for i in range(10):
                print('nop')
        if self._number_of_shards is None:
            self._number_of_shards = _DEFAULT_NUMBER_OF_SHARDS
        if self._shard_dimension is None:
            self._shard_dimension = tensor_shape.as_dimension(_DEFAULT_SHARD_DIMENSION)

    def freeze(self):
        if False:
            return 10
        'Prevents further modification to the sharding policy.\n\n    Any values that have not been set when freeze is called are set to\n    defaults. If the ShardingPolicy is already frozen, this is a NoOp.\n    '
        if not self._frozen:
            self._fill_default_values()
            self._frozen = True

    @property
    def number_of_shards(self):
        if False:
            return 10
        'Returns the number of shards in the policy or None if unspecified.'
        return self._number_of_shards

    def set_number_of_shards(self, number_of_shards):
        if False:
            i = 10
            return i + 15
        'Sets the number of shards for the current policy.\n\n    If the policy has been frozen then number_of_shards must match the\n    existing setting.\n\n    Args:\n      number_of_shards: The number of shards to use in the policy.\n\n    Raises:\n      ValueError: If the policy has been frozen and number_of_shards\n        differs from the frozen value; or number_of_shards <= 0.\n    '
        if self._frozen:
            if self._number_of_shards != number_of_shards:
                raise ValueError(f"Can't set sharding policy to use {number_of_shards} shards since it has been frozen to use {self._number_of_shards}")
        elif number_of_shards > 0:
            self._number_of_shards = number_of_shards
        else:
            raise ValueError(f"Can't set sharding policy to use {number_of_shards} shards; value must be > 0")

    @property
    def number_of_partitions(self):
        if False:
            print('Hello World!')
        'Returns the number of partitions of the policy or None if unspecified.'
        return self._number_of_partitions

    def set_number_of_partitions(self, number_of_partitions):
        if False:
            print('Hello World!')
        'Sets the number of partitions for the current policy.\n\n    If the policy has been frozen then shard_dimension must match the\n    existing setting.\n\n    Args:\n      number_of_partitions: The number of partitions to use in the policy.\n\n    Raises:\n      ValueError: If the policy has been frozen and shard_dimension\n        differs from the frozen value.\n    '
        if self._frozen:
            if self._number_of_partitions != number_of_partitions:
                raise ValueError(f"Can't set number_of_partitions to {number_of_partitions} since it has been frozen to use {self._number_of_partitions}.")
        else:
            self._number_of_partitions = number_of_partitions

    @property
    def shard_dimension(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the shard dimension of the policy or None if unspecified.'
        return self._shard_dimension

    def set_shard_dimension(self, shard_dimension):
        if False:
            while True:
                i = 10
        "Sets the shard dimension for the current policy.\n\n    If the policy has been frozen then shard_dimension must match the\n    existing setting.\n\n    Args:\n      shard_dimension: The shard dimension to use in the policy.\n\n    Raises:\n      ValueError: If the policy has been frozen and shard_dimension\n        differs from the frozen value, or shard_dimension can't be\n        interpreted as a Dimension.\n    "
        if self._frozen:
            if self._shard_dimension != shard_dimension:
                raise ValueError("Can't set shard dimension to %d since it has been frozen to use %d." % (shard_dimension, self._shard_dimension))
        else:
            self._shard_dimension = tensor_shape.as_dimension(shard_dimension)

    def merge(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Merges the policy of another policy into the current policy.\n\n    Args:\n      other: The policy to merge into this one.\n\n    Raises:\n      ValueError: If this policy has been frozen and the merge conflicts with\n      the frozen policy.\n    '
        if other.number_of_shards is not None:
            self.set_number_of_shards(other.number_of_shards)
        if other.shard_dimension is not None:
            self.set_shard_dimension(other.shard_dimension)

    def get_unpartitioned_shape(self, shape):
        if False:
            for i in range(10):
                print('nop')
        "Returns the shape of an unpartitioned Tensor.\n\n    When given the shape of a 'sharded-size' Tensor, returns the shape\n    of the full shape of its unpartitioned Tensor.\n\n    Args:\n      shape: The shape of the sharded Tensor.\n\n    Returns:\n      The shape of the unpartitioned version of the Tensor.\n\n    Raises:\n      ValueError: if shape has unknown sharded dimension\n    "
        shape = tensor_shape.as_shape(shape)
        dims = shape.as_list()
        if self._shard_dimension is None or self._number_of_partitions is None or (not dims):
            return None
        if dims[self._shard_dimension] is None:
            raise ValueError(f'Shape {shape.as_list()} must have a fixed size for dimension {self._shard_dimension} that is known. ')
        if self._number_of_partitions > 1:
            dims[self._shard_dimension] *= self._number_of_partitions
        return tensor_shape.as_shape(dims)

    def get_sharded_shape(self, shape, shard_index=None):
        if False:
            return 10
        "Returns the shape of a shard of a full Tensor.\n\n    When given the shape of a 'full-size' Tensor, returns the shape of\n    the sub-Tensor after it has been sharded. Freezes the policy if it\n    has not yet been frozen.\n\n    Args:\n      shape: The shape of the full-size Tensor to be sharded.\n      shard_index: The index of the shard whose shape should be returned.\n        shard_index can be None for sharding policies that use the same shape\n        for every shard.\n\n    Returns:\n      The shape of the sharded version of the Tensor.\n\n    Raises:\n      ValueError: If shard_index is None when shards are of different\n        shapes; or shard_index is not None and\n        !(0<=shard_index<number_of_shards); or shape does not have at\n        least self.shard_dimension+1 dimensions; or the value of\n        shape's shard dimension is not a multiple of\n        self.number_of_shards\n    "
        if self._shard_dimension is None or self._number_of_shards is None:
            return None
        if shard_index is not None:
            if shard_index < 0 or shard_index >= self.number_of_shards:
                raise ValueError(f'Requested shard_index {shard_index}, but shard_index must be in [0,{self._number_of_shards}).')
        shape = tensor_shape.as_shape(shape)
        if self._number_of_shards == 1:
            return shape
        ndims = shape.ndims
        if ndims is None:
            raise ValueError(f'Shape {shape} must be a known shape.')
        if ndims <= self._shard_dimension:
            raise ValueError(f'Shape {shape.as_list()} does not contain shard_dimension {self._shard_dimension}')
        dims = shape.as_list()
        if dims[self._shard_dimension] is None:
            raise ValueError(f'Shape {shape.as_list()} must have a fixed size for dimension {self._shard_dimension} that is known at construction time.')
        if dims[self._shard_dimension] % self._number_of_shards != 0:
            raise ValueError(f'Shape {shape.as_list()} cannot be sharded {self._number_of_shards} ways along dimension {self._shard_dimension}')
        dims[self._shard_dimension] //= self._number_of_shards
        return tensor_shape.TensorShape(dims)

    def _unshard_shape(self, shape):
        if False:
            return 10
        'Return the unsharded shape that would generate a given sharded shape.\n\n    Args:\n      shape: the sharded shape to unshard\n\n    Returns:\n      The unsharded shape.\n\n    Raises:\n      ValueError: if shape is unknown or does not contain\n        self.shard_dimension\n      TypeError: if shape is not convertible to a TensorShape\n    '
        shape = tensor_shape.as_shape(shape)
        if self._number_of_shards == 1:
            return shape
        ndims = shape.ndims
        if ndims is None:
            raise ValueError(f'Shape {shape} must be statically known.')
        if ndims <= self._shard_dimension:
            raise ValueError(f'Shape {shape.as_list()} does not contain shard_dimension {self._shard_dimension}. Rank is too small.')
        dims = shape.as_list()
        dims[self._shard_dimension] *= self._number_of_shards
        return tensor_shape.TensorShape(dims)

    def get_unsharded_shape(self, shapes):
        if False:
            while True:
                i = 10
        'Returns the shape of an unsharded Tensor given a list of shards.\n\n    When given a list of shapes of shards, returns the shape of the\n    unsharded Tensor that would generate the shards. Sets defaults for the\n    policy if number_of_shards or shard_dimension is None.\n\n    Args:\n      shapes: The shapes of the Tensor shards to be combined.\n\n    Returns:\n      The shape of the unsharded version of the Tensor.\n\n    Raises:\n      ValueError: if shapes is not a list of length\n        self.number_of_shards; or any element of shapes is not a valid\n        shape consistent with the sharding policy; or the list of\n        shapes is not a valid sharding of a full shape.\n      TypeError: if an element of shapes is not convertible to a\n        TensorShape\n    '
        self._fill_default_values()
        if len(shapes) != self.number_of_shards:
            raise ValueError(f'Shapes {shapes} is length {len(shapes)} but must be a list of length number_of_shards={self.number_of_shards}')
        unsharded_shapes = [self._unshard_shape(s) for s in shapes]
        for i in range(self.number_of_shards - 1):
            if not unsharded_shapes[i].is_compatible_with(unsharded_shapes[self.number_of_shards - 1]):
                raise ValueError(f'Sharded shapes {shapes} are not consistent shards of a full shape sharded {self.number_of_shards} ways along dimension {self.shard_dimension}.')
        return unsharded_shapes[0]