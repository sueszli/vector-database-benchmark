from typing import List
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import TensorType, TensorStructType

@PublicAPI
class RepeatedValues:
    """Represents a variable-length list of items from spaces.Repeated.

    RepeatedValues are created when you use spaces.Repeated, and are
    accessible as part of input_dict["obs"] in ModelV2 forward functions.

    Example:
        Suppose the gym space definition was:
            Repeated(Repeated(Box(K), N), M)

        Then in the model forward function, input_dict["obs"] is of type:
            RepeatedValues(RepeatedValues(<Tensor shape=(B, M, N, K)>))

        The tensor is accessible via:
            input_dict["obs"].values.values

        And the actual data lengths via:
            # outer repetition, shape [B], range [0, M]
            input_dict["obs"].lengths
                -and-
            # inner repetition, shape [B, M], range [0, N]
            input_dict["obs"].values.lengths

    Attributes:
        values: The padded data tensor of shape [B, max_len, ..., sz],
            where B is the batch dimension, max_len is the max length of this
            list, followed by any number of sub list max lens, followed by the
            actual data size.
        lengths (List[int]): Tensor of shape [B, ...] that represents the
            number of valid items in each list. When the list is nested within
            other lists, there will be extra dimensions for the parent list
            max lens.
        max_len: The max number of items allowed in each list.

    TODO(ekl): support conversion to tf.RaggedTensor.
    """

    def __init__(self, values: TensorType, lengths: List[int], max_len: int):
        if False:
            i = 10
            return i + 15
        self.values = values
        self.lengths = lengths
        self.max_len = max_len
        self._unbatched_repr = None

    def unbatch_all(self) -> List[List[TensorType]]:
        if False:
            for i in range(10):
                print('nop')
        'Unbatch both the repeat and batch dimensions into Python lists.\n\n        This is only supported in PyTorch / TF eager mode.\n\n        This lets you view the data unbatched in its original form, but is\n        not efficient for processing.\n\n        .. testcode::\n            :skipif: True\n\n            batch = RepeatedValues(<Tensor shape=(B, N, K)>)\n            items = batch.unbatch_all()\n            print(len(items) == B)\n\n        .. testoutput::\n\n            True\n\n        .. testcode::\n            :skipif: True\n\n            print(max(len(x) for x in items) <= N)\n\n        .. testoutput::\n\n            True\n\n        .. testcode::\n            :skipif: True\n\n            print(items)\n\n        .. testoutput::\n\n            [[<Tensor_1 shape=(K)>, ..., <Tensor_N, shape=(K)>],\n             ...\n             [<Tensor_1 shape=(K)>, <Tensor_2 shape=(K)>],\n             ...\n             [<Tensor_1 shape=(K)>],\n             ...\n             [<Tensor_1 shape=(K)>, ..., <Tensor_N shape=(K)>]]\n        '
        if self._unbatched_repr is None:
            B = _get_batch_dim_helper(self.values)
            if B is None:
                raise ValueError('Cannot call unbatch_all() when batch_dim is unknown. This is probably because you are using TF graph mode.')
            else:
                B = int(B)
            slices = self.unbatch_repeat_dim()
            result = []
            for i in range(B):
                if hasattr(self.lengths[i], 'item'):
                    dynamic_len = int(self.lengths[i].item())
                else:
                    dynamic_len = int(self.lengths[i].numpy())
                dynamic_slice = []
                for j in range(dynamic_len):
                    dynamic_slice.append(_batch_index_helper(slices, i, j))
                result.append(dynamic_slice)
            self._unbatched_repr = result
        return self._unbatched_repr

    def unbatch_repeat_dim(self) -> List[TensorType]:
        if False:
            for i in range(10):
                print('nop')
        'Unbatches the repeat dimension (the one `max_len` in size).\n\n        This removes the repeat dimension. The result will be a Python list of\n        with length `self.max_len`. Note that the data is still padded.\n\n        .. testcode::\n            :skipif: True\n\n            batch = RepeatedValues(<Tensor shape=(B, N, K)>)\n            items = batch.unbatch()\n            len(items) == batch.max_len\n\n        .. testoutput::\n\n            True\n\n        .. testcode::\n            :skipif: True\n\n            print(items)\n\n        .. testoutput::\n\n            [<Tensor_1 shape=(B, K)>, ..., <Tensor_N shape=(B, K)>]\n        '
        return _unbatch_helper(self.values, self.max_len)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'RepeatedValues(value={}, lengths={}, max_len={})'.format(repr(self.values), repr(self.lengths), self.max_len)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return repr(self)

def _get_batch_dim_helper(v: TensorStructType) -> int:
    if False:
        return 10
    'Tries to find the batch dimension size of v, or None.'
    if isinstance(v, dict):
        for u in v.values():
            return _get_batch_dim_helper(u)
    elif isinstance(v, tuple):
        return _get_batch_dim_helper(v[0])
    elif isinstance(v, RepeatedValues):
        return _get_batch_dim_helper(v.values)
    else:
        B = v.shape[0]
        if hasattr(B, 'value'):
            B = B.value
        return B

def _unbatch_helper(v: TensorStructType, max_len: int) -> TensorStructType:
    if False:
        return 10
    'Recursively unpacks the repeat dimension (max_len).'
    if isinstance(v, dict):
        return {k: _unbatch_helper(u, max_len) for (k, u) in v.items()}
    elif isinstance(v, tuple):
        return tuple((_unbatch_helper(u, max_len) for u in v))
    elif isinstance(v, RepeatedValues):
        unbatched = _unbatch_helper(v.values, max_len)
        return [RepeatedValues(u, v.lengths[:, i, ...], v.max_len) for (i, u) in enumerate(unbatched)]
    else:
        return [v[:, i, ...] for i in range(max_len)]

def _batch_index_helper(v: TensorStructType, i: int, j: int) -> TensorStructType:
    if False:
        while True:
            i = 10
    'Selects the item at the ith batch index and jth repetition.'
    if isinstance(v, dict):
        return {k: _batch_index_helper(u, i, j) for (k, u) in v.items()}
    elif isinstance(v, tuple):
        return tuple((_batch_index_helper(u, i, j) for u in v))
    elif isinstance(v, list):
        return _batch_index_helper(v[j], i, j)
    elif isinstance(v, RepeatedValues):
        unbatched = v.unbatch_all()
        return unbatched[i]
    else:
        return v[i, ...]