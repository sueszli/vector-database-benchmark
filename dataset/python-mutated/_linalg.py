import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
_SliceIndices = tuple[int, int, int]
_BlockIdx = tuple[_SliceIndices, _SliceIndices]
_BlockLocationMap = dict[_BlockIdx, dict[int, int]]
_ExecutionPlan = list[tuple[_BlockIdx, _BlockIdx, int]]
_BatchIdx = tuple[_SliceIndices, ...]

@dataclasses.dataclass
class _Blocking:
    i_partitions: list[int]
    j_partitions: list[int]
    k_partitions: list[int]

def _find_blocking(location_map_a: _BlockLocationMap, location_map_b: _BlockLocationMap) -> _Blocking:
    if False:
        for i in range(10):
            print('nop')
    i_partitions: list[int] = []
    j_partitions: list[int] = []
    k_partitions: list[int] = []

    def add_to_partitions(indices, partitions):
        if False:
            print('Hello World!')
        (start, stop, step) = indices
        if step != 1:
            raise RuntimeError('Step other than 1 is not supported')
        partitions.append(start)
        partitions.append(stop)
    for (i_indices, k_indices) in location_map_a.keys():
        add_to_partitions(i_indices, i_partitions)
        add_to_partitions(k_indices, k_partitions)
    for (k_indices, j_indices) in location_map_b.keys():
        add_to_partitions(k_indices, k_partitions)
        add_to_partitions(j_indices, j_partitions)

    def to_unique_sorted(partitions):
        if False:
            return 10
        if len(partitions) == 0:
            raise RuntimeError('Array has no chunk')
        partitions.sort()
        res = [partitions[0]]
        for (x, y) in zip(partitions, partitions[1:]):
            if x != y:
                res.append(y)
        return res
    i_partitions = to_unique_sorted(i_partitions)
    j_partitions = to_unique_sorted(j_partitions)
    k_partitions = to_unique_sorted(k_partitions)

    def check_indices(indices, partitions):
        if False:
            return 10
        (start, stop, _) = indices
        if partitions.index(start) + 1 != partitions.index(stop):
            raise RuntimeError('Inconsistent index mapping')
    for (i_indices, k_indices) in location_map_a.keys():
        check_indices(i_indices, i_partitions)
        check_indices(k_indices, k_partitions)
    for (k_indices, j_indices) in location_map_b.keys():
        check_indices(k_indices, k_partitions)
        check_indices(j_indices, j_partitions)
    return _Blocking(i_partitions, j_partitions, k_partitions)

def _make_execution_plan(blocking: _Blocking, location_map_a: _BlockLocationMap, location_map_b: _BlockLocationMap) -> _ExecutionPlan:
    if False:
        while True:
            i = 10
    i_partitions = blocking.i_partitions
    j_partitions = blocking.j_partitions
    k_partitions = blocking.k_partitions
    plan: _ExecutionPlan = []
    for i_range in zip(i_partitions, i_partitions[1:]):
        for j_range in zip(j_partitions, j_partitions[1:]):
            for k_range in zip(k_partitions, k_partitions[1:]):
                block_a = (i_range + (1,), k_range + (1,))
                block_b = (k_range + (1,), j_range + (1,))
                devices_a = set(location_map_a[block_a].keys())
                devices_b = set(location_map_b[block_b].keys())
                intersection = devices_a & devices_b
                if intersection:
                    dev = intersection.pop()
                    plan.append((block_a, block_b, dev))
                else:
                    raise RuntimeError(f'There is no device that can perform multiplication between block {block_a} and {block_b}')
    return plan

def _convert_to_tuples(slices: tuple[slice, ...], shape: tuple[int, ...]) -> tuple[_SliceIndices, ...]:
    if False:
        i = 10
        return i + 15
    assert len(slices) == len(shape)
    return tuple((s.indices(l) for (s, l) in zip(slices, shape)))

def _convert_to_slices(tuples: tuple[_SliceIndices, ...]) -> tuple[slice, ...]:
    if False:
        i = 10
        return i + 15
    return tuple((slice(*t) for t in tuples))

def _group_by_batch(shape: tuple[int, ...], index_map: dict[int, list[tuple[slice, ...]]]) -> dict[_BatchIdx, _BlockLocationMap]:
    if False:
        for i in range(10):
            print('nop')
    location_maps: dict[_BatchIdx, _BlockLocationMap] = {}
    for (dev, idxs) in index_map.items():
        for (chunk_i, idx) in enumerate(idxs):
            idx_tuples = _convert_to_tuples(idx, shape)
            (batch_idx, block_idx) = (idx_tuples[:-2], idx_tuples[-2:])
            block_idx = typing.cast(_BlockIdx, block_idx)
            location_map = location_maps.setdefault(batch_idx, {})
            location = location_map.setdefault(block_idx, {})
            location[dev] = chunk_i
    return location_maps

def _reshape_array_with(arr: '_array.DistributedArray', f_shape: Callable[[tuple[int, ...]], tuple[int, ...]], f_idx: Callable[[tuple[slice, ...]], tuple[slice, ...]]) -> '_array.DistributedArray':
    if False:
        i = 10
        return i + 15

    def reshape_chunk(chunk: _chunk._Chunk) -> _chunk._Chunk:
        if False:
            print('Hello World!')
        data = chunk.array.reshape(f_shape(chunk.array.shape))
        index = f_idx(chunk.index)
        updates = [(data, f_idx(idx)) for (data, idx) in chunk.updates]
        return _chunk._Chunk(data, chunk.ready, index, updates, chunk.prevent_gc)
    chunks_map = {}
    for (dev, chunks) in arr._chunks_map.items():
        chunks_map[dev] = [reshape_chunk(chunk) for chunk in chunks]
    shape = f_shape(arr.shape)
    return _array.DistributedArray(shape, arr.dtype, chunks_map, arr._mode, arr._comms)

def _prepend_one_to_shape(arr) -> '_array.DistributedArray':
    if False:
        i = 10
        return i + 15
    return _reshape_array_with(arr, lambda shape: (1,) + shape, lambda idx: (slice(None),) + idx)

def _append_one_to_shape(arr) -> '_array.DistributedArray':
    if False:
        return 10
    return _reshape_array_with(arr, lambda shape: shape + (1,), lambda idx: idx + (slice(None),))

def _pop_from_shape(arr) -> '_array.DistributedArray':
    if False:
        for i in range(10):
            print('nop')
    assert arr.shape[-1] == 1
    return _reshape_array_with(arr, lambda shape: shape[:-1], lambda idx: idx[:-1])

def _pop_front_from_shape(arr) -> '_array.DistributedArray':
    if False:
        return 10
    assert arr.shape[0] == 1
    return _reshape_array_with(arr, lambda shape: shape[1:], lambda idx: idx[1:])

def matmul(a: '_array.DistributedArray', b: '_array.DistributedArray', out: Optional['_array.DistributedArray']=None, **kwargs) -> '_array.DistributedArray':
    if False:
        return 10
    "Matrix multiplication between distributed arrays.\n\n    The arguments must have compatible :attr:`~DistributedArray.shape` and\n    :attr:`~DistributedArray.index_map`.\n\n    This operation converts its operands into the replica mode, and compute\n    their product in the sum mode.\n\n    Args:\n        a, b: Input distributed arrays.\n        out (optional): A location into which the result is stored. This option\n            is currently not supported.\n    Returns:\n        The matrix product of the inputs.\n\n    Example:\n        >>> A = distributed_array(\n        ...     cupy.arange(6).reshape(2, 3),\n        ...     make_2d_index_map([0, 2], [0, 1, 3],\n        ...                       [[{0}, {1, 2}]]))\n        >>> B = distributed_array(\n        ...     cupy.arange(12).reshape(3, 4),\n        ...     make_2d_index_map([0, 1, 3], [0, 2, 4],\n        ...                       [[{0}, {0}],\n        ...                        [{1}, {2}]]))\n        >>> C = A @ B\n        >>> C.mode\n        'sum'\n        >>> C.all_chunks()\n        {0: [array([[0, 0],\n                    [0, 3]]),\n             array([[0, 0],\n                    [6, 9]])],\n         1: [array([[20, 23],\n                    [56, 65]])],\n         2: [array([[26, 29],\n                    [74, 83]])]}\n        >>> C\n        array([[20, 23, 26, 29],\n               [56, 68, 80, 92]])\n\n    .. seealso:: :obj:`numpy.matmul`\n    "
    if out is not None:
        raise RuntimeError('Argument `out` is not supported')
    for param in ('subok', 'axes', 'axis'):
        if param in kwargs:
            raise RuntimeError(f'Argument `{param}` is not supported')
    if not isinstance(a, _array.DistributedArray) or not isinstance(b, _array.DistributedArray):
        raise RuntimeError('Mixing a distributed array with a non-distributed array is not supported')
    a = a._to_op_mode(_modes.REPLICA)
    b = b._to_op_mode(_modes.REPLICA)
    one_prepended = one_appended = False
    if a.ndim == 1:
        one_prepended = True
        a = _prepend_one_to_shape(a)
    if b.ndim == 1:
        one_appended = True
        b = _append_one_to_shape(b)
    (n, m) = a.shape[-2:]
    (m2, p) = b.shape[-2:]
    if m != m2 or a.shape[:-2] != b.shape[:-2]:
        raise ValueError('Shapes are incompatible')
    location_maps_a = _group_by_batch(a.shape, a.index_map)
    location_maps_b = _group_by_batch(b.shape, b.index_map)
    if location_maps_a.keys() != location_maps_b.keys():
        raise RuntimeError('Mismatched batch shapes')
    chunks_map: dict[int, list[_chunk._Chunk]] = {dev: [] for dev in a.devices}
    dtype = None
    for batch_idx in location_maps_a.keys():
        location_map_a = location_maps_a[batch_idx]
        location_map_b = location_maps_b[batch_idx]
        blocking = _find_blocking(location_map_a, location_map_b)
        plan = _make_execution_plan(blocking, location_map_a, location_map_b)
        index_prefix = _convert_to_slices(batch_idx)
        for (block_a, block_b, dev) in plan:
            loc_a = location_map_a[block_a]
            loc_b = location_map_b[block_b]
            chunk_a = a._chunks_map[dev][loc_a[dev]]
            chunk_b = b._chunks_map[dev][loc_b[dev]]
            chunk_a.flush(_modes.REPLICA)
            chunk_b.flush(_modes.REPLICA)
            index = index_prefix + (slice(*block_a[0]), slice(*block_b[1]))
            with chunk_a.on_ready() as stream:
                stream.wait_event(chunk_b.ready)
                chunk_ab_array = cupy.linalg._product.matmul(chunk_a.array, chunk_b.array, **kwargs)
                chunk_ab = _chunk._Chunk(chunk_ab_array, stream.record(), index, prevent_gc=(chunk_a, chunk_b))
                chunks_map[dev].append(chunk_ab)
                dtype = chunk_ab_array.dtype
    shape = a.shape[:-2] + (n, p)
    res = _array.DistributedArray(shape, dtype, chunks_map, _modes.SUM, a._comms)
    if one_prepended:
        res = _pop_front_from_shape(res)
    if one_appended:
        res = _pop_from_shape(res)
    return res

def make_2d_index_map(i_partitions: list[int], j_partitions: list[int], devices: list[list[set[int]]]) -> dict[int, list[tuple[slice, ...]]]:
    if False:
        while True:
            i = 10
    'Create an ``index_map`` for a 2D matrix with a specified blocking.\n\n    Args:\n        i_partitions (list of ints): boundaries of blocks on the `i` axis\n        j_partitions (list of ints): boundaries of blocks on the `j` axis\n        devices (2D list of sets of ints): devices owning each block\n\n    Returns:\n        dict from int to array indices: index_map\n            Indices for the chunks that devices with designated IDs are going\n            to own.\n\n    Example:\n        >>> index_map = make_2d_index_map(\n        ...     [0, 2, 4], [0, 3, 5],\n        ...     [[{0}, {1}],\n        ...      [{2}, {0, 1}]])\n        >>> pprint(index_map)\n        {0: [(slice(0, 2, None), slice(0, 3, None)),\n             (slice(2, 4, None), slice(3, 5, None))],\n         1: [(slice(0, 2, None), slice(3, 5, None)),\n             (slice(2, 4, None), slice(3, 5, None))],\n         2: [(slice(2, 4, None), slice(0, 3, None))]}\n    '
    assert i_partitions[0] == 0
    assert sorted(set(i_partitions)) == i_partitions
    assert j_partitions[0] == 0
    assert sorted(set(j_partitions)) == j_partitions
    index_map: dict[int, list[tuple[slice, ...]]] = {}
    assert len(devices) == len(i_partitions) - 1
    for i in range(len(devices)):
        assert len(devices[i]) == len(j_partitions) - 1
        for j in range(len(devices[i])):
            i_start = i_partitions[i]
            i_stop = i_partitions[i + 1]
            j_start = j_partitions[j]
            j_stop = j_partitions[j + 1]
            idx = (slice(i_start, i_stop), slice(j_start, j_stop))
            for dev in devices[i][j]:
                index_map.setdefault(dev, []).append(idx)
    return index_map