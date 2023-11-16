import typing
from typing import Sequence
from itertools import chain
import cupy
import cupy._creation.basic as _creation_basic
from cupy._core.core import ndarray
from cupy.cuda.device import Device
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _modes

def _find_updates(args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray'], dev: int, chunk_i: int) -> list['_data_transfer._PartialUpdate']:
    if False:
        print('Hello World!')
    updates: list[_data_transfer._PartialUpdate] = []
    at_most_one_update = True
    for arg in chain(args, kwargs.values()):
        updates_now = arg._chunks_map[dev][chunk_i].updates
        if updates_now:
            if updates:
                at_most_one_update = False
                break
            updates = updates_now
    if at_most_one_update:
        return updates
    for arg in chain(args, kwargs.values()):
        for chunk in chain.from_iterable(arg._chunks_map.values()):
            chunk.flush(arg._mode)
    return []

def _prepare_chunks_array(stream: Stream, args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray'], dev: int, chunk_i: int) -> tuple[list[ndarray], dict[str, ndarray]]:
    if False:
        print('Hello World!')

    def access_array(d_array):
        if False:
            return 10
        chunk = d_array._chunks_map[dev][chunk_i]
        stream.wait_event(chunk.ready)
        return chunk.array
    arg_arrays = [access_array(arg) for arg in args]
    kwarg_arrays = {key: access_array(arg) for (key, arg) in kwargs.items()}
    return (arg_arrays, kwarg_arrays)

def _change_all_to_replica_mode(args: list['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray']) -> None:
    if False:
        i = 10
        return i + 15
    args[:] = [arg._to_op_mode(_modes.REPLICA) for arg in args]
    kwargs.update(((k, arg._to_op_mode(_modes.REPLICA)) for (k, arg) in kwargs.items()))

def _execute_kernel(kernel, args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray']) -> '_array.DistributedArray':
    if False:
        for i in range(10):
            print('nop')
    args = list(args)
    _change_all_to_replica_mode(args, kwargs)
    out_dtype = None
    out_chunks_map: dict[int, list[_chunk._Chunk]] = {}
    for arg in args or kwargs.values():
        index_map = arg.index_map
        break
    for (dev, idxs) in index_map.items():
        out_chunks_map[dev] = []
        with Device(dev):
            stream = get_current_stream()
            for (chunk_i, idx) in enumerate(idxs):
                updates = _find_updates(args, kwargs, dev, chunk_i)
                (arg_arrays, kwarg_arrays) = _prepare_chunks_array(stream, args, kwargs, dev, chunk_i)
                out_chunk = None
                for data in chain(arg_arrays, kwarg_arrays.values()):
                    if isinstance(data, _chunk._ArrayPlaceholder):
                        assert out_chunk is None
                        out_chunk = _chunk._Chunk.create_placeholder(data.shape, data.device, idx)
                if out_chunk is None:
                    out_array = kernel(*arg_arrays, **kwarg_arrays)
                    out_dtype = out_array.dtype
                    out_chunk = _chunk._Chunk(out_array, stream.record(), idx, prevent_gc=(arg_arrays, kwarg_arrays))
                out_chunks_map[dev].append(out_chunk)
                if not updates:
                    continue
                arg_slices = [None] * len(arg_arrays)
                kwarg_slices = {}
                for (update, idx) in updates:
                    for (i, data) in enumerate(arg_arrays):
                        if isinstance(data, _chunk._ArrayPlaceholder):
                            arg_slices[i] = update.array
                        else:
                            arg_slices[i] = data[idx]
                    for (k, data) in kwarg_arrays.items():
                        if isinstance(data, _chunk._ArrayPlaceholder):
                            kwarg_slices[k] = update.array
                        else:
                            kwarg_slices[k] = data[idx]
                    stream.wait_event(update.ready)
                    out_update_array = kernel(*arg_slices, **kwarg_slices)
                    out_dtype = out_update_array.dtype
                    ready = stream.record()
                    out_update = _data_transfer._AsyncData(out_update_array, ready, prevent_gc=(arg_slices, kwarg_slices))
                    out_chunk.add_update(out_update, idx)
    for chunk in chain.from_iterable(out_chunks_map.values()):
        if not isinstance(chunk.array, (ndarray, _chunk._ArrayPlaceholder)):
            raise RuntimeError('Kernels returning other than signle array are not supported')
    shape = comms = None
    for arg in args or kwargs.values():
        shape = arg.shape
        comms = arg._comms
        break
    assert shape is not None
    return _array.DistributedArray(shape, out_dtype, out_chunks_map, _modes.REPLICA, comms)

def _execute_peer_access(kernel, args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray']) -> '_array.DistributedArray':
    if False:
        print('Hello World!')
    'Arguments must be in the replica mode.'
    assert len(args) >= 2
    if len(args) > 2:
        raise RuntimeError('Element-wise operation over more than two distributed arrays is not supported unless they share the same index_map.')
    if kwargs:
        raise RuntimeError('Keyword argument is not supported unless arguments share the same index_map.')
    args = list(args)
    for (i, arg) in enumerate(args):
        args[i] = arg._to_op_mode(_modes.REPLICA)
        for chunk in chain.from_iterable(args[i]._chunks_map.values()):
            chunk.flush(_modes.REPLICA)
    (a, b) = args
    if isinstance(kernel, cupy._core._kernel.ufunc):
        op = kernel._ops._guess_routine_from_in_types((a.dtype, b.dtype))
        if op is None:
            raise RuntimeError(f'Could not guess the return type of {kernel.name} with arguments of type {(a.dtype.type, b.dtype.type)}')
        out_types = op.out_types
    else:
        assert isinstance(kernel, cupy._core._kernel.ElementwiseKernel)
        (_, out_types, _) = kernel._decide_params_type((a.dtype.type, b.dtype.type), ())
    if len(out_types) != 1:
        print(out_types)
        raise RuntimeError('Kernels returning other than signle array are not supported')
    dtype = out_types[0]
    shape = a.shape
    comms = a._comms
    out_chunks_map: dict[int, list[_chunk._Chunk]] = {}
    for a_chunk in chain.from_iterable(a._chunks_map.values()):
        a_dev = a_chunk.array.device.id
        with a_chunk.on_ready() as stream:
            out_array = _creation_basic.empty(a_chunk.array.shape, dtype)
            for b_chunk in chain.from_iterable(b._chunks_map.values()):
                intersection = _index_arith._index_intersection(a_chunk.index, b_chunk.index, shape)
                if intersection is None:
                    continue
                b_dev = b_chunk.array.device.id
                if cupy.cuda.runtime.deviceCanAccessPeer(a_dev, b_dev) != 1:
                    b_chunk = _array._make_chunk_async(b_dev, a_dev, b_chunk.index, b_chunk.array, b._comms)
                else:
                    cupy._core._kernel._check_peer_access(b_chunk.array, a_dev)
                stream.wait_event(b_chunk.ready)
                a_new_idx = _index_arith._index_for_subindex(a_chunk.index, intersection, shape)
                b_new_idx = _index_arith._index_for_subindex(b_chunk.index, intersection, shape)
                assert kernel.nin == 2
                kernel(typing.cast(ndarray, a_chunk.array)[a_new_idx], typing.cast(ndarray, b_chunk.array)[b_new_idx], out_array[a_new_idx])
            out_chunk = _chunk._Chunk(out_array, stream.record(), a_chunk.index, prevent_gc=b._chunks_map)
            out_chunks_map.setdefault(a_dev, []).append(out_chunk)
    return _array.DistributedArray(shape, dtype, out_chunks_map, _modes.REPLICA, comms)

def _is_peer_access_needed(args: Sequence['_array.DistributedArray'], kwargs: dict[str, '_array.DistributedArray']) -> bool:
    if False:
        print('Hello World!')
    index_map = None
    for arg in chain(args, kwargs.values()):
        if index_map is None:
            index_map = arg.index_map
        elif arg.index_map != index_map:
            return True
    return False

def _execute(kernel, args: tuple, kwargs: dict):
    if False:
        return 10
    for arg in chain(args, kwargs.values()):
        if not isinstance(arg, _array.DistributedArray):
            raise RuntimeError('Mixing a distributed array with a non-distributed one is not supported')
    needs_peer_access = _is_peer_access_needed(args, kwargs)
    if needs_peer_access:
        return _execute_peer_access(kernel, args, kwargs)
    else:
        return _execute_kernel(kernel, args, kwargs)