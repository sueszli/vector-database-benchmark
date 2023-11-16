"""
A set of primitive functions for performing collective ops.

Each should also handle single rank scenario.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, cast, Generic, List, Optional, Tuple, TypeVar, Union
import torch.distributed as dist
T = TypeVar('T')

@dataclass
class SyncPayload(Generic[T]):
    stage_name: Optional[str]
    success: bool
    payload: T
    exception: Optional[Exception] = None

def broadcast(data_or_fn: Union[T, Callable[[], T]], *, success: bool=True, stage_name: Optional[str]=None, rank: int=0, pg: Optional[dist.ProcessGroup]=None) -> T:
    if False:
        return 10
    '\n    Broadcasts the data payload from rank 0 to all other ranks.\n    Or if a function is passed, execute it in rank 0 and broadcast result to all other ranks.\n\n    Can be used to broadcast a failure signal to stop all ranks.\n\n    If the function raises an exception, all ranks will raise.\n\n    Args:\n        data_or_fn: the data to broadcast or function to execute and broadcast result.\n        success: False to stop all ranks.\n        stage_name: the name of the logical stage for synchronization and debugging\n        rank: rank to broadcast data or execute function and broadcast resutls.\n        pg: the process group for sync\n    Throws:\n        RuntimeError from original exception trace\n    Returns:\n        the value after synchronization\n\n    Example usage:\n    >> id = broadcast(data_or_fn=allocate_id, rank=0, pg=ext_pg.my_pg)\n    '
    if not success and data_or_fn is not None:
        raise AssertionError('Data or Function is expected to be None if not successful')
    payload: Optional[T] = None
    exception: Optional[Exception] = None
    if pg is None and rank == 0 or (pg is not None and pg.rank() == rank):
        if callable(data_or_fn):
            try:
                payload = data_or_fn()
            except Exception as e:
                success = False
                exception = e
        else:
            payload = data_or_fn
    sync_obj = SyncPayload(stage_name=stage_name, success=success, payload=payload, exception=exception)
    if pg is not None:
        broadcast_list = [sync_obj]
        dist.broadcast_object_list(broadcast_list, src=rank, group=pg)
        assert len(broadcast_list) == 1
        sync_obj = broadcast_list[0]
    if not sync_obj.success:
        error_msg = f'Rank {rank} failed'
        if stage_name is not None:
            error_msg += f': stage {sync_obj.stage_name}'
        if sync_obj.exception is not None:
            error_msg += f': exception {sync_obj.exception}'
        raise RuntimeError(error_msg) from sync_obj.exception
    return cast(T, sync_obj.payload)

def all_gather(data_or_fn: Union[T, Callable[[], T]], stage_name: Optional[str]=None, pg: Optional[dist.ProcessGroup]=None) -> List[T]:
    if False:
        print('Hello World!')
    '\n    A simple all_gather primitive with basic synchronization guard logic,\n    by checking payload from all ranks has the same stage name.\n\n    Args:\n        data_or_fn: the data to be all gathered across ranks or function to be executed\n        stage_name: the sync stage name for out-of-sync protection\n        pg: the process group for sync\n    Throws:\n        RuntimeError from original exception trace\n    Returns:\n        a list of synced data from all ranks\n\n    Example usage:\n    >> all_ids = all_gather(data_or_fn=allocate_id, pg=ext_pg.my_pg)\n    '
    payload: Optional[T] = None
    exception: Optional[Exception] = None
    success = True
    if callable(data_or_fn):
        try:
            payload = data_or_fn()
        except Exception as e:
            success = False
            exception = e
    else:
        payload = data_or_fn
    sync_obj = SyncPayload(stage_name=stage_name, success=success, payload=payload, exception=exception)
    if pg is not None:
        total_list = [None] * dist.get_world_size(pg)
        all_gather_object_enforce_type(pg, total_list, sync_obj)
        stage_name = cast(SyncPayload[T], total_list[0]).stage_name
        exception_list: List[Tuple[int, Exception]] = []
        ret_list: List[T] = []
        error_msg: str = ''
        for (i, sp) in enumerate(cast(List[SyncPayload[T]], total_list)):
            if sp.stage_name != stage_name:
                error_msg += f'Unexpected stage name received from rank {i}: {sp.stage_name} '
                continue
            if not sp.success and sp.exception is not None:
                exception_list.append((i, sp.exception))
                continue
            ret_list.append(sp.payload)
        if len(exception_list) > 0:
            raise RuntimeError(error_msg, exception_list) from exception_list[0]
        return ret_list
    else:
        if not sync_obj.success:
            raise RuntimeError(f'all_gather failed with exception {sync_obj.exception}') from sync_obj.exception
        return [sync_obj.payload]

def all_gather_object_enforce_type(pg: dist.ProcessGroup, object_list: List[Any], obj: Any, type_checker: Callable[[Any, Any], bool]=lambda x, y: type(x) == type(y)) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Similar to plain all_gather_object but with additional type checking\n    AFTER gather is done to ensure basic consistency.\n    If check does not pass, all ranks will fail with exception.\n\n    This is generally to prevent conditional logic leading to\n    unexpected messages being received. This is considered fatal code error,\n    but due to logic stacks this might happen implicitly in practice.\n\n    The default check does not check sub type (considered different)\n    or covariance (considered same) but users can pass in custom checker\n    if more complicated check is needed.\n    '
    dist.all_gather_object(object_list, obj, group=pg)
    list_len = len(object_list)
    if list_len == 0:
        return
    first_obj = object_list[0]
    for i in range(1, list_len):
        if not type_checker(first_obj, object_list[i]):
            raise TypeError(f'Object type at index {i} is {type(object_list[i])}, while first object type is {type(first_obj)}')