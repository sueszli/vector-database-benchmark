from typing import Optional
import torch
import torch.distributed as dist
from .planner import SavePlanner
from .default_planner import DefaultSavePlanner
from .storage import StorageWriter
from .metadata import Metadata, STATE_DICT_TYPE
from .utils import _DistWrapper
__all__ = ['save_state_dict']

def save_state_dict(state_dict: STATE_DICT_TYPE, storage_writer: StorageWriter, process_group: Optional[dist.ProcessGroup]=None, coordinator_rank: int=0, no_dist: bool=False, planner: Optional[SavePlanner]=None) -> Metadata:
    if False:
        i = 10
        return i + 15
    '\n    Save a distributed model in SPMD style.\n\n    This function is different from ``torch.save()`` as it handles\n    ``ShardedTensor`` by having each rank only save their local shards.\n\n    .. warning::\n        There is no guarantees of Backwards Compatibility across PyTorch versions\n        for saved state_dicts.\n\n    .. warning::\n        If using the `process_group` argument, make sure that only its ranks\n        call `save_state_dict` and that all data in state_dict belong to it.\n\n    .. note::\n        When saving checkpoint for FSDP\'s `ShardingStrategy.HYBRID_SHARD`, only one of\n        the shard_group should be calling `save_state_dict` and the corresponding process\n        group needs to be passed in.\n\n    .. note::\n        This function can be used to save a state_dict without having a process group\n        initialized by passing ``no_dist=True``.\n\n\n    Args:\n        state_dict (Dict[str, Any]): The state_dict to save.\n        storage_writer (StorageWriter):\n            Instance of StorageWrite use to perform writes.\n        process_group (ProcessGroup):\n            ProcessGroup to be used for cross-rank synchronization.\n        coordinator_rank (int): Rank to use to coordinate the checkpoint.\n            rank0 is used by default.\n        no_dist (bool): If ``True``, distributed checkpoint will not save\n            in SPMD style. (Default: ``False``)\n\n    Returns:\n        Metadata: Metadata object for the saved checkpoint.\n\n    Example:\n        >>> # xdoctest: +SKIP\n        >>> my_model = MyModule()\n\n        >>> model_state_dict = my_model.state_dict()\n\n        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter("/checkpoint/1")\n        >>> torch.distributed.checkpoint.save_state_dict(\n        >>>     state_dict=model_state_dict,\n        >>>     storage_writer=fs_storage_writer,\n        >>> )\n\n    .. note::\n        save_state_dict uses collectives to coordinate writes across ranks.\n        For NCCL-based process groups, internal tensor representations of\n        objects must be moved to the GPU device before communication takes place.\n        In this case, the device used is given by ``torch.cuda.current_device()``\n        and it is the user\'s responsibility to ensure that this is set so that\n        each rank has an individual GPU, via ``torch.cuda.set_device()``.\n    '
    torch._C._log_api_usage_once('torch.distributed.checkpoint.save_state_dict')
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None
    global_metatadata = None

    def local_step():
        if False:
            for i in range(10):
                print('nop')
        assert planner is not None
        planner.set_up_planner(state_dict, distW.is_coordinator)
        storage_writer.set_up_storage_writer(distW.is_coordinator)
        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        if False:
            while True:
                i = 10
        nonlocal global_metatadata
        assert planner is not None
        (all_local_plans, global_metatadata) = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans
    central_plan = distW.reduce_scatter('plan', local_step, global_step)

    def write_data():
        if False:
            i = 10
            return i + 15
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)
        all_writes.wait()
        return all_writes.value()

    def finish_checkpoint(all_results):
        if False:
            while True:
                i = 10
        assert global_metatadata is not None
        storage_writer.finish(metadata=global_metatadata, results=all_results)
        return global_metatadata
    return distW.all_reduce('write', write_data, finish_checkpoint)