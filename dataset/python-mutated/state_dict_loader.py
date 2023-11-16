from typing import Any, Dict, Optional
import torch
import torch.distributed as dist
from .storage import StorageReader
from .planner import LoadPlanner
from .default_planner import DefaultLoadPlanner
from .utils import _DistWrapper
__all__ = ['load_state_dict']

def load_state_dict(state_dict: Dict[str, Any], storage_reader: StorageReader, process_group: Optional[dist.ProcessGroup]=None, coordinator_rank: int=0, no_dist: bool=False, planner: Optional[LoadPlanner]=None) -> None:
    if False:
        while True:
            i = 10
    '\n    Load a distributed ``state_dict`` in SPMD style.\n\n    Each rank will try to read the least amount of data necessary\n    to fullfill the requested `state_dict`. When loading :class:`ShardedTensor`\n    instances, each rank only reads data for their local shards.\n\n    .. warning::\n        All tensors in ``state_dict`` must be allocated on their\n        destination device *prior to* calling this function.\n\n        All non-tensor data is loaded using `torch.load()` and modified in place\n        on state_dict.\n\n    .. warning::\n        Users must call `load_state_dict` on the root module to ensure load\n        pos-processing and non-tensor data properly propagates.\n\n    .. note:\n        This function can be used for local inference and load a checkpoint\n        produced by ``save_state_dict`` without having a process group initialized\n        by passing ``no_dist=True`` and by using Tensors instead of ShardedTensors.\n\n    Args:\n        state_dict (Dict[str, Any]) : The state_dict to load. Note that this\n            state dict will updated in place.\n        storage_reader (StorageReader): StorageReader used to load data from.\n        process_group (ProcessGroup):\n            ProcessGroup to be used for cross-rank synchronization.\n        coordinator_rank (int):\n            Rank to use to coordinate the checkpoint.\n            rank0 is used by default.\n        no_dist (bool): If ``True``, distributed checkpoint will not load\n            in SPMD style. (Default: ``False``)\n\n    Returns:\n        None.\n\n    Examples\n        >>> # xdoctest: +SKIP\n        >>> my_model = MyModule()\n        >>> optimizer = Adagrad(my_model.parameters())\n        >>> model_state_dict = my_model.state_dict()\n        >>> fs_storage_reader = torch.distributed.checkpoint.FileSystemReader("/checkpoint/1")\n\n        >>> torch.distributed.checkpoint.load_state_dict(\n        >>>     state_dict=model_state_dict,\n        >>>     storage_reader=fs_storage_reader,\n        >>> )\n\n        >>> # module.load_state_dict() function might have customized steps\n        >>> # to flush the state_dict, must call it to\n        >>> # ensure correct behavior.\n        >>> my_model.load_state_dict(model_state_dict)\n\n    .. note::\n        load_state_dict uses collectives to coordinate reads across ranks.\n        For NCCL-based process groups, internal tensor representations of\n        objects must be moved to the GPU device before communication takes place.\n        In this case, the device used is given by ``torch.cuda.current_device()``\n        and it is the user\'s responsibility to ensure that this is set so that each\n        rank has an individual GPU, via ``torch.cuda.set_device()``.\n    '
    torch._C._log_api_usage_once('torch.distributed.checkpoint.load_state_dict')
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()

    def local_step():
        if False:
            print('Hello World!')
        assert planner is not None
        metadata = storage_reader.read_metadata()
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
        storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)
        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        if False:
            print('Hello World!')
        assert planner is not None
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans
    central_plan = distW.reduce_scatter('plan', local_step, global_step)

    def read_data():
        if False:
            for i in range(10):
                print('nop')
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)
        all_reads.wait()
        return None
    _ = distW.all_gather('read', read_data)