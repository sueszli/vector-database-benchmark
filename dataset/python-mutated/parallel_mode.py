from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.distributed._spmd.data_parallel import DataParallelStyle, partition_data_parallel
from torch.distributed._spmd.distribute import _convert_to_distributed, Schema
from torch.distributed._tensor import DeviceMesh, Placement, Replicate, Shard
from torch.fx import GraphModule

class ParallelMode(ABC):
    """
    Basic Parallel Mode interface. Each parallelism pattern should implement
    this interface to describe how to partition and compile the graph in the
    spmd compiler.
    """

    @abstractmethod
    def partition(self, gm: GraphModule, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], params_and_buffers: Dict[str, Any], named_states: Dict[str, Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> GraphModule:
        if False:
            return 10
        '\n        Partition a single device graph to a distributed graph.\n\n        TODO(@wanchaol): some of these arguments are not necessary for\n        partitioning, remove the unnecessary ones later.\n        '
        raise NotImplementedError()

    @abstractmethod
    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform and compile a distributed graph with a set of graph\n        transformation and optimization passes for each parallel mode.\n\n        The returned result should be a compiled executable graph in\n        the distributed environment.\n        '
        raise NotImplementedError()

class DataParallel(ParallelMode):
    """Data Parallelism mode."""

    def __init__(self, parallel_style: str='replicate', *, input_batch_dim: int=0, custom_passes: Optional[Callable[[GraphModule], GraphModule]]=None):
        if False:
            return 10
        '\n        DataParallel Mode that partition the model and graph to data parallel style\n        parallelism (i.e. DDP/FSDP/ZERO-3). It currently supports three different\n        parallel styles: "replicate", "fully_shard", and "default". See\n        :class:`DataParallelStyle` for more details.\n\n        Args:\n            parallel_style (str): parallel style to use. Currently supports\n                "replicate", "fully_shard", and "default".\n\n        Keyword args:\n            input_batch_dim (int): the batch dimension of the input tensor.\n                 default: 0\n            custom_passes (Callable[[GraphModule], GraphModule], optional):\n                A custom callable that overrides the default graph transformation\n                and optimization passes.\n        '
        if parallel_style == 'replicate':
            self.parallel_style = DataParallelStyle.REPLICATE
        elif parallel_style == 'fully_shard':
            self.parallel_style = DataParallelStyle.FULLY_SHARD
        elif parallel_style == 'default':
            self.parallel_style = DataParallelStyle.DEFAULT
        else:
            raise RuntimeError(f'Unknown parallel style: {parallel_style}')
        self.input_batch_dim = input_batch_dim
        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            self._gm_passes = lambda gm: gm

    def partition(self, gm: GraphModule, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], params_and_buffers: Dict[str, Any], named_states: Dict[str, Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> GraphModule:
        if False:
            for i in range(10):
                print('nop')
        mesh = DeviceMesh('cuda', torch.arange(dist.get_world_size()))
        gm = partition_data_parallel(gm, model, optimizer, params_and_buffers, named_states, args, kwargs, mesh, self.parallel_style, self.input_batch_dim)
        return gm

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        if False:
            print('Hello World!')
        'optimize a distributed graph with a set of optimization passes'
        return self._gm_passes(gm)

class DTensorExpandMode(ParallelMode):
    """
    The DTensor Expand mode. It's replicating the parameters and
    shard the inputs to represent DDP like behavior, it's currently
    a transitent mode before we move to the new data parallel expansion.
    """

    def __init__(self, custom_passes: Optional[Callable[[GraphModule], GraphModule]]=None):
        if False:
            while True:
                i = 10
        self._placements_override: Dict[int, List[Placement]] = {}
        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            self._gm_passes = lambda gm: gm

    def partition(self, gm: GraphModule, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], params_and_buffers: Dict[str, Any], named_states: Dict[str, Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> GraphModule:
        if False:
            print('Hello World!')
        flat_args = pytree.arg_tree_leaves(*args, **kwargs)
        mesh = DeviceMesh('cuda', torch.arange(dist.get_world_size()).cuda())
        shard_schema: Schema = Schema(mesh=mesh, placements=[Shard(0)])
        replicate_schema: Schema = Schema(mesh=mesh, placements=[Replicate()])
        (inps, schemas) = ([], [])
        for p in pytree.tree_leaves(params_and_buffers):
            assert isinstance(p, torch.Tensor), f'expecting Tensor but got {type(p)}'
            inps.append(p)
            schemas.append(replicate_schema)
        for o in pytree.tree_leaves(named_states):
            if isinstance(o, torch.Tensor):
                inps.append(o)
                schemas.append(replicate_schema)
            else:
                inps.append(torch.empty(0))
                schemas.append(replicate_schema)
        for a in flat_args:
            if isinstance(a, torch.Tensor):
                inps.append(a)
                if id(a) in self._placements_override:
                    schemas.append(Schema(mesh=mesh, placements=self._placements_override[id(a)]))
                else:
                    schemas.append(shard_schema)
            else:
                inps.append(torch.empty(0))
                schemas.append(shard_schema)
        with FakeTensorMode(allow_non_fake_inputs=True):
            fake_inps = [torch.empty_like(inp) for inp in inps]
        return _convert_to_distributed(gm, fake_inps, schemas, default_mesh=mesh, _allow_partial=False)[0]

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform and compile a distributed graph with a set of graph transformation\n        and optimization passes for the dtensor fallback parallel mode.\n        '
        return self._gm_passes(gm)