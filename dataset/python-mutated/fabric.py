import importlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Type, runtime_checkable
from lightning.app.components.multi_node.base import MultiNode
from lightning.app.components.multi_node.pytorch_spawn import _PyTorchSpawnRunExecutor
from lightning.app.core.work import LightningWork
from lightning.app.utilities.packaging.cloud_compute import CloudCompute
from lightning.app.utilities.tracer import Tracer

@runtime_checkable
class _FabricWorkProtocol(Protocol):

    @staticmethod
    def run() -> None:
        if False:
            while True:
                i = 10
        ...

@dataclass
class _FabricRunExecutor(_PyTorchSpawnRunExecutor):

    @staticmethod
    def run(local_rank: int, work_run: Callable, main_address: str, main_port: int, num_nodes: int, node_rank: int, nprocs: int):
        if False:
            for i in range(10):
                print('nop')
        fabrics = []
        strategies = []
        mps_accelerators = []
        for pkg_name in ('lightning.fabric', 'lightning_' + 'fabric'):
            try:
                pkg = importlib.import_module(pkg_name)
                fabrics.append(pkg.Fabric)
                strategies.append(pkg.strategies.DDPStrategy)
                mps_accelerators.append(pkg.accelerators.MPSAccelerator)
            except (ImportError, ModuleNotFoundError):
                continue
        os.environ['MASTER_ADDR'] = main_address
        os.environ['MASTER_PORT'] = str(main_port)
        os.environ['GROUP_RANK'] = str(node_rank)
        os.environ['RANK'] = str(local_rank + node_rank * nprocs)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(num_nodes * nprocs)
        os.environ['LOCAL_WORLD_SIZE'] = str(nprocs)
        os.environ['TORCHELASTIC_RUN_ID'] = '1'
        os.environ['LT_CLI_USED'] = '1'

        def pre_fn(fabric, *args: Any, **kwargs: Any):
            if False:
                print('Hello World!')
            kwargs['devices'] = nprocs
            kwargs['num_nodes'] = num_nodes
            if any((acc.is_available() for acc in mps_accelerators)):
                old_acc_value = kwargs.get('accelerator', 'auto')
                kwargs['accelerator'] = 'cpu'
                if old_acc_value != kwargs['accelerator']:
                    warnings.warn('Forcing `accelerator=cpu` as MPS does not support distributed training.')
            else:
                kwargs['accelerator'] = 'auto'
            strategy = kwargs.get('strategy', None)
            if strategy:
                if isinstance(strategy, str):
                    if strategy == 'ddp_spawn':
                        strategy = 'ddp'
                    elif strategy == 'ddp_sharded_spawn':
                        strategy = 'ddp_sharded'
                elif isinstance(strategy, tuple(strategies)) and strategy._start_method in ('spawn', 'fork'):
                    raise ValueError("DDP Spawned strategies aren't supported yet.")
            kwargs['strategy'] = strategy
            return ({}, args, kwargs)
        tracer = Tracer()
        for lf in fabrics:
            tracer.add_traced(lf, '__init__', pre_fn=pre_fn)
        tracer._instrument()
        ret_val = work_run()
        tracer._restore()
        return ret_val

class FabricMultiNode(MultiNode):

    def __init__(self, work_cls: Type['LightningWork'], cloud_compute: 'CloudCompute', num_nodes: int, *work_args: Any, **work_kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert issubclass(work_cls, _FabricWorkProtocol)
        work_cls._run_executor_cls = _FabricRunExecutor
        super().__init__(work_cls, *work_args, num_nodes=num_nodes, cloud_compute=cloud_compute, **work_kwargs)