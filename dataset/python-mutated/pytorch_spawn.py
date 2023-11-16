from typing import Any, Callable, Protocol, Type, runtime_checkable
from lightning.app.components.multi_node.base import MultiNode
from lightning.app.core.queues import MultiProcessQueue
from lightning.app.core.work import LightningWork
from lightning.app.utilities.packaging.cloud_compute import CloudCompute
from lightning.app.utilities.proxies import WorkRunExecutor, WorkStateObserver, _proxy_setattr, unwrap

@runtime_checkable
class _PyTorchSpawnWorkProtocol(Protocol):

    def run(self, world_size: int, node_rank: int, global_rank: int, local_rank: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

class _PyTorchSpawnRunExecutor(WorkRunExecutor):
    enable_start_observer: bool = False

    def __call__(self, main_address: str, main_port: int, num_nodes: int, node_rank: int):
        if False:
            i = 10
            return i + 15
        import torch
        with self.enable_spawn():
            nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
            queue = self.delta_queue if isinstance(self.delta_queue, MultiProcessQueue) else self.delta_queue.to_dict()
            torch.multiprocessing.spawn(self.dispatch_run, args=(self.__class__, self.work, queue, main_address, main_port, num_nodes, node_rank, nprocs), nprocs=nprocs)

    @staticmethod
    def dispatch_run(local_rank, cls, work, delta_queue, *args: Any, **kwargs: Any):
        if False:
            print('Hello World!')
        if local_rank == 0:
            if isinstance(delta_queue, dict):
                delta_queue = cls.process_queue(delta_queue)
                work._request_queue = cls.process_queue(work._request_queue)
                work._response_queue = cls.process_queue(work._response_queue)
            state_observer = WorkStateObserver(work, delta_queue=delta_queue)
            state_observer.start()
            _proxy_setattr(work, delta_queue, state_observer)
        cls.run(local_rank, unwrap(work.run), *args, **kwargs)
        if local_rank == 0:
            state_observer.join(0)

    @staticmethod
    def run(local_rank: int, work_run: Callable, main_address: str, main_port: int, num_nodes: int, node_rank: int, nprocs: int):
        if False:
            for i in range(10):
                print('nop')
        import torch
        global_rank = local_rank + node_rank * nprocs
        world_size = num_nodes * nprocs
        if torch.distributed.is_available():
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group('nccl' if torch.cuda.is_available() else 'gloo', rank=global_rank, world_size=world_size, init_method=f'tcp://{main_address}:{main_port}')
        elif world_size > 1:
            raise Exception('Torch distributed should be available.')
        return work_run(world_size, node_rank, global_rank, local_rank)

class PyTorchSpawnMultiNode(MultiNode):

    def __init__(self, work_cls: Type['LightningWork'], cloud_compute: 'CloudCompute', num_nodes: int, *work_args: Any, **work_kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        assert issubclass(work_cls, _PyTorchSpawnWorkProtocol)
        work_cls._run_executor_cls = _PyTorchSpawnRunExecutor
        super().__init__(work_cls, num_nodes, cloud_compute, *work_args, **work_kwargs)