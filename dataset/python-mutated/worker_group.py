import logging
import os
import socket
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import exception_cause, skip_exceptions
from ray.types import ObjectRef
from ray.util.placement_group import PlacementGroup
T = TypeVar('T')
logger = logging.getLogger(__name__)

class RayTrainWorker:
    """A class to execute arbitrary functions. Does not hold any state."""

    def __execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        if False:
            i = 10
            return i + 15
        'Executes the input function and returns the output.\n\n        Args:\n            func: The function to execute.\n            args, kwargs: The arguments to pass into func.\n        '
        try:
            return func(*args, **kwargs)
        except Exception as e:
            skipped = skip_exceptions(e)
            raise skipped from exception_cause(skipped)

@dataclass
class WorkerMetadata:
    """Metadata for each worker/actor.

    This information is expected to stay the same throughout the lifetime of
    actor.

    Args:
        node_id: ID of the node this worker is on.
        node_ip: IP address of the node this worker is on.
        hostname: Hostname that this worker is on.
        resource_ids: Map of accelerator resources
        ("GPU", "neuron_cores", ..) to their IDs.
        pid: Process ID of this worker.
    """
    node_id: str
    node_ip: str
    hostname: str
    resource_ids: Dict[str, List[str]]
    pid: int

@dataclass
class Worker:
    """Class representing a Worker."""
    actor: ActorHandle
    metadata: WorkerMetadata

def create_executable_class(executable_cls: Optional[Type]=None) -> Type:
    if False:
        return 10
    'Create the executable class to use as the Ray actors.'
    if not executable_cls:
        return RayTrainWorker
    elif issubclass(executable_cls, RayTrainWorker):
        return executable_cls
    else:

        class _WrappedExecutable(executable_cls, RayTrainWorker):

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__(*args, **kwargs)
        return _WrappedExecutable

def construct_metadata() -> WorkerMetadata:
    if False:
        while True:
            i = 10
    'Creates metadata for this worker.\n\n    This function is expected to be run on the actor.\n    '
    node_id = ray.get_runtime_context().get_node_id()
    node_ip = ray.util.get_node_ip_address()
    hostname = socket.gethostname()
    resource_ids = ray.get_runtime_context().get_resource_ids()
    pid = os.getpid()
    return WorkerMetadata(node_id=node_id, node_ip=node_ip, hostname=hostname, resource_ids=resource_ids, pid=pid)

class WorkerGroup:
    """Group of Ray Actors that can execute arbitrary functions.

    ``WorkerGroup`` launches Ray actors according to the given
    specification. It can then execute arbitrary Python functions in each of
    these workers.

    If not enough resources are available to launch the actors, the Ray
    cluster will automatically scale up if autoscaling is enabled.

    Args:
        num_workers: The number of workers (Ray actors) to launch.
            Defaults to 1.
        num_cpus_per_worker: The number of CPUs to reserve for each
            worker. Fractional values are allowed. Defaults to 1.
        num_gpus_per_worker: The number of GPUs to reserve for each
            worker. Fractional values are allowed. Defaults to 0.
        additional_resources_per_worker (Optional[Dict[str, float]]):
            Dictionary specifying the extra resources that will be
            requested for each worker in addition to ``num_cpus_per_worker``
            and ``num_gpus_per_worker``.
        actor_cls (Optional[Type]): If specified use this class as the
            remote actors.
        remote_cls_args, remote_cls_kwargs: If ``remote_cls`` is provided,
            these args will be used for the worker initialization.
        placement_group (PlacementGroup|str): The placement group that workers
            should be created in. Defaults to "default" which will inherit the
            parent placement group (if child tasks should be captured).


    Example:

    .. code_block:: python

        worker_group = WorkerGroup(num_workers=2)
        output = worker_group.execute(lambda: 1)
        assert len(output) == 2
        assert all(o == 1 for o in output)
    """

    def __init__(self, num_workers: int=1, num_cpus_per_worker: float=1, num_gpus_per_worker: float=0, additional_resources_per_worker: Optional[Dict[str, float]]=None, actor_cls: Type=None, actor_cls_args: Optional[Tuple]=None, actor_cls_kwargs: Optional[Dict]=None, placement_group: Union[PlacementGroup, str]='default'):
        if False:
            for i in range(10):
                print('nop')
        if num_workers <= 0:
            raise ValueError(f'The provided `num_workers` must be greater than 0. Received num_workers={num_workers} instead.')
        if num_cpus_per_worker < 0 or num_gpus_per_worker < 0:
            raise ValueError(f'The number of CPUs and GPUs per worker must not be negative. Received num_cpus_per_worker={num_cpus_per_worker} and num_gpus_per_worker={num_gpus_per_worker}.')
        if (actor_cls_args or actor_cls_kwargs) and (not actor_cls):
            raise ValueError('`actor_cls_args` or `actor_class_kwargs` are passed in but no `actor_cls` is passed in.')
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = num_gpus_per_worker
        self.additional_resources_per_worker = additional_resources_per_worker
        self.workers = []
        self._base_cls = create_executable_class(actor_cls)
        assert issubclass(self._base_cls, RayTrainWorker)
        self._actor_cls_args = actor_cls_args or []
        self._actor_cls_kwargs = actor_cls_kwargs or {}
        self._placement_group = placement_group
        self._remote_cls = ray.remote(num_cpus=self.num_cpus_per_worker, num_gpus=self.num_gpus_per_worker, resources=self.additional_resources_per_worker)(self._base_cls)
        self.start()

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Starts all the workers in this worker group.'
        if self.workers and len(self.workers) > 0:
            raise RuntimeError('The workers have already been started. Please call `shutdown` first if you want to restart them.')
        logger.debug(f'Starting {self.num_workers} workers.')
        self.add_workers(self.num_workers)
        logger.debug(f'{len(self.workers)} workers have successfully started.')

    def shutdown(self, patience_s: float=5):
        if False:
            for i in range(10):
                print('nop')
        'Shutdown all the workers in this worker group.\n\n        Args:\n            patience_s: Attempt a graceful shutdown\n                of the workers for this many seconds. Fallback to force kill\n                if graceful shutdown is not complete after this time. If\n                this is less than or equal to 0, immediately force kill all\n                workers.\n        '
        logger.debug(f'Shutting down {len(self.workers)} workers.')
        if patience_s <= 0:
            for worker in self.workers:
                ray.kill(worker.actor)
        else:
            done_refs = [w.actor.__ray_terminate__.remote() for w in self.workers]
            (done, not_done) = ray.wait(done_refs, timeout=patience_s)
            if not_done:
                logger.debug('Graceful termination failed. Falling back to force kill.')
                for worker in self.workers:
                    ray.kill(worker.actor)
        logger.debug('Shutdown successful.')
        self.workers = []

    def execute_async(self, func: Callable[..., T], *args, **kwargs) -> List[ObjectRef]:
        if False:
            for i in range(10):
                print('nop')
        'Execute ``func`` on each worker and return the futures.\n\n        Args:\n            func: A function to call on each worker.\n            args, kwargs: Passed directly into func.\n\n        Returns:\n            (List[ObjectRef]) A list of ``ObjectRef`` representing the\n                output of ``func`` from each worker. The order is the same\n                as ``self.workers``.\n\n        '
        if len(self.workers) <= 0:
            raise RuntimeError('There are no active workers. This worker group has most likely been shut down. Pleasecreate a new WorkerGroup or restart this one.')
        return [w.actor._RayTrainWorker__execute.options(name=f'_RayTrainWorker__execute.{func.__name__}').remote(func, *args, **kwargs) for w in self.workers]

    def execute(self, func: Callable[..., T], *args, **kwargs) -> List[T]:
        if False:
            for i in range(10):
                print('nop')
        'Execute ``func`` on each worker and return the outputs of ``func``.\n\n        Args:\n            func: A function to call on each worker.\n            args, kwargs: Passed directly into func.\n\n        Returns:\n            (List[T]) A list containing the output of ``func`` from each\n                worker. The order is the same as ``self.workers``.\n\n        '
        return ray.get(self.execute_async(func, *args, **kwargs))

    def execute_single_async(self, worker_index: int, func: Callable[..., T], *args, **kwargs) -> ObjectRef:
        if False:
            for i in range(10):
                print('nop')
        'Execute ``func`` on worker ``worker_index`` and return futures.\n\n        Args:\n            worker_index: The index to execute func on.\n            func: A function to call on the first worker.\n            args, kwargs: Passed directly into func.\n\n        Returns:\n            (ObjectRef) An ObjectRef representing the output of func.\n\n        '
        if worker_index >= len(self.workers):
            raise ValueError(f'The provided worker_index {worker_index} is not valid for {self.num_workers} workers.')
        return self.workers[worker_index].actor._RayTrainWorker__execute.options(name=f'_RayTrainWorker__execute.{func.__name__}').remote(func, *args, **kwargs)

    def execute_single(self, worker_index: int, func: Callable[..., T], *args, **kwargs) -> T:
        if False:
            print('Hello World!')
        'Execute ``func`` on worker with index ``worker_index``.\n\n        Args:\n            worker_index: The index to execute func on.\n            func: A function to call on the first worker.\n            args, kwargs: Passed directly into func.\n\n        Returns:\n            (T) The output of func.\n\n        '
        return ray.get(self.execute_single_async(worker_index, func, *args, **kwargs))

    def remove_workers(self, worker_indexes: List[int]):
        if False:
            print('Hello World!')
        'Removes the workers with the specified indexes.\n\n        The removed workers will go out of scope and their actor processes\n        will be terminated.\n\n        Args:\n            worker_indexes (List[int]): The indexes of the workers to remove.\n        '
        new_workers = []
        for i in range(len(self.workers)):
            if i not in worker_indexes:
                new_workers.append(self.workers[i])
        self.workers = new_workers

    def add_workers(self, num_workers: int):
        if False:
            i = 10
            return i + 15
        'Adds ``num_workers`` to this WorkerGroup.\n\n        Note: Adding workers when the cluster/placement group is at capacity\n        may lead to undefined hanging behavior. If you are attempting to\n        replace existing workers in the WorkerGroup, remove_workers() should\n        be called first.\n\n        Args:\n            num_workers: The number of workers to add.\n        '
        new_actors = []
        new_actor_metadata = []
        for _ in range(num_workers):
            actor = self._remote_cls.options(placement_group=self._placement_group).remote(*self._actor_cls_args, **self._actor_cls_kwargs)
            new_actors.append(actor)
            new_actor_metadata.append(actor._RayTrainWorker__execute.options(name='_RayTrainWorker__execute.construct_metadata').remote(construct_metadata))
        metadata = ray.get(new_actor_metadata)
        for i in range(len(new_actors)):
            self.workers.append(Worker(actor=new_actors[i], metadata=metadata[i]))

    def group_workers_by_ip(self, _first_ip: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        'Groups workers by IP.\n\n        This is useful for collocating workers on the same node.\n\n        Args:\n            _first_ip: The first IP to group by.\n                Hack to avoid OOMs.\n                This is just a temporary solution for Train loading entire checkpoints\n                into memory by ensuring that the rank 0 worker is on the same node as\n                trainable, thus allowing for lazy checkpoint transfer to be used.\n                See https://github.com/ray-project/ray/issues/33073\n                for more context.\n                TODO remove this argument.\n        '
        ip_to_workers = defaultdict(list)
        if _first_ip is not None:
            ip_to_workers[_first_ip] = []
        for worker in self.workers:
            ip_to_workers[worker.metadata.node_ip].append(worker)
        sorted_workers = []
        for workers in ip_to_workers.values():
            sorted_workers.extend(workers)
        self.workers = sorted_workers

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.workers)