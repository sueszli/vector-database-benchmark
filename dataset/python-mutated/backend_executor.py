import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import ray
import ray._private.ray_constants as ray_constants
from ray._private.ray_constants import env_integer
from ray.data import Dataset
from ray.exceptions import RayActorError
from ray.train import Checkpoint, DataConfig
from ray.train._internal.session import TrialInfo, _TrainingResult, get_session, init_session, shutdown_session
from ray.train._internal.storage import StorageContext
from ray.train._internal.utils import check_for_failure
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.constants import ENABLE_DETAILED_AUTOFILLED_METRICS_ENV, ENABLE_SHARE_CUDA_VISIBLE_DEVICES_ENV, ENABLE_SHARE_NEURON_CORES_ACCELERATOR_ENV, TRAIN_ENABLE_WORKER_SPREAD_ENV, TRAIN_PLACEMENT_GROUP_TIMEOUT_S_ENV
from ray.util.placement_group import get_current_placement_group, remove_placement_group
T = TypeVar('T')
logger = logging.getLogger(__name__)

class TrainBackendError(Exception):
    """Errors with BackendExecutor that should not be exposed to user."""

class TrainingWorkerError(Exception):
    """Raised if a worker fails during training."""

@dataclass
class ResourceConfig:
    """
    Resource configuration for resource_ids to share between workers.

    Args:
        resource_name: The name of the resource to configure
         (Example: "neuron_cores" or "gpu").
        resource_enable_sharing_env_var: The environment variable to
         check if the resource should be shared.
        share_resource_ids_env_var: The environment variable to configure for
         sharing the resources with other workers.
    """
    resource_name: str
    resource_enable_sharing_env_var: str
    share_resource_ids_env_var: str

class BackendExecutor:
    """Main execution class for training backends.

    This class holds a worker group and is responsible for executing the
    training function on the workers, and collecting intermediate results
    from ``session.report()``.

    Args:
        backend_config: The configurations for this
            specific backend.
        num_workers: Number of workers to use for training.
        num_cpus_per_worker: Number of CPUs to use per worker.
        num_gpus_per_worker: Number of GPUs to use per worker.
        additional_resources_per_worker (Optional[Dict[str, float]]):
            Dictionary specifying the extra resources that will be
            requested for each worker in addition to ``num_cpus_per_worker``
            and ``num_gpus_per_worker``.
        max_retries: Number of retries when Ray actors fail.
            Defaults to 3. Set to -1 for unlimited retries.
    """

    def __init__(self, backend_config: BackendConfig, trial_info: Optional[TrialInfo]=None, num_workers: int=1, num_cpus_per_worker: float=1, num_gpus_per_worker: float=0, additional_resources_per_worker: Optional[Dict[str, float]]=None, max_retries: int=3):
        if False:
            i = 10
            return i + 15
        self._backend_config = backend_config
        self._backend = backend_config.backend_cls()
        self._num_workers = num_workers
        self._num_cpus_per_worker = num_cpus_per_worker
        self._num_gpus_per_worker = num_gpus_per_worker
        self._additional_resources_per_worker = additional_resources_per_worker
        self._max_failures = max_retries
        if self._max_failures < 0:
            self._max_failures = float('inf')
        self._num_failures = 0
        self._last_failure = None
        self._initialization_hook = None
        self._placement_group = None
        self._trial_info = trial_info
        self.worker_group = InactiveWorkerGroup()
        self.dataset_shards = None
        self._resource_configs = [ResourceConfig(ray_constants.NEURON_CORES, ENABLE_SHARE_NEURON_CORES_ACCELERATOR_ENV, ray_constants.NEURON_RT_VISIBLE_CORES_ENV_VAR)]

    def start(self, initialization_hook: Optional[Callable[[], None]]=None, train_cls: Optional[Type]=None, train_cls_args: Optional[Tuple]=None, train_cls_kwargs: Optional[Dict]=None):
        if False:
            for i in range(10):
                print('nop')
        'Starts the worker group.'
        self._create_placement_group()
        placement_group = self._placement_group or 'default'
        self.worker_group = WorkerGroup(num_workers=self._num_workers, num_cpus_per_worker=self._num_cpus_per_worker, num_gpus_per_worker=self._num_gpus_per_worker, additional_resources_per_worker=self._additional_resources_per_worker, actor_cls=train_cls, actor_cls_args=train_cls_args, actor_cls_kwargs=train_cls_kwargs, placement_group=placement_group)
        trial_driver_ip = self._trial_info.driver_ip if self._trial_info else None
        self.worker_group.group_workers_by_ip(trial_driver_ip)
        try:
            if initialization_hook:
                self._initialization_hook = initialization_hook
                self.worker_group.execute(initialization_hook)
            from ray.data import DataContext

            def _set_driver_dataset_context(ctx: DataContext):
                if False:
                    i = 10
                    return i + 15
                DataContext._set_current(ctx)
            self.worker_group.execute(_set_driver_dataset_context, DataContext.get_current())
            share_cuda_visible_devices_enabled = bool(env_integer(ENABLE_SHARE_CUDA_VISIBLE_DEVICES_ENV, self._backend.share_cuda_visible_devices))
            if self._num_gpus_per_worker > 0 and share_cuda_visible_devices_enabled:
                self._share_cuda_visible_devices()
            elif self._additional_resources_per_worker:
                for resource_config in self._resource_configs:
                    if self._is_share_resources_enabled(resource_config.resource_name, resource_config.resource_enable_sharing_env_var):
                        self._share_resource_ids(resource_config.resource_name, resource_config.share_resource_ids_env_var)
            self._backend.on_start(self.worker_group, self._backend_config)
        except RayActorError as exc:
            logger.exception(str(exc))
            logger.warning('Failure occurred during startup. Restarting all workers and attempting to startup again.')
            self._increment_failures()
            self._restart()

    def _create_placement_group(self):
        if False:
            for i in range(10):
                print('nop')
        'Creates a placement group if it does not exist.\n\n        If a placement group is already detected (Tune) this will be a no-op.\n\n        By default the placement group will be created with PACK strategy.\n        This is optimized for colocating GPUs on a minimal number of nodes.\n        This behavior can be overridden to use the SPREAD strategy by defining\n        ``TRAIN_ENABLE_WORKER_SPREAD_ENV``\n\n        If a placement group is created it will be stored as\n        self._placement_group.\n        '
        current_placement_group = get_current_placement_group()
        worker = ray._private.worker.global_worker
        should_capture_child_tasks_in_placement_group = worker.should_capture_child_tasks_in_placement_group
        should_create_placement_group = current_placement_group is None or not should_capture_child_tasks_in_placement_group
        if should_create_placement_group:
            additional_resources_per_worker = self._additional_resources_per_worker or {}
            bundle = {'CPU': self._num_cpus_per_worker, 'GPU': self._num_gpus_per_worker, **additional_resources_per_worker}
            bundles = [bundle.copy() for _ in range(self._num_workers)]
            use_spread = bool(env_integer(TRAIN_ENABLE_WORKER_SPREAD_ENV, 0))
            strategy = 'SPREAD' if use_spread else 'PACK'
            placement_group = ray.util.placement_group(bundles, strategy=strategy)
            logger.debug('Waiting for placement group to start.')
            timeout = env_integer(TRAIN_PLACEMENT_GROUP_TIMEOUT_S_ENV, 100)
            (ready, _) = ray.wait([placement_group.ready()], timeout=timeout)
            if ready:
                logger.debug('Placement group has started.')
            else:
                raise TimeoutError('Placement group creation timed out. Make sure your cluster either has enough resources or use an autoscaling cluster. If you are running on a cluster, make sure you specify an address in `ray.init()`, for example, `ray.init("auto")`. You can also increase the timeout by setting the TRAIN_PLACEMENT_GROUP_TIMEOUT_S environment variable. Current resources available: {}, resources requested by the placement group: {}'.format(ray.available_resources(), placement_group.bundle_specs))
            self._placement_group = placement_group

    def _share_cuda_visible_devices(self):
        if False:
            while True:
                i = 10
        'Sets CUDA_VISIBLE_DEVICES on all workers.\n\n        For each worker, CUDA_VISIBLE_DEVICES will be set to the GPU IDs\n        visible to all workers on that worker\'s node.\n\n        This allows GPU workers on the same node to communicate with one\n        another.\n\n        Example:\n\n            Setup:\n            - Node1:\n                - Worker1: {0, 1}\n                - Worker2: {2, 3}\n            - Node2:\n                - Worker3: {0, 1}\n\n            CUDA_VISIBLE_DEVICES:\n            - Worker1: "0,1,2,3"\n            - Worker2: "0,1,2,3"\n            - Worker2: "0,1"\n\n        '
        self._share_resource_ids(ray_constants.GPU, ray_constants.CUDA_VISIBLE_DEVICES_ENV_VAR)

    def _share_resource_ids(self, resource: str, env_var: str):
        if False:
            while True:
                i = 10
        'Sets the given env_var on all workers.\n\n        For each worker, the cores/devices are visible to all the\n        workers on that worker\'s node.This allows workers on the\n        same node to communicate with one another.\n\n        Example:\n\n            Setup:\n            - Node1:\n                - Worker1: {0, 1}\n                - Worker2: {2, 3}\n            - Node2:\n                - Worker3: {0, 1}\n\n            NEURON_RT_VISIBLE_CORES/TPU_VISIBLE_CHIPS/...:\n            - Worker1: "0,1,2,3"\n            - Worker2: "0,1,2,3"\n            - Worker2: "0,1"\n\n        Args:\n            resource: The name of the resource/accelerator.\n            env_var: The name of the environment variable to set.\n        '
        node_ids_and_resource_ids = [(w.metadata.node_id, w.metadata.resource_ids[resource]) for w in self.worker_group.workers]
        node_id_to_worker_id = defaultdict(set)
        node_id_to_resource_ids = defaultdict(set)
        for (worker_id, (node_id, resource_ids)) in enumerate(node_ids_and_resource_ids):
            node_id_to_worker_id[node_id].add(worker_id)
            node_id_to_resource_ids[node_id].update(resource_ids)
        futures = []
        for (node_id, resource_ids) in node_id_to_resource_ids.items():
            resource_ids = sorted(resource_ids)
            all_resource_ids = ','.join(resource_ids)

            def set_resource_ids():
                if False:
                    while True:
                        i = 10
                os.environ[env_var] = all_resource_ids
            for worker_id in node_id_to_worker_id[node_id]:
                futures.append(self.worker_group.execute_single_async(worker_id, set_resource_ids))
        ray.get(futures)

    def _is_share_resources_enabled(self, resource_name: str, enable_sharing_env: str):
        if False:
            for i in range(10):
                print('nop')
        'Whether to share resource IDs on all workers\n        based on enable_sharing_env.\n\n        This will return true if resources are requested and greater than 0.\n        Also, user can disable by configuring the `enable_sharing_env` to "0".\n\n        Args:\n            resource_name: The name of the resource/accelerator.\n            enable_sharing_env: The name of the environment variable\n                to check.\n        '
        has_resource_requested = self._additional_resources_per_worker.get(resource_name, 0) > 0
        return has_resource_requested and ray_constants.env_bool(enable_sharing_env, True)

    def _create_rank_world_size_mappings(self) -> List[Dict]:
        if False:
            i = 10
            return i + 15
        'Create rank and world size mappings for workers.\n        There are three maps returned:\n            - local_rank_map, which maps from worker world_rank to local_rank.\n            - local_world_size_map, which maps from world_rank to local_world_size\n            - node_rank_map, which maps from world rank to node rank\n\n        Example:\n            Worker 0: 0.0.0.0\n            Worker 1: 0.0.0.0\n            Worker 2: 0.0.0.1\n            Worker 3: 0.0.0.0\n            Worker 4: 0.0.0.1\n\n            Workers 0, 1, 3 are on 0.0.0.0.\n            Workers 2, 4 are on 0.0.0.1.\n\n            Expected local_rank_map:\n            {\n                0 -> 0,\n                1 -> 1,\n                2 -> 0,\n                3 -> 2,\n                4 -> 1\n            }\n\n            Expected local_world_size_map:\n            {\n                0 -> 3,\n                1 -> 3,\n                2 -> 2,\n                3 -> 3,\n                4 -> 2\n            }\n\n            Expected node_rank_map:\n            {\n                0 -> 0,\n                1 -> 0,\n                2 -> 1,\n                3 -> 0,\n                4 -> 1\n            }\n\n        '
        local_rank_map = {}
        local_world_size_map = {}
        node_rank_map = {}
        node_ips = {}
        node_cnt = 0
        ip_dict = defaultdict(int)
        for world_rank in range(len(self.worker_group)):
            worker = self.worker_group.workers[world_rank]
            node_ip = worker.metadata.node_ip
            local_rank_map[world_rank] = ip_dict[node_ip]
            ip_dict[node_ip] += 1
            if node_ip not in node_ips:
                node_ips[node_ip] = node_cnt
                node_cnt += 1
            node_rank_map[world_rank] = node_ips[node_ip]
        for world_rank in range(len(self.worker_group)):
            worker = self.worker_group.workers[world_rank]
            node_ip = worker.metadata.node_ip
            local_world_size_map[world_rank] = ip_dict[node_ip]
        workers_info = '\n'.join([f'- (ip={w.metadata.node_ip}, pid={w.metadata.pid}) world_rank={i}, local_rank={local_rank_map[i]}, node_rank={node_rank_map[i]}' for (i, w) in enumerate(self.worker_group.workers)])
        logger.info(f'Started distributed worker processes: \n{workers_info}')
        return (local_rank_map, local_world_size_map, node_rank_map)

    def start_training(self, train_func: Callable[[], T], datasets: Dict[str, Dataset], metadata: Dict[str, Any], data_config: DataConfig, storage: StorageContext, checkpoint: Optional[Checkpoint]=None, on_session_init: Callable[[], None]=None) -> None:
        if False:
            while True:
                i = 10
        'Executes a training function on all workers in a separate thread.\n\n        ``finish_training`` should be called after this.\n\n        Args:\n            train_func: The training function to run on each worker.\n            datasets: The base datasets.\n            data_config: The config object for creating dataset shards for workers.\n            checkpoint: The checkpoint data that\n                should be loaded onto each worker and accessed by the\n                training function via ``session.get_checkpoint()``. If this\n                is ``None`` then no checkpoint will be loaded.\n        '
        use_detailed_autofilled_metrics = env_integer(ENABLE_DETAILED_AUTOFILLED_METRICS_ENV, 0)

        def initialize_session(train_func, world_rank, local_rank, node_rank, local_world_size, world_size, trial_info, checkpoint, dataset_shard, metadata, storage):
            if False:
                while True:
                    i = 10
            try:
                init_session(training_func=train_func, world_rank=world_rank, local_rank=local_rank, node_rank=node_rank, local_world_size=local_world_size, world_size=world_size, trial_info=trial_info, dataset_shard=dataset_shard, metadata=metadata, checkpoint=checkpoint, detailed_autofilled_metrics=use_detailed_autofilled_metrics, storage=storage)
            except ValueError:
                raise TrainBackendError('Attempting to start training but a previous training run is still ongoing. You must call `finish_training` before calling `start_training` again.')
        if self.dataset_shards is None:
            actors = [worker.actor for worker in self.worker_group.workers]
            node_ids = [worker.metadata.node_id for worker in self.worker_group.workers]
            self.dataset_shards = data_config.configure(datasets, world_size=len(self.worker_group), worker_handles=actors, worker_node_ids=node_ids)
        (local_rank_map, local_world_size_map, node_rank_map) = self._create_rank_world_size_mappings()
        futures = []
        for index in range(len(self.worker_group)):
            futures.append(self.worker_group.execute_single_async(index, initialize_session, world_rank=index, local_rank=local_rank_map[index], node_rank=node_rank_map[index], local_world_size=local_world_size_map[index], world_size=len(self.worker_group), trial_info=self._trial_info, train_func=train_func, dataset_shard=self.dataset_shards[index], metadata=metadata, checkpoint=checkpoint, storage=storage))
        self._backend.on_training_start(self.worker_group, self._backend_config)
        self.get_with_failure_handling(futures)
        if on_session_init:
            on_session_init()

        def train_async():
            if False:
                i = 10
                return i + 15
            session = get_session()
            session.start()
        self.worker_group.execute_async(train_async)

    def get_next_results(self) -> Optional[List[_TrainingResult]]:
        if False:
            return 10
        'Fetches the next ``_TrainingResult`` from each worker.\n\n        Each ``_TrainingResult`` is expected to correspond to the same step from\n        each worker (e.g. the same call to ``train.report()``).\n\n        Returns:\n            A list of ``_TrainingResult``s or ``None`` if there are no more results\n            since the training function has exited on all workers.\n        '

        def get_next():
            if False:
                print('Hello World!')
            session = _get_session('get_next_results')
            try:
                result = session.get_next()
            except RuntimeError:
                raise TrainBackendError('`get_next_results` has been called before `start_training`. Please call `start_training` before `get_next_results`.')
            return result
        futures = self.worker_group.execute_async(get_next)
        results = self.get_with_failure_handling(futures)
        if any((r is None for r in results)):
            if not all((r is None for r in results)):
                raise RuntimeError("Some workers returned results while others didn't. Make sure that `session.report()` are called the same number of times on all workers.")
            else:
                return None
        return results

    def pause_reporting(self):
        if False:
            i = 10
            return i + 15
        'Disable workers from enqueuing results from ``session.report()``.\n\n        Note: Already reported results may still be enqueued at this point,\n              and should be handled appropriately.\n        '

        def pause_session_reporting():
            if False:
                for i in range(10):
                    print('nop')
            session = _get_session('pause_reporting')
            return session.pause_reporting()
        futures = self.worker_group.execute_async(pause_session_reporting)
        self.get_with_failure_handling(futures)

    def finish_training(self):
        if False:
            print('Hello World!')
        'Finish training and return final results. Propagate any exceptions.\n\n        Blocks until training is finished on all workers.\n\n        Assumes `start_training` has already been called.\n\n        Returns:\n            A list of return values from calling ``train_func`` on each worker.\n                Each item corresponds to the return value from a single worker.\n        '

        def end_training():
            if False:
                i = 10
                return i + 15
            session = _get_session('finish_training')
            try:
                output = session.finish()
            finally:
                shutdown_session()
            return output
        futures = self.worker_group.execute_async(end_training)
        results = self.get_with_failure_handling(futures)
        return results

    def get_with_failure_handling(self, remote_values):
        if False:
            while True:
                i = 10
        'Gets the remote values while handling for worker failures.\n\n        This method should be called instead of ``ray.get()`` directly in\n        order to handle worker failures.\n\n        If a worker failure is identified, backend specific failure handling\n        is executed and a ``TrainingWorkerError`` is raised.\n\n        Args:\n            remote_values: List of object refs representing functions\n                that may fail in the middle of execution. For example, running\n                a Train training loop in multiple parallel actor calls.\n        Returns:\n            The resolved objects represented by the passed in ObjectRefs.\n        '
        (success, exception) = check_for_failure(remote_values)
        if success:
            return ray.get(remote_values)
        else:
            self._last_failure = exception
            self._increment_failures()
            logger.warning('Failure identified during training. Restarting all workers and continuing training from latest checkpoint.')
            self._restart()
            raise TrainingWorkerError

    def shutdown(self, graceful_termination: bool=True):
        if False:
            return 10
        'Shuts down the workers in the worker group.\n\n        Args:\n            graceful_termination: If set to True, attempt to clean up the backend\n                before terminating the Ray actors.\n\n        '
        if graceful_termination:
            try:
                self._backend.on_shutdown(self.worker_group, self._backend_config)
            except RayActorError:
                logger.warning('Graceful shutdown of backend failed. This is expected if one of the workers has crashed.')
        if graceful_termination:
            self.worker_group.shutdown()
        else:
            self.worker_group.shutdown(patience_s=0)
        self.worker_group = InactiveWorkerGroup()
        if self._placement_group:
            remove_placement_group(self._placement_group)
            self._placement_group = None
        self.dataset_shards = None

    def is_started(self):
        if False:
            for i in range(10):
                print('nop')
        return not isinstance(self.worker_group, InactiveWorkerGroup)

    def _restart(self):
        if False:
            i = 10
            return i + 15
        self.worker_group.shutdown()
        if self._initialization_hook is not None:
            initialization_hook = self._initialization_hook
        else:
            initialization_hook = None
        if self._placement_group:
            remove_placement_group(self._placement_group)
            self._placement_group = None
        self.start(initialization_hook=initialization_hook)

    def _increment_failures(self):
        if False:
            return 10
        self._num_failures += 1
        if self._num_failures >= self._max_failures:
            failure = self._last_failure
            self._last_failure = None
            if self._max_failures > 0:
                exc = RuntimeError(f'Training has failed after {self._num_failures} attempts.')
                raise exc.with_traceback(None) from failure
            else:
                raise failure

    def get_worker_group(self):
        if False:
            print('Hello World!')
        return self.worker_group

    def _get_num_failures(self):
        if False:
            while True:
                i = 10
        return self._num_failures

class InactiveWorkerGroupError(Exception):
    """Raised when underlying worker group is inactive."""

class InactiveWorkerGroup:

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return vars(self)

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        vars(self).update(state)

    def __getattr__(self, name):
        if False:
            return 10
        raise InactiveWorkerGroupError()

    def __len__(self):
        if False:
            return 10
        raise InactiveWorkerGroupError()

def _get_session(method_name: str):
    if False:
        for i in range(10):
            print('nop')
    session = get_session()
    if not session:
        raise TrainBackendError(f'`{method_name}` has been called before `start_training`. Please call `start_training` before `{method_name}`.')
    return session