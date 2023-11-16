from collections import defaultdict, deque
from functools import partial
import pathlib
from typing import Any, Callable, List, Mapping, Optional, Set, Type, TYPE_CHECKING, Union
import uuid
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.rl_module.rl_module import ModuleID, SingleAgentRLModuleSpec, RLMODULE_STATE_DIR_NAME
from ray.rllib.core.learner.learner import LearnerSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.minibatch_utils import ShardBatchIterator
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.train._internal.backend_executor import BackendExecutor
from ray.tune.utils.file_transfer import sync_dir_between_nodes
if TYPE_CHECKING:
    from ray.rllib.core.learner.learner import Learner

def _get_backend_config(learner_class: Type['Learner']) -> str:
    if False:
        return 10
    if learner_class.framework == 'torch':
        from ray.train.torch import TorchConfig
        backend_config = TorchConfig()
    elif learner_class.framework == 'tf2':
        from ray.train.tensorflow import TensorflowConfig
        backend_config = TensorflowConfig()
    else:
        raise ValueError('framework must be either torch or tf')
    return backend_config

def _is_module_trainable(module_id: ModuleID, batch: MultiAgentBatch) -> bool:
    if False:
        i = 10
        return i + 15
    'Default implemntation for is_module_trainable()\n\n    It assumes that the module is trainable by default.\n    '
    return True

class LearnerGroup:
    """Coordinator of Learners.

    Args:
        learner_spec: The specification for constructing Learners.
        max_queue_len: The maximum number of batches to queue up if doing async_update
            If the queue is full itwill evict the oldest batch first.

    """

    def __init__(self, learner_spec: LearnerSpec, max_queue_len: int=20):
        if False:
            while True:
                i = 10
        scaling_config = learner_spec.learner_group_scaling_config
        learner_class = learner_spec.learner_class
        self._is_local = scaling_config.num_workers == 0
        self._learner = None
        self._workers = None
        self._is_shut_down = False
        self._is_module_trainable = _is_module_trainable
        self._in_queue_ts_dropped = 0
        if self._is_local:
            self._learner = learner_class(**learner_spec.get_params_dict())
            self._learner.build()
            self._worker_manager = None
            self._in_queue = []
        else:
            backend_config = _get_backend_config(learner_class)
            backend_executor = BackendExecutor(backend_config=backend_config, num_workers=scaling_config.num_workers, num_cpus_per_worker=scaling_config.num_cpus_per_worker, num_gpus_per_worker=scaling_config.num_gpus_per_worker, max_retries=0)
            backend_executor.start(train_cls=learner_class, train_cls_kwargs=learner_spec.get_params_dict())
            self._backend_executor = backend_executor
            self._workers = [w.actor for w in backend_executor.worker_group.workers]
            ray.get([w.build.remote() for w in self._workers])
            self._worker_manager = FaultTolerantActorManager(self._workers, max_remote_requests_in_flight_per_actor=3)
            self._inflight_request_tags: Set[str] = set()
            self._in_queue = deque(maxlen=max_queue_len)

    def get_in_queue_stats(self) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        'Returns the current stats for the input queue for this learner group.'
        return {'learner_group_queue_size': len(self._in_queue), 'learner_group_queue_ts_dropped': self._in_queue_ts_dropped}

    @property
    def is_local(self) -> bool:
        if False:
            print('Hello World!')
        return self._is_local

    def update(self, batch: MultiAgentBatch, *, minibatch_size: Optional[int]=None, num_iters: int=1, reduce_fn: Optional[Callable[[List[Mapping[str, Any]]], ResultDict]]=_reduce_mean_results) -> Union[Mapping[str, Any], List[Mapping[str, Any]]]:
        if False:
            print('Hello World!')
        "Do one or more gradient based updates to the Learner(s) based on given data.\n\n        Args:\n            batch: The data batch to use for the update.\n            minibatch_size: The minibatch size to use for the update.\n            num_iters: The number of complete passes over all the sub-batches in the\n                input multi-agent batch.\n            reduce_fn: An optional callable to reduce the results from a list of the\n                Learner actors into a single result. This can be any arbitrary function\n                that takes a list of dictionaries and returns a single dictionary. For\n                example you can either take an average (default) or concatenate the\n                results (for example for metrics) or be more selective about you want to\n                report back to the algorithm's training_step. If None is passed, the\n                results will not get reduced.\n\n        Returns:\n            A dictionary with the reduced results of the updates from the Learner(s) or\n            a list of dictionaries of results from the updates from the Learner(s).\n        "
        train_batch = {}
        for module_id in batch.policy_batches.keys():
            if self._is_module_trainable(module_id, batch):
                train_batch[module_id] = batch.policy_batches[module_id]
        train_batch = MultiAgentBatch(train_batch, batch.count)
        if self.is_local:
            results = [self._learner.update(train_batch, minibatch_size=minibatch_size, num_iters=num_iters, reduce_fn=reduce_fn)]
        else:

            def _learner_update(learner, minibatch):
                if False:
                    for i in range(10):
                        print('nop')
                return learner.update(minibatch, minibatch_size=minibatch_size, num_iters=num_iters, reduce_fn=reduce_fn)
            results = self._get_results(self._worker_manager.foreach_actor([partial(_learner_update, minibatch=minibatch) for minibatch in ShardBatchIterator(batch, len(self._workers))]))
        if reduce_fn is None:
            return results
        else:
            return reduce_fn(results)

    def async_update(self, batch: MultiAgentBatch, *, minibatch_size: Optional[int]=None, num_iters: int=1, reduce_fn: Optional[Callable[[List[Mapping[str, Any]]], ResultDict]]=_reduce_mean_results) -> Union[List[Mapping[str, Any]], List[List[Mapping[str, Any]]]]:
        if False:
            for i in range(10):
                print('nop')
        "Asnychronously do gradient based updates to the Learner(s) with `batch`.\n\n        Args:\n            batch: The data batch to use for the update.\n            minibatch_size: The minibatch size to use for the update.\n            num_iters: The number of complete passes over all the sub-batches in the\n                input multi-agent batch.\n            reduce_fn: An optional callable to reduce the results from a list of the\n                Learner actors into a single result. This can be any arbitrary function\n                that takes a list of dictionaries and returns a single dictionary. For\n                example you can either take an average (default) or concatenate the\n                results (for example for metrics) or be more selective about you want to\n                report back to the algorithm's training_step. If None is passed, the\n                results will not get reduced.\n\n        Returns:\n            A list of list of dictionaries of results, where the outer list\n            corresponds to separate calls to `async_update`, and the inner\n            list corresponds to the results from each Learner(s). Or if the results\n            are reduced, a list of dictionaries of the reduced results from each\n            call to async_update that is ready.\n        "
        if self.is_local:
            raise ValueError('Cannot call `async_update` when running in local mode with num_workers=0.')
        else:
            if minibatch_size is not None:
                minibatch_size //= len(self._workers)

            def _learner_update(learner, minibatch):
                if False:
                    print('Hello World!')
                return learner.update(minibatch, minibatch_size=minibatch_size, num_iters=num_iters, reduce_fn=reduce_fn)
            if len(self._in_queue) == self._in_queue.maxlen:
                self._in_queue_ts_dropped += len(self._in_queue[0])
            self._in_queue.append(batch)
            results = self._worker_manager.fetch_ready_async_reqs(tags=list(self._inflight_request_tags))
            if self._worker_manager_ready():
                count = 0
                while len(self._in_queue) > 0 and count < 3:
                    update_tag = str(uuid.uuid4())
                    self._inflight_request_tags.add(update_tag)
                    batch = self._in_queue.popleft()
                    self._worker_manager.foreach_actor_async([partial(_learner_update, minibatch=minibatch) for minibatch in ShardBatchIterator(batch, len(self._workers))], tag=update_tag)
                    count += 1
            results = self._get_async_results(results)
            if reduce_fn is None:
                return results
            else:
                return [reduce_fn(r) for r in results]

    def _worker_manager_ready(self):
        if False:
            while True:
                i = 10
        return self._worker_manager.num_outstanding_async_reqs() <= self._worker_manager.num_actors() * 2

    def _get_results(self, results):
        if False:
            while True:
                i = 10
        processed_results = []
        for result in results:
            result_or_error = result.get()
            if result.ok:
                processed_results.append(result_or_error)
            else:
                raise result_or_error
        return processed_results

    def _get_async_results(self, results):
        if False:
            print('Hello World!')
        'Get results from the worker manager and group them by tag.\n\n        Returns:\n            A list of lists of results, where each inner list contains all results\n            for same tags.\n\n        '
        unprocessed_results = defaultdict(list)
        for result in results:
            result_or_error = result.get()
            if result.ok:
                assert result.tag, 'Cannot call _get_async_results on untagged async requests.'
                unprocessed_results[result.tag].append(result_or_error)
            else:
                raise result_or_error
        for tag in unprocessed_results.keys():
            self._inflight_request_tags.remove(tag)
        return list(unprocessed_results.values())

    def additional_update(self, *, reduce_fn: Callable[[ResultDict], ResultDict]=_reduce_mean_results, **kwargs) -> Union[Mapping[str, Any], List[Mapping[str, Any]]]:
        if False:
            i = 10
            return i + 15
        'Apply additional non-gradient based updates to the Learners.\n\n        For example, this could be used to do a polyak averaging update\n        of a target network in off policy algorithms like SAC or DQN.\n\n        By default this is a pass through that calls `Learner.additional_update`\n\n        Args:\n            reduce_fn: See `update()` documentation for more details.\n            **kwargs: Keyword arguments to pass to each Learner.\n\n        Returns:\n            A list of dictionaries of results from the updates from each worker.\n        '
        if self.is_local:
            return self._learner.additional_update(**kwargs)
        else:
            results = self._worker_manager.foreach_actor([lambda w: w.additional_update(**kwargs) for _ in self._workers])
            results = self._get_results(results)
            if reduce_fn is None:
                return results
            return reduce_fn(results)

    def add_module(self, *, module_id: ModuleID, module_spec: SingleAgentRLModuleSpec) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a module to the Learners maintained by this LearnerGroup.\n\n        Args:\n            module_id: The id of the module to add.\n            module_spec:  #TODO (Kourosh) fill in here.\n        '
        if self.is_local:
            self._learner.add_module(module_id=module_id, module_spec=module_spec)
        else:
            results = self._worker_manager.foreach_actor(lambda w: w.add_module(module_id=module_id, module_spec=module_spec))
            return self._get_results(results)

    def remove_module(self, module_id: ModuleID) -> None:
        if False:
            print('Hello World!')
        'Remove a module from the Learners maintained by this LearnerGroup.\n\n        Args:\n            module_id: The id of the module to remove.\n\n        '
        if self.is_local:
            self._learner.remove_module(module_id)
        else:
            refs = []
            for worker in self._workers:
                ref = worker.remove_module.remote(module_id)
                refs.append(ref)
            ray.get(refs)

    def set_weights(self, weights: Mapping[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Set the weights of the MultiAgentRLModule maintained by each Learner.\n\n        The weights don't have to include all the modules in the MARLModule.\n            This way the weights of only some of the Agents can be set.\n\n        Args:\n            weights: The weights to set each RLModule in the MARLModule to.\n\n        "
        if self.is_local:
            self._learner.set_module_state(weights)
        else:
            results_or_errors = self._worker_manager.foreach_actor(lambda w: w.set_module_state(weights))
            self._get_results(results_or_errors)

    def get_weights(self, module_ids: Optional[Set[str]]=None) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        'Get the weights of the MultiAgentRLModule maintained by each Learner.\n\n        Args:\n            module_ids: The ids of the modules to get the weights of.\n\n        Returns:\n            A mapping of module ids to their weights.\n\n        '
        if self.is_local:
            state = self._learner.get_module_state(module_ids)
        else:
            worker = self._worker_manager.healthy_actor_ids()[0]
            assert len(self._workers) == self._worker_manager.num_healthy_actors()
            state = self._worker_manager.foreach_actor(lambda w: w.get_module_state(module_ids), remote_actor_ids=[worker])
            state = self._get_results(state)[0]
        return convert_to_numpy(state)

    def get_state(self) -> Mapping[ModuleID, Mapping[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Get the states of the first Learners.\n\n        This should be the same across Learners\n        '
        if self.is_local:
            return self._learner.get_state()
        else:
            worker = self._worker_manager.healthy_actor_ids()[0]
            assert len(self._workers) == self._worker_manager.num_healthy_actors()
            results = self._worker_manager.foreach_actor(lambda w: w.get_state(), remote_actor_ids=[worker])
            return self._get_results(results)[0]

    def set_state(self, state: List[Mapping[ModuleID, Mapping[str, Any]]]) -> None:
        if False:
            print('Hello World!')
        'Sets the states of the Learners.\n\n        Args:\n            state: The state of the Learners\n\n        '
        if self.is_local:
            self._learner.set_state(state)
        else:
            self._worker_manager.foreach_actor(lambda w: w.set_state(state))

    def set_is_module_trainable(self, is_module_trainable: Callable[[ModuleID, MultiAgentBatch], bool]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Sets the function that determines whether a module is trainable.\n\n        Args:\n            is_module_trainable: A function that takes in a module id and a batch\n                and returns a boolean indicating whether the module should be trained\n                on the batch.\n        '
        if is_module_trainable is not None:
            self._is_module_trainable = is_module_trainable

    def save_state(self, path: str) -> None:
        if False:
            i = 10
            return i + 15
        'Saves the state of the LearnerGroup.\n\n        Args:\n            path: The path to save the state to.\n        '
        if self.is_local:
            self._learner.save_state(path)
        else:
            worker = self._worker_manager.healthy_actor_ids()[0]
            worker_ip_addr = self._worker_manager.foreach_actor(self._get_ip_address, remote_actor_ids=[worker])
            worker_ip_addr = self._get_results(worker_ip_addr)[0]
            self_ip_addr = self._get_ip_address()
            if worker_ip_addr == self_ip_addr:
                self._worker_manager.foreach_actor(lambda w: w.save_state(path), remote_actor_ids=[worker])
            else:
                worker_temp_dir = self._worker_manager.foreach_actor(self._create_temporary_dir, remote_actor_ids=[worker])
                worker_temp_dir = self._get_results(worker_temp_dir)[0]
                self._worker_manager.foreach_actor(lambda w: w.save_state(worker_temp_dir), remote_actor_ids=[worker])
                sync_dir_between_nodes(worker_ip_addr, worker_temp_dir, self_ip_addr, path)

                def remove_dir(w):
                    if False:
                        return 10
                    import shutil
                    shutil.rmtree(worker_temp_dir)
                self._worker_manager.foreach_actor(remove_dir, remote_actor_ids=[worker])

    def load_state(self, path: str) -> None:
        if False:
            print('Hello World!')
        'Loads the state of the LearnerGroup.\n\n        Args:\n            path: The path to load the state from.\n        '
        path = str(self._resolve_checkpoint_path(path))
        if self.is_local:
            self._learner.load_state(path)
        else:
            assert len(self._workers) == self._worker_manager.num_healthy_actors()
            head_node_ip = ray.util.get_node_ip_address()
            workers = self._worker_manager.healthy_actor_ids()

            def _load_state(w):
                if False:
                    while True:
                        i = 10
                import ray
                import tempfile
                worker_node_ip = ray.util.get_node_ip_address()
                if worker_node_ip == head_node_ip:
                    w.load_state(path)
                else:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        sync_dir_between_nodes(head_node_ip, path, worker_node_ip, temp_dir)
                        w.load_state(temp_dir)
            self._worker_manager.foreach_actor(_load_state, remote_actor_ids=workers)

    def load_module_state(self, *, marl_module_ckpt_dir: Optional[str]=None, modules_to_load: Optional[Set[str]]=None, rl_module_ckpt_dirs: Optional[Mapping[ModuleID, str]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Load the checkpoints of the modules being trained by this LearnerGroup.\n\n        `load_module_state` can be used 3 ways:\n            1. Load a checkpoint for the MultiAgentRLModule being trained by this\n                LearnerGroup. Limit the modules that are loaded from the checkpoint\n                by specifying the `modules_to_load` argument.\n            2. Load the checkpoint(s) for single agent RLModules that\n                are in the MultiAgentRLModule being trained by this LearnerGroup.\n            3. Load a checkpoint for the MultiAgentRLModule being trained by this\n                LearnerGroup and load the checkpoint(s) for single agent RLModules\n                that are in the MultiAgentRLModule. The checkpoints for the single\n                agent RLModules take precedence over the module states in the\n                MultiAgentRLModule checkpoint.\n\n        NOTE: At lease one of marl_module_ckpt_dir or rl_module_ckpt_dirs is\n            must be specified. modules_to_load can only be specified if\n            marl_module_ckpt_dir is specified.\n\n        Args:\n            marl_module_ckpt_dir: The path to the checkpoint for the\n                MultiAgentRLModule.\n            modules_to_load: A set of module ids to load from the checkpoint.\n            rl_module_ckpt_dirs: A mapping from module ids to the path to a\n                checkpoint for a single agent RLModule.\n        '
        if not (marl_module_ckpt_dir or rl_module_ckpt_dirs):
            raise ValueError('At least one of multi_agent_module_state or single_agent_module_states must be specified.')
        if marl_module_ckpt_dir:
            if not isinstance(marl_module_ckpt_dir, str):
                raise ValueError('multi_agent_module_state must be a string path.')
            marl_module_ckpt_dir = self._resolve_checkpoint_path(marl_module_ckpt_dir)
        if rl_module_ckpt_dirs:
            if not isinstance(rl_module_ckpt_dirs, dict):
                raise ValueError('single_agent_module_states must be a dictionary.')
            for (module_id, path) in rl_module_ckpt_dirs.items():
                if not isinstance(path, str):
                    raise ValueError('rl_module_ckpt_dirs must be a dictionary mapping module ids to string paths.')
                rl_module_ckpt_dirs[module_id] = self._resolve_checkpoint_path(path)
        if modules_to_load:
            if not isinstance(modules_to_load, set):
                raise ValueError('modules_to_load must be a set.')
            for module_id in modules_to_load:
                if not isinstance(module_id, str):
                    raise ValueError('modules_to_load must be a list of strings.')
        if self.is_local:
            module_keys = set(self._learner.module.keys())
        else:
            workers = self._worker_manager.healthy_actor_ids()
            module_keys = set(self._get_results(self._worker_manager.foreach_actor(lambda w: w.module.keys(), remote_actor_ids=[workers[0]]))[0])
        if marl_module_ckpt_dir and rl_module_ckpt_dirs:
            if modules_to_load:
                if any((module_id in modules_to_load for module_id in rl_module_ckpt_dirs.keys())):
                    raise ValueError(f'module_id {module_id} was specified in both modules_to_load and rl_module_ckpt_dirs. Please only specify a module to be loaded only once, either in modules_to_load or rl_module_ckpt_dirs, but not both.')
            else:
                modules_to_load = module_keys - set(rl_module_ckpt_dirs.keys())
        if self._is_local:
            if marl_module_ckpt_dir:
                self._learner.module.load_state(marl_module_ckpt_dir, modules_to_load=modules_to_load)
            if rl_module_ckpt_dirs:
                for (module_id, path) in rl_module_ckpt_dirs.items():
                    self._learner.module[module_id].load_state(path / RLMODULE_STATE_DIR_NAME)
        else:
            self._distributed_load_module_state(marl_module_ckpt_dir=marl_module_ckpt_dir, modules_to_load=modules_to_load, rl_module_ckpt_dirs=rl_module_ckpt_dirs)

    def _distributed_load_module_state(self, *, marl_module_ckpt_dir: Optional[str]=None, modules_to_load: Optional[Set[str]]=None, rl_module_ckpt_dirs: Optional[Mapping[ModuleID, str]]=None):
        if False:
            return 10
        'Load the checkpoints of the modules being trained by this LearnerGroup.\n\n           This method only needs to be called if the LearnerGroup is training\n           distributed learners (e.g num_learner_workers > 0).\n\n        Args:\n            marl_module_ckpt_dir: The path to the checkpoint for the\n                MultiAgentRLModule.\n            modules_to_load: A set of module ids to load from the checkpoint.\n            rl_module_ckpt_dirs: A mapping from module ids to the path to a\n                checkpoint for a single agent RLModule.\n\n        '
        assert len(self._workers) == self._worker_manager.num_healthy_actors()
        workers = self._worker_manager.healthy_actor_ids()
        head_node_ip = ray.util.get_node_ip_address()

        def _load_module_state(w):
            if False:
                while True:
                    i = 10
            import ray
            import tempfile
            import shutil
            worker_node_ip = ray.util.get_node_ip_address()
            tmp_marl_module_ckpt_dir = marl_module_ckpt_dir
            tmp_rl_module_ckpt_dirs = rl_module_ckpt_dirs
            if worker_node_ip != head_node_ip:
                if marl_module_ckpt_dir:
                    tmp_marl_module_ckpt_dir = tempfile.mkdtemp()
                    sync_dir_between_nodes(source_ip=head_node_ip, source_path=marl_module_ckpt_dir, target_ip=worker_node_ip, target_path=tmp_marl_module_ckpt_dir)
                if rl_module_ckpt_dirs:
                    tmp_rl_module_ckpt_dirs = {}
                    for (module_id, path) in rl_module_ckpt_dirs.items():
                        tmp_rl_module_ckpt_dirs[module_id] = tempfile.mkdtemp()
                        sync_dir_between_nodes(source_ip=head_node_ip, source_path=path, target_ip=worker_node_ip, target_path=tmp_rl_module_ckpt_dirs[module_id])
                        tmp_rl_module_ckpt_dirs[module_id] = pathlib.Path(tmp_rl_module_ckpt_dirs[module_id])
            if marl_module_ckpt_dir:
                w.module.load_state(tmp_marl_module_ckpt_dir, modules_to_load=modules_to_load)
            if rl_module_ckpt_dirs:
                for (module_id, path) in tmp_rl_module_ckpt_dirs.items():
                    w.module[module_id].load_state(path / RLMODULE_STATE_DIR_NAME)
            if worker_node_ip != head_node_ip:
                if marl_module_ckpt_dir:
                    shutil.rmtree(tmp_marl_module_ckpt_dir)
                if rl_module_ckpt_dirs:
                    for (module_id, path) in tmp_rl_module_ckpt_dirs.items():
                        shutil.rmtree(path)
        self._worker_manager.foreach_actor(_load_module_state, remote_actor_ids=workers)

    @staticmethod
    def _resolve_checkpoint_path(path: str) -> pathlib.Path:
        if False:
            for i in range(10):
                print('nop')
        'Checks that the provided checkpoint path is a dir and makes it absolute.'
        path = pathlib.Path(path)
        if not path.is_dir():
            raise ValueError(f'Path {path} is not a directory. Please specify a directory containing the checkpoint files.')
        if not path.exists():
            raise ValueError(f'Path {path} does not exist.')
        path = path.absolute()
        return path

    @staticmethod
    def _create_temporary_dir(_=None) -> str:
        if False:
            return 10
        'Creates a temporary directory.\n\n        Args:\n            _: Unused arg. Exists to make this function compatible with foreach_actor\n            calls.\n\n        Returns:\n            The path to the temporary directory.\n        '
        import tempfile
        return tempfile.mkdtemp()

    @staticmethod
    def _get_ip_address(_=None) -> str:
        if False:
            i = 10
            return i + 15
        "Returns this process's address.\n\n        Args:\n            _: Unused arg. Exists to make this function compatible with foreach_actor\n            calls.\n\n        Returns:\n            The address of this process.\n\n        "
        import ray
        return ray.util.get_node_ip_address()

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        'Shuts down the LearnerGroup.'
        if not self._is_local:
            self._backend_executor.shutdown()
            self._is_shut_down = True

    def __del__(self):
        if False:
            i = 10
            return i + 15
        if not self._is_shut_down:
            self.shutdown()