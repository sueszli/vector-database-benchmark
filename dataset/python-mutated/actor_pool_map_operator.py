import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import ExecutionOptions, ExecutionResources, NodeIdStr, PhysicalOperator, RefBundle, TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
logger = DatasetLogger(__name__)
DEFAULT_MAX_TASKS_IN_FLIGHT = 4
DEFAULT_WAIT_FOR_MIN_ACTORS_SEC = 60 * 10

class ActorPoolMapOperator(MapOperator):
    """A MapOperator implementation that executes tasks on an actor pool.

    This class manages the state of a pool of actors used for task execution, as well
    as dispatch of tasks to those actors.

    It operates in two modes. In bulk mode, tasks are queued internally and executed
    when the operator has free actor slots. In streaming mode, the streaming executor
    only adds input when `should_add_input() = True` (i.e., there are free slots).
    This allows for better control of backpressure (e.g., suppose we go over memory
    limits after adding put, then there isn't any way to "take back" the inputs prior
    to actual execution).
    """

    def __init__(self, map_transformer: MapTransformer, input_op: PhysicalOperator, target_max_block_size: Optional[int], autoscaling_policy: 'AutoscalingPolicy', name: str='ActorPoolMap', min_rows_per_bundle: Optional[int]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        "Create an ActorPoolMapOperator instance.\n\n        Args:\n            transform_fn: The function to apply to each ref bundle input.\n            init_fn: The callable class to instantiate on each actor.\n            input_op: Operator generating input data for this op.\n            autoscaling_policy: A policy controlling when the actor pool should be\n                scaled up and scaled down.\n            name: The name of this operator.\n            target_max_block_size: The target maximum number of bytes to\n                include in an output block.\n            min_rows_per_bundle: The number of rows to gather per batch passed to the\n                transform_fn, or None to use the block size. Setting the batch size is\n                important for the performance of GPU-accelerated transform functions.\n                The actual rows passed may be less if the dataset is small.\n            ray_remote_args: Customize the ray remote args for this op's tasks.\n        "
        super().__init__(map_transformer, input_op, name, target_max_block_size, min_rows_per_bundle, ray_remote_args)
        self._ray_remote_args = self._apply_default_remote_args(self._ray_remote_args)
        self._min_rows_per_bundle = min_rows_per_bundle
        self._autoscaling_policy = autoscaling_policy
        self._actor_pool = _ActorPool(autoscaling_policy._config.max_tasks_in_flight)
        self._bundle_queue = collections.deque()
        self._cls = None
        self._inputs_done = False

    def internal_queue_size(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._bundle_queue)

    def start(self, options: ExecutionOptions):
        if False:
            return 10
        self._actor_locality_enabled = options.actor_locality_enabled
        super().start(options)
        self._cls = ray.remote(**self._ray_remote_args)(_MapWorker)
        for _ in range(self._autoscaling_policy.min_workers):
            self._start_actor()
        refs = self._actor_pool.get_pending_actor_refs()
        logger.get_logger().info(f'{self._name}: Waiting for {len(refs)} pool actors to start...')
        try:
            ray.get(refs, timeout=DEFAULT_WAIT_FOR_MIN_ACTORS_SEC)
        except ray.exceptions.GetTimeoutError:
            raise ray.exceptions.GetTimeoutError('Timed out while starting actors. This may mean that the cluster does not have enough resources for the requested actor pool.')

    def should_add_input(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._actor_pool.num_free_slots() > 0

    def notify_resource_usage(self, input_queue_size: int, under_resource_limits: bool) -> None:
        if False:
            while True:
                i = 10
        free_slots = self._actor_pool.num_free_slots()
        if input_queue_size > free_slots and under_resource_limits:
            self._scale_up_if_needed()
        else:
            self._scale_down_if_needed()

    def _start_actor(self):
        if False:
            i = 10
            return i + 15
        'Start a new actor and add it to the actor pool as a pending actor.'
        assert self._cls is not None
        ctx = DataContext.get_current()
        actor = self._cls.remote(ctx, src_fn_name=self.name, map_transformer=self._map_transformer)
        res_ref = actor.get_location.remote()

        def _task_done_callback(res_ref):
            if False:
                print('Hello World!')
            has_actor = self._actor_pool.pending_to_running(res_ref)
            if not has_actor:
                return
            self._dispatch_tasks()
        self._submit_metadata_task(res_ref, lambda : _task_done_callback(res_ref))
        self._actor_pool.add_pending_actor(actor, res_ref)

    def _add_bundled_input(self, bundle: RefBundle):
        if False:
            print('Hello World!')
        self._bundle_queue.append(bundle)
        self._dispatch_tasks()

    def _dispatch_tasks(self):
        if False:
            return 10
        'Try to dispatch tasks from the bundle buffer to the actor pool.\n\n        This is called when:\n            * a new input bundle is added,\n            * a task finishes,\n            * a new worker has been created.\n        '
        while self._bundle_queue:
            if self._actor_locality_enabled:
                actor = self._actor_pool.pick_actor(self._bundle_queue[0])
            else:
                actor = self._actor_pool.pick_actor()
            if actor is None:
                break
            bundle = self._bundle_queue.popleft()
            input_blocks = [block for (block, _) in bundle.blocks]
            ctx = TaskContext(task_idx=self._next_data_task_idx, target_max_block_size=self.actual_target_max_block_size)
            gen = actor.submit.options(num_returns='streaming', name=self.name).remote(DataContext.get_current(), ctx, *input_blocks)

            def _task_done_callback(actor_to_return):
                if False:
                    for i in range(10):
                        print('nop')
                self._actor_pool.return_actor(actor_to_return)
                self._dispatch_tasks()
            actor_to_return = actor
            self._submit_data_task(gen, bundle, lambda : _task_done_callback(actor_to_return))
        if self._bundle_queue:
            self._scale_up_if_needed()
        else:
            self._scale_down_if_needed()

    def _scale_up_if_needed(self):
        if False:
            for i in range(10):
                print('nop')
        'Try to scale up the pool if the autoscaling policy allows it.'
        while self._autoscaling_policy.should_scale_up(num_total_workers=self._actor_pool.num_total_actors(), num_running_workers=self._actor_pool.num_running_actors()):
            self._start_actor()

    def _scale_down_if_needed(self):
        if False:
            while True:
                i = 10
        'Try to scale down the pool if the autoscaling policy allows it.'
        self._kill_inactive_workers_if_done()
        while self._autoscaling_policy.should_scale_down(num_total_workers=self._actor_pool.num_total_actors(), num_idle_workers=self._actor_pool.num_idle_actors()):
            killed = self._actor_pool.kill_inactive_actor()
            if not killed:
                break

    def all_inputs_done(self):
        if False:
            i = 10
            return i + 15
        super().all_inputs_done()
        self._inputs_done = True
        self._scale_down_if_needed()

    def _kill_inactive_workers_if_done(self):
        if False:
            for i in range(10):
                print('nop')
        if self._inputs_done and (not self._bundle_queue):
            self._actor_pool.kill_all_inactive_actors()

    def shutdown(self):
        if False:
            i = 10
            return i + 15
        self._actor_pool.kill_all_actors()
        super().shutdown()
        min_workers = self._autoscaling_policy.min_workers
        if len(self._output_metadata) < min_workers:
            logger.get_logger().warning(f'To ensure full parallelization across an actor pool of size {min_workers}, the Dataset should consist of at least {min_workers} distinct blocks. Consider increasing the parallelism when creating the Dataset.')

    def progress_str(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        base = f'{self._actor_pool.num_running_actors()} actors'
        pending = self._actor_pool.num_pending_actors()
        if pending:
            base += f' ({pending} pending)'
        if self._actor_locality_enabled:
            base += ' ' + locality_string(self._actor_pool._locality_hits, self._actor_pool._locality_misses)
        else:
            base += ' [locality off]'
        return base

    def base_resource_usage(self) -> ExecutionResources:
        if False:
            while True:
                i = 10
        min_workers = self._autoscaling_policy.min_workers
        return ExecutionResources(cpu=self._ray_remote_args.get('num_cpus', 0) * min_workers, gpu=self._ray_remote_args.get('num_gpus', 0) * min_workers)

    def current_resource_usage(self) -> ExecutionResources:
        if False:
            return 10
        num_active_workers = self._actor_pool.num_total_actors()
        return ExecutionResources(cpu=self._ray_remote_args.get('num_cpus', 0) * num_active_workers, gpu=self._ray_remote_args.get('num_gpus', 0) * num_active_workers, object_store_memory=self.metrics.obj_store_mem_cur)

    def incremental_resource_usage(self) -> ExecutionResources:
        if False:
            for i in range(10):
                print('nop')
        if self._autoscaling_policy.should_scale_up(num_total_workers=self._actor_pool.num_total_actors(), num_running_workers=self._actor_pool.num_running_actors()):
            num_cpus = self._ray_remote_args.get('num_cpus', 0)
            num_gpus = self._ray_remote_args.get('num_gpus', 0)
        else:
            num_cpus = 0
            num_gpus = 0
        return ExecutionResources(cpu=num_cpus, gpu=num_gpus)

    def _extra_metrics(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        res = {}
        if self._actor_locality_enabled:
            res['locality_hits'] = self._actor_pool._locality_hits
            res['locality_misses'] = self._actor_pool._locality_misses
        return res

    @staticmethod
    def _apply_default_remote_args(ray_remote_args: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Apply defaults to the actor creation remote args.'
        ray_remote_args = ray_remote_args.copy()
        if 'scheduling_strategy' not in ray_remote_args:
            ctx = DataContext.get_current()
            ray_remote_args['scheduling_strategy'] = ctx.scheduling_strategy
        if 'max_restarts' not in ray_remote_args:
            ray_remote_args['max_restarts'] = -1
        if 'max_task_retries' not in ray_remote_args and ray_remote_args.get('max_restarts') != 0:
            ray_remote_args['max_task_retries'] = -1
        return ray_remote_args

class _MapWorker:
    """An actor worker for MapOperator."""

    def __init__(self, ctx: DataContext, src_fn_name: str, map_transformer: MapTransformer):
        if False:
            for i in range(10):
                print('nop')
        DataContext._set_current(ctx)
        self.src_fn_name: str = src_fn_name
        self._map_transformer = map_transformer
        self._map_transformer.init()

    def get_location(self) -> NodeIdStr:
        if False:
            while True:
                i = 10
        return ray.get_runtime_context().get_node_id()

    def submit(self, data_context: DataContext, ctx: TaskContext, *blocks: Block) -> Iterator[Union[Block, List[BlockMetadata]]]:
        if False:
            for i in range(10):
                print('nop')
        yield from _map_task(self._map_transformer, data_context, ctx, *blocks)

    def __repr__(self):
        if False:
            return 10
        return f'MapWorker({self.src_fn_name})'

@dataclass
class AutoscalingConfig:
    """Configuration for an autoscaling actor pool."""
    min_workers: int
    max_workers: int
    max_tasks_in_flight: int = DEFAULT_MAX_TASKS_IN_FLIGHT
    ready_to_total_workers_ratio: float = 0.8
    idle_to_total_workers_ratio: float = 0.5

    def __post_init__(self):
        if False:
            return 10
        if self.min_workers < 1:
            raise ValueError('min_workers must be >= 1, got: ', self.min_workers)
        if self.max_workers is not None and self.min_workers > self.max_workers:
            raise ValueError('min_workers must be <= max_workers, got: ', self.min_workers, self.max_workers)
        if self.max_tasks_in_flight < 1:
            raise ValueError('max_tasks_in_flight must be >= 1, got: ', self.max_tasks_in_flight)

    @classmethod
    def from_compute_strategy(cls, compute_strategy: ActorPoolStrategy):
        if False:
            for i in range(10):
                print('nop')
        'Convert a legacy ActorPoolStrategy to an AutoscalingConfig.'
        assert isinstance(compute_strategy, ActorPoolStrategy)
        return cls(min_workers=compute_strategy.min_size, max_workers=compute_strategy.max_size, max_tasks_in_flight=compute_strategy.max_tasks_in_flight_per_actor or DEFAULT_MAX_TASKS_IN_FLIGHT, ready_to_total_workers_ratio=compute_strategy.ready_to_total_workers_ratio)

class AutoscalingPolicy:
    """Autoscaling policy for an actor pool, determining when the pool should be scaled
    up and when it should be scaled down.
    """

    def __init__(self, autoscaling_config: 'AutoscalingConfig'):
        if False:
            for i in range(10):
                print('nop')
        self._config = autoscaling_config

    @property
    def min_workers(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The minimum number of actors that must be in the actor pool.'
        return self._config.min_workers

    @property
    def max_workers(self) -> int:
        if False:
            while True:
                i = 10
        'The maximum number of actors that can be added to the actor pool.'
        return self._config.max_workers

    def should_scale_up(self, num_total_workers: int, num_running_workers: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Whether the actor pool should scale up by adding a new actor.\n\n        Args:\n            num_total_workers: Total number of workers in actor pool.\n            num_running_workers: Number of currently running workers in actor pool.\n\n        Returns:\n            Whether the actor pool should be scaled up by one actor.\n        '
        if num_total_workers < self._config.min_workers:
            return True
        else:
            return num_total_workers < self._config.max_workers and num_total_workers > 0 and (num_running_workers / num_total_workers > self._config.ready_to_total_workers_ratio)

    def should_scale_down(self, num_total_workers: int, num_idle_workers: int) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether the actor pool should scale down by terminating an inactive actor.\n\n        Args:\n            num_total_workers: Total number of workers in actor pool.\n            num_idle_workers: Number of currently idle workers in the actor pool.\n\n        Returns:\n            Whether the actor pool should be scaled down by one actor.\n        '
        return num_total_workers > self._config.min_workers and num_idle_workers / num_total_workers > self._config.idle_to_total_workers_ratio

class _ActorPool:
    """A pool of actors for map task execution.

    This class is in charge of tracking the number of in-flight tasks per actor,
    providing the least heavily loaded actor to the operator, and killing idle
    actors when the operator is done submitting work to the pool.
    """

    def __init__(self, max_tasks_in_flight: int=DEFAULT_MAX_TASKS_IN_FLIGHT):
        if False:
            for i in range(10):
                print('nop')
        self._max_tasks_in_flight = max_tasks_in_flight
        self._num_tasks_in_flight: Dict[ray.actor.ActorHandle, int] = {}
        self._actor_locations: Dict[ray.actor.ActorHandle, str] = {}
        self._pending_actors: Dict[ObjectRef, ray.actor.ActorHandle] = {}
        self._should_kill_idle_actors = False
        self._locality_hits: int = 0
        self._locality_misses: int = 0

    def add_pending_actor(self, actor: ray.actor.ActorHandle, ready_ref: ray.ObjectRef):
        if False:
            while True:
                i = 10
        "Adds a pending actor to the pool.\n\n        This actor won't be pickable until it is marked as running via a\n        pending_to_running() call.\n\n        Args:\n            actor: The not-yet-ready actor to add as pending to the pool.\n            ready_ref: The ready future for the actor.\n        "
        assert not self._should_kill_idle_actors
        self._pending_actors[ready_ref] = actor

    def pending_to_running(self, ready_ref: ray.ObjectRef) -> bool:
        if False:
            while True:
                i = 10
        'Mark the actor corresponding to the provided ready future as running, making\n        the actor pickable.\n\n        Args:\n            ready_ref: The ready future for the actor that we wish to mark as running.\n\n        Returns:\n            Whether the actor was still pending. This can return False if the actor had\n            already been killed.\n        '
        if ready_ref not in self._pending_actors:
            return False
        actor = self._pending_actors.pop(ready_ref)
        self._num_tasks_in_flight[actor] = 0
        self._actor_locations[actor] = ray.get(ready_ref)
        return True

    def pick_actor(self, locality_hint: Optional[RefBundle]=None) -> Optional[ray.actor.ActorHandle]:
        if False:
            for i in range(10):
                print('nop')
        'Picks an actor for task submission based on busyness and locality.\n\n        None will be returned if all actors are either at capacity (according to\n        max_tasks_in_flight) or are still pending.\n\n        Args:\n            locality_hint: Try to pick an actor that is local for this bundle.\n        '
        if not self._num_tasks_in_flight:
            return None
        if locality_hint:
            preferred_loc = self._get_location(locality_hint)
        else:
            preferred_loc = None

        def penalty_key(actor):
            if False:
                return 10
            'Returns the key that should be minimized for the best actor.\n\n            We prioritize valid actors, those with argument locality, and those that\n            are not busy, in that order.\n            '
            busyness = self._num_tasks_in_flight[actor]
            invalid = busyness >= self._max_tasks_in_flight
            requires_remote_fetch = self._actor_locations[actor] != preferred_loc
            return (invalid, requires_remote_fetch, busyness)
        actor = min(self._num_tasks_in_flight.keys(), key=penalty_key)
        if self._num_tasks_in_flight[actor] >= self._max_tasks_in_flight:
            return None
        if locality_hint:
            if self._actor_locations[actor] == preferred_loc:
                self._locality_hits += 1
            else:
                self._locality_misses += 1
        self._num_tasks_in_flight[actor] += 1
        return actor

    def return_actor(self, actor: ray.actor.ActorHandle):
        if False:
            i = 10
            return i + 15
        'Returns the provided actor to the pool.'
        assert actor in self._num_tasks_in_flight
        assert self._num_tasks_in_flight[actor] > 0
        self._num_tasks_in_flight[actor] -= 1
        if self._should_kill_idle_actors and self._num_tasks_in_flight[actor] == 0:
            self._kill_running_actor(actor)

    def get_pending_actor_refs(self) -> List[ray.ObjectRef]:
        if False:
            print('Hello World!')
        return list(self._pending_actors.keys())

    def num_total_actors(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the total number of actors managed by this pool, including pending\n        actors\n        '
        return self.num_pending_actors() + self.num_running_actors()

    def num_running_actors(self) -> int:
        if False:
            return 10
        'Return the number of running actors in the pool.'
        return len(self._num_tasks_in_flight)

    def num_idle_actors(self) -> int:
        if False:
            while True:
                i = 10
        'Return the number of idle actors in the pool.'
        return sum((1 if tasks_in_flight == 0 else 0 for tasks_in_flight in self._num_tasks_in_flight.values()))

    def num_pending_actors(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the number of pending actors in the pool.'
        return len(self._pending_actors)

    def num_free_slots(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the number of free slots for task execution.'
        if not self._num_tasks_in_flight:
            return 0
        return sum((max(0, self._max_tasks_in_flight - num_tasks_in_flight) for num_tasks_in_flight in self._num_tasks_in_flight.values()))

    def num_active_actors(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the number of actors in the pool with at least one active task.'
        return sum((1 if num_tasks_in_flight > 0 else 0 for num_tasks_in_flight in self._num_tasks_in_flight.values()))

    def kill_inactive_actor(self) -> bool:
        if False:
            print('Hello World!')
        'Kills a single pending or idle actor, if any actors are pending/idle.\n\n        Returns whether an inactive actor was actually killed.\n        '
        killed = self._maybe_kill_pending_actor()
        if not killed:
            killed = self._maybe_kill_idle_actor()
        return killed

    def _maybe_kill_pending_actor(self) -> bool:
        if False:
            return 10
        if self._pending_actors:
            self._kill_pending_actor(next(iter(self._pending_actors.keys())))
            return True
        return False

    def _maybe_kill_idle_actor(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        for (actor, tasks_in_flight) in self._num_tasks_in_flight.items():
            if tasks_in_flight == 0:
                self._kill_running_actor(actor)
                return True
        return False

    def kill_all_inactive_actors(self):
        if False:
            for i in range(10):
                print('nop')
        'Kills all currently inactive actors and ensures that all actors that become\n        idle in the future will be eagerly killed.\n\n        This is called once the operator is done submitting work to the pool, and this\n        function is idempotent. Adding new pending actors after calling this function\n        will raise an error.\n        '
        self._kill_all_pending_actors()
        self._kill_all_idle_actors()

    def kill_all_actors(self):
        if False:
            for i in range(10):
                print('nop')
        'Kills all actors, including running/active actors.\n\n        This is called once the operator is shutting down.\n        '
        self._kill_all_pending_actors()
        self._kill_all_running_actors()

    def _kill_all_pending_actors(self):
        if False:
            i = 10
            return i + 15
        pending_actor_refs = list(self._pending_actors.keys())
        for ref in pending_actor_refs:
            self._kill_pending_actor(ref)

    def _kill_all_idle_actors(self):
        if False:
            i = 10
            return i + 15
        idle_actors = [actor for (actor, tasks_in_flight) in self._num_tasks_in_flight.items() if tasks_in_flight == 0]
        for actor in idle_actors:
            self._kill_running_actor(actor)
        self._should_kill_idle_actors = True

    def _kill_all_running_actors(self):
        if False:
            i = 10
            return i + 15
        actors = list(self._num_tasks_in_flight.keys())
        for actor in actors:
            self._kill_running_actor(actor)

    def _kill_running_actor(self, actor: ray.actor.ActorHandle):
        if False:
            for i in range(10):
                print('nop')
        'Kill the provided actor and remove it from the pool.'
        ray.kill(actor)
        del self._num_tasks_in_flight[actor]

    def _kill_pending_actor(self, ready_ref: ray.ObjectRef):
        if False:
            while True:
                i = 10
        'Kill the provided pending actor and remove it from the pool.'
        actor = self._pending_actors.pop(ready_ref)
        ray.kill(actor)

    def _get_location(self, bundle: RefBundle) -> Optional[NodeIdStr]:
        if False:
            while True:
                i = 10
        'Ask Ray for the node id of the given bundle.\n\n        This method may be overriden for testing.\n\n        Returns:\n            A node id associated with the bundle, or None if unknown.\n        '
        return bundle.get_cached_location()