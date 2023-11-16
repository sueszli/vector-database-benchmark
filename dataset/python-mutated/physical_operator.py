from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import ray
from .ref_bundle import RefBundle
from ray._raylet import StreamingObjectRefGenerator
from ray.data._internal.execution.interfaces.execution_options import ExecutionOptions, ExecutionResources
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.logical.interfaces import Operator
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
Waitable = Union[ray.ObjectRef, StreamingObjectRefGenerator]

class OpTask(ABC):
    """Abstract class that represents a task that is created by an PhysicalOperator.

    The task can be either a regular task or an actor task.
    """

    @abstractmethod
    def get_waitable(self) -> Waitable:
        if False:
            while True:
                i = 10
        'Return the ObjectRef or StreamingObjectRefGenerator to wait on.'
        pass

class DataOpTask(OpTask):
    """Represents an OpTask that handles Block data."""

    def __init__(self, streaming_gen: StreamingObjectRefGenerator, output_ready_callback: Callable[[RefBundle], None], task_done_callback: Callable[[], None]):
        if False:
            return 10
        '\n        Args:\n            streaming_gen: The streaming generator of this task. It should yield blocks.\n            output_ready_callback: The callback to call when a new RefBundle is output\n                from the generator.\n            task_done_callback: The callback to call when the task is done.\n        '
        self._streaming_gen = streaming_gen
        self._output_ready_callback = output_ready_callback
        self._task_done_callback = task_done_callback

    def get_waitable(self) -> StreamingObjectRefGenerator:
        if False:
            for i in range(10):
                print('nop')
        return self._streaming_gen

    def on_data_ready(self, max_blocks_to_read: Optional[int]) -> int:
        if False:
            print('Hello World!')
        'Callback when data is ready to be read from the streaming generator.\n\n        Args:\n            max_blocks_to_read: Max number of blocks to read. If None, all available\n                will be read.\n        Returns: The number of blocks read.\n        '
        num_blocks_read = 0
        while max_blocks_to_read is None or num_blocks_read < max_blocks_to_read:
            try:
                block_ref = self._streaming_gen._next_sync(0)
                if block_ref.is_nil():
                    break
            except StopIteration:
                self._task_done_callback()
                break
            try:
                meta = ray.get(next(self._streaming_gen))
            except StopIteration:
                ex = ray.get(block_ref)
                self._task_done_callback()
                raise ex
            self._output_ready_callback(RefBundle([(block_ref, meta)], owns_blocks=True))
            num_blocks_read += 1
        return num_blocks_read

class MetadataOpTask(OpTask):
    """Represents an OpTask that only handles metadata, instead of Block data."""

    def __init__(self, object_ref: ray.ObjectRef, task_done_callback: Callable[[], None]):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            object_ref: The ObjectRef of the task.\n            task_done_callback: The callback to call when the task is done.\n        '
        self._object_ref = object_ref
        self._task_done_callback = task_done_callback

    def get_waitable(self) -> ray.ObjectRef:
        if False:
            return 10
        return self._object_ref

    def on_task_finished(self):
        if False:
            return 10
        'Callback when the task is finished.'
        self._task_done_callback()

class PhysicalOperator(Operator):
    """Abstract class for physical operators.

    An operator transforms one or more input streams of RefBundles into a single
    output stream of RefBundles.

    Physical operators are stateful and non-serializable; they live on the driver side
    of the Dataset only.

    Here's a simple example of implementing a basic "Map" operator:

        class MapOperator(PhysicalOperator):
            def __init__(self):
                self.active_tasks = []

            def add_input(self, refs, _):
                self.active_tasks.append(map_task.remote(refs))

            def has_next(self):
                ready, _ = ray.wait(self.active_tasks, timeout=0)
                return len(ready) > 0

            def get_next(self):
                ready, remaining = ray.wait(self.active_tasks, num_returns=1)
                self.active_tasks = remaining
                return ready[0]

    Note that the above operator fully supports both bulk and streaming execution,
    since `add_input` and `get_next` can be called in any order. In bulk execution,
    all inputs would be added up-front, but in streaming execution the calls could
    be interleaved.
    """

    def __init__(self, name: str, input_dependencies: List['PhysicalOperator'], target_max_block_size: Optional[int]):
        if False:
            print('Hello World!')
        super().__init__(name, input_dependencies)
        for x in input_dependencies:
            assert isinstance(x, PhysicalOperator), x
        self._inputs_complete = not input_dependencies
        self._target_max_block_size = target_max_block_size
        self._dependents_complete = False
        self._started = False
        self._metrics = OpRuntimeMetrics(self)
        self._estimated_output_blocks = None

    def __reduce__(self):
        if False:
            print('Hello World!')
        raise ValueError('Operator is not serializable.')

    @property
    def target_max_block_size(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        '\n        Target max block size output by this operator. If this returns None,\n        then the default from DataContext should be used.\n        '
        return self._target_max_block_size

    @property
    def actual_target_max_block_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        The actual target max block size output by this operator.\n        '
        target_max_block_size = self._target_max_block_size
        if target_max_block_size is None:
            target_max_block_size = DataContext.get_current().target_max_block_size
        return target_max_block_size

    def completed(self) -> bool:
        if False:
            return 10
        'Return True when this operator is completed.\n\n        An operator is completed if any of the following conditions are met:\n        - All upstream operators are completed and all outputs are taken.\n        - All downstream operators are completed.\n        '
        return self._inputs_complete and self.num_active_tasks() == 0 and (not self.has_next()) or self._dependents_complete

    def get_stats(self) -> StatsDict:
        if False:
            return 10
        'Return recorded execution stats for use with DatasetStats.'
        raise NotImplementedError

    @property
    def metrics(self) -> OpRuntimeMetrics:
        if False:
            return 10
        'Returns the runtime metrics of this operator.'
        self._metrics._extra_metrics = self._extra_metrics()
        return self._metrics

    def _extra_metrics(self) -> Dict[str, Any]:
        if False:
            return 10
        'Subclasses should override this method to report extra metrics\n        that are specific to them.'
        return {}

    def progress_str(self) -> str:
        if False:
            return 10
        'Return any extra status to be displayed in the operator progress bar.\n\n        For example, `<N> actors` to show current number of actors in an actor pool.\n        '
        return ''

    def num_outputs_total(self) -> int:
        if False:
            return 10
        'Returns the total number of output bundles of this operator.\n\n        The value returned may be an estimate based off the consumption so far.\n        This is useful for reporting progress.\n        '
        if self._estimated_output_blocks is not None:
            return self._estimated_output_blocks
        if len(self.input_dependencies) == 1:
            return self.input_dependencies[0].num_outputs_total()
        raise AttributeError

    def start(self, options: ExecutionOptions) -> None:
        if False:
            i = 10
            return i + 15
        'Called by the executor when execution starts for an operator.\n\n        Args:\n            options: The global options used for the overall execution.\n        '
        self._started = True

    def should_add_input(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Return whether it is desirable to add input to this operator right now.\n\n        Operators can customize the implementation of this method to apply additional\n        backpressure (e.g., waiting for internal actors to be created).\n        '
        return True

    def need_more_inputs(self) -> bool:
        if False:
            return 10
        'Return true if the operator still needs more inputs.\n\n        Once this return false, it should never return true again.\n        '
        return True

    def add_input(self, refs: RefBundle, input_index: int) -> None:
        if False:
            i = 10
            return i + 15
        'Called when an upstream result is available.\n\n        Inputs may be added in any order, and calls to `add_input` may be interleaved\n        with calls to `get_next` / `has_next` to implement streaming execution.\n\n        Subclasses should override `_add_input_inner` instead of this method.\n\n        Args:\n            refs: The ref bundle that should be added as input.\n            input_index: The index identifying the input dependency producing the\n                input. For most operators, this is always `0` since there is only\n                one upstream input operator.\n        '
        self._metrics.on_input_received(refs)
        self._add_input_inner(refs, input_index)

    def _add_input_inner(self, refs: RefBundle, input_index: int) -> None:
        if False:
            while True:
                i = 10
        'Subclasses should override this method to implement `add_input`.'
        raise NotImplementedError

    def input_done(self, input_index: int) -> None:
        if False:
            return 10
        'Called when the upstream operator at index `input_index` has completed().\n\n        After this is called, the executor guarantees that no more inputs will be added\n        via `add_input` for the given input index.\n        '
        pass

    def all_inputs_done(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Called when all upstream operators have completed().\n\n        After this is called, the executor guarantees that no more inputs will be added\n        via `add_input` for any input index.\n        '
        self._inputs_complete = True

    def all_dependents_complete(self) -> None:
        if False:
            while True:
                i = 10
        'Called when all downstream operators have completed().\n\n        After this is called, the operator is marked as completed.\n        '
        self._dependents_complete = True

    def has_next(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns when a downstream output is available.\n\n        When this returns true, it is safe to call `get_next()`.\n        '
        raise NotImplementedError

    def get_next(self) -> RefBundle:
        if False:
            i = 10
            return i + 15
        'Get the next downstream output.\n\n        It is only allowed to call this if `has_next()` has returned True.\n\n        Subclasses should override `_get_next_inner` instead of this method.\n        '
        output = self._get_next_inner()
        self._metrics.on_output_taken(output)
        return output

    def _get_next_inner(self) -> RefBundle:
        if False:
            print('Hello World!')
        'Subclasses should override this method to implement `get_next`.'
        raise NotImplementedError

    def get_active_tasks(self) -> List[OpTask]:
        if False:
            for i in range(10):
                print('nop')
        'Get a list of the active tasks of this operator.'
        return []

    def num_active_tasks(self) -> int:
        if False:
            print('Hello World!')
        'Return the number of active tasks.\n\n        Subclasses can override this as a performance optimization.\n        '
        return len(self.get_active_tasks())

    def throttling_disabled(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Whether to disable resource throttling for this operator.\n\n        This should return True for operators that only manipulate bundle metadata\n        (e.g., the OutputSplitter operator). This hints to the execution engine that\n        these operators should not be throttled based on resource usage.\n        '
        return False

    def internal_queue_size(self) -> int:
        if False:
            print('Hello World!')
        'If the operator has an internal input queue, return its size.\n\n        This is used to report tasks pending submission to actor pools.\n        '
        return 0

    def shutdown(self) -> None:
        if False:
            print('Hello World!')
        'Abort execution and release all resources used by this operator.\n\n        This release any Ray resources acquired by this operator such as active\n        tasks, actors, and objects.\n        '
        if not self._started:
            raise ValueError('Operator must be started before being shutdown.')

    def current_resource_usage(self) -> ExecutionResources:
        if False:
            print('Hello World!')
        'Returns the current estimated resource usage of this operator.\n\n        This method is called by the executor to decide how to allocate resources\n        between different operators.\n        '
        return ExecutionResources()

    def base_resource_usage(self) -> ExecutionResources:
        if False:
            while True:
                i = 10
        'Returns the minimum amount of resources required for execution.\n\n        For example, an operator that creates an actor pool requiring 8 GPUs could\n        return ExecutionResources(gpu=8) as its base usage.\n        '
        return ExecutionResources()

    def incremental_resource_usage(self) -> ExecutionResources:
        if False:
            print('Hello World!')
        'Returns the incremental resources required for processing another input.\n\n        For example, an operator that launches a task per input could return\n        ExecutionResources(cpu=1) as its incremental usage.\n        '
        return ExecutionResources()

    def notify_resource_usage(self, input_queue_size: int, under_resource_limits: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Called periodically by the executor.\n\n        Args:\n            input_queue_size: The number of inputs queued outside this operator.\n            under_resource_limits: Whether this operator is under resource limits.\n        '
        pass