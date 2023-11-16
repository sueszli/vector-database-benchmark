"""EvaluationContext tracks global state, triggers and watermarks."""
import collections
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from apache_beam.runners.direct.direct_metrics import DirectMetrics
from apache_beam.runners.direct.executor import TransformExecutor
from apache_beam.runners.direct.watermark_manager import WatermarkManager
from apache_beam.transforms import sideinputs
from apache_beam.transforms.trigger import InMemoryUnmergedState
from apache_beam.utils import counters
if TYPE_CHECKING:
    from apache_beam import pvalue
    from apache_beam.pipeline import AppliedPTransform
    from apache_beam.runners.direct.bundle_factory import BundleFactory, _Bundle
    from apache_beam.runners.direct.util import TimerFiring
    from apache_beam.runners.direct.util import TransformResult
    from apache_beam.runners.direct.watermark_manager import _TransformWatermarks
    from apache_beam.utils.timestamp import Timestamp

class _ExecutionContext(object):
    """Contains the context for the execution of a single PTransform.

  It holds the watermarks for that transform, as well as keyed states.
  """

    def __init__(self, watermarks, keyed_states):
        if False:
            return 10
        self.watermarks = watermarks
        self.keyed_states = keyed_states
        self._step_context = None

    def get_step_context(self):
        if False:
            i = 10
            return i + 15
        if not self._step_context:
            self._step_context = DirectStepContext(self.keyed_states)
        return self._step_context

    def reset(self):
        if False:
            return 10
        self._step_context = None

class _SideInputView(object):

    def __init__(self, view):
        if False:
            while True:
                i = 10
        self._view = view
        self.blocked_tasks = collections.deque()
        self.elements = []
        self.value = None
        self.watermark = None

    def __repr__(self):
        if False:
            while True:
                i = 10
        elements_string = ', '.join((str(elm) for elm in self.elements)) if self.elements else '[]'
        return '_SideInputView(elements=%s)' % elements_string

class _SideInputsContainer(object):
    """An in-process container for side inputs.

  It provides methods for blocking until a side-input is available and writing
  to a side input.
  """

    def __init__(self, side_inputs):
        if False:
            i = 10
            return i + 15
        self._lock = threading.Lock()
        self._views = {}
        self._transform_to_side_inputs = collections.defaultdict(list)
        self._side_input_to_blocked_tasks = collections.defaultdict(list)
        for side in side_inputs:
            self._views[side] = _SideInputView(side)
            self._transform_to_side_inputs[side.pvalue.producer].append(side)

    def __repr__(self):
        if False:
            while True:
                i = 10
        views_string = ', '.join((str(elm) for elm in self._views.values())) if self._views else '[]'
        return '_SideInputsContainer(_views=%s)' % views_string

    def get_value_or_block_until_ready(self, side_input, task, block_until):
        if False:
            i = 10
            return i + 15
        'Returns the value of a view whose task is unblocked or blocks its task.\n\n    It gets the value of a view whose watermark has been updated and\n    surpasses a given value.\n\n    Args:\n      side_input: ``_UnpickledSideInput`` value.\n      task: ``TransformExecutor`` task waiting on a side input.\n      block_until: Timestamp after which the task gets unblocked.\n\n    Returns:\n      The ``SideInputMap`` value of a view when the tasks it blocks are\n      unblocked. Otherwise, None.\n    '
        with self._lock:
            view = self._views[side_input]
            if view.watermark and view.watermark.output_watermark >= block_until:
                view.value = self._pvalue_to_value(side_input, view.elements)
                return view.value
            else:
                view.blocked_tasks.append((task, block_until))
                task.blocked = True

    def add_values(self, side_input, values):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            view = self._views[side_input]
            view.elements.extend(values)

    def update_watermarks_for_transform_and_unblock_tasks(self, ptransform, watermark):
        if False:
            return 10
        'Updates _SideInputsContainer after a watermark update and unbloks tasks.\n\n    It traverses the list of side inputs per PTransform and calls\n    _update_watermarks_for_side_input_and_unblock_tasks to unblock tasks.\n\n    Args:\n      ptransform: Value of a PTransform.\n      watermark: Value of the watermark after an update for a PTransform.\n\n    Returns:\n      Tasks that get unblocked as a result of the watermark advancing.\n    '
        unblocked_tasks = []
        for side in self._transform_to_side_inputs[ptransform]:
            unblocked_tasks.extend(self._update_watermarks_for_side_input_and_unblock_tasks(side, watermark))
        return unblocked_tasks

    def _update_watermarks_for_side_input_and_unblock_tasks(self, side_input, watermark):
        if False:
            while True:
                i = 10
        'Helps update _SideInputsContainer after a watermark update.\n\n    For each view of the side input, it updates the value of the watermark\n    recorded when the watermark moved and unblocks tasks accordingly.\n\n    Args:\n      side_input: ``_UnpickledSideInput`` value.\n      watermark: Value of the watermark after an update for a PTransform.\n\n    Returns:\n      Tasks that get unblocked as a result of the watermark advancing.\n    '
        with self._lock:
            view = self._views[side_input]
            view.watermark = watermark
            unblocked_tasks = []
            tasks_just_unblocked = []
            for (task, block_until) in view.blocked_tasks:
                if watermark.output_watermark >= block_until:
                    view.value = self._pvalue_to_value(side_input, view.elements)
                    unblocked_tasks.append(task)
                    tasks_just_unblocked.append((task, block_until))
                    task.blocked = False
            for task in tasks_just_unblocked:
                view.blocked_tasks.remove(task)
            return unblocked_tasks

    def _pvalue_to_value(self, side_input, values):
        if False:
            while True:
                i = 10
        'Given a side input, returns the associated value in its requested form.\n\n    Args:\n      side_input: _UnpickledSideInput object.\n      values: Iterable values associated with the side input.\n\n    Returns:\n      The side input in its requested form.\n\n    Raises:\n      ValueError: If values cannot be converted into the requested form.\n    '
        return sideinputs.SideInputMap(type(side_input), side_input._view_options(), values)

class EvaluationContext(object):
    """Evaluation context with the global state information of the pipeline.

  The evaluation context for a specific pipeline being executed by the
  DirectRunner. Contains state shared within the execution across all
  transforms.

  EvaluationContext contains shared state for an execution of the
  DirectRunner that can be used while evaluating a PTransform. This
  consists of views into underlying state and watermark implementations, access
  to read and write side inputs, and constructing counter sets and
  execution contexts. This includes executing callbacks asynchronously when
  state changes to the appropriate point (e.g. when a side input is
  requested and known to be empty).

  EvaluationContext also handles results by committing finalizing
  bundles based on the current global state and updating the global state
  appropriately. This includes updating the per-(step,key) state, updating
  global watermarks, and executing any callbacks that can be executed.
  """

    def __init__(self, pipeline_options, bundle_factory, root_transforms, value_to_consumers, step_names, views, clock):
        if False:
            for i in range(10):
                print('nop')
        self.pipeline_options = pipeline_options
        self._bundle_factory = bundle_factory
        self._root_transforms = root_transforms
        self._value_to_consumers = value_to_consumers
        self._step_names = step_names
        self.views = views
        self._pcollection_to_views = collections.defaultdict(list)
        for view in views:
            self._pcollection_to_views[view.pvalue].append(view)
        self._transform_keyed_states = self._initialize_keyed_states(root_transforms, value_to_consumers)
        self._side_inputs_container = _SideInputsContainer(views)
        self._watermark_manager = WatermarkManager(clock, root_transforms, value_to_consumers, self._transform_keyed_states)
        self._pending_unblocked_tasks = []
        self._counter_factory = counters.CounterFactory()
        self._metrics = DirectMetrics()
        self._lock = threading.Lock()
        self.shutdown_requested = False

    def _initialize_keyed_states(self, root_transforms, value_to_consumers):
        if False:
            i = 10
            return i + 15
        'Initialize user state dicts.\n\n    These dicts track user state per-key, per-transform and per-window.\n    '
        transform_keyed_states = {}
        for transform in root_transforms:
            transform_keyed_states[transform] = {}
        for consumers in value_to_consumers.values():
            for consumer in consumers:
                transform_keyed_states[consumer] = {}
        return transform_keyed_states

    def metrics(self):
        if False:
            for i in range(10):
                print('nop')
        return self._metrics

    def is_root_transform(self, applied_ptransform):
        if False:
            while True:
                i = 10
        return applied_ptransform in self._root_transforms

    def handle_result(self, completed_bundle, completed_timers, result):
        if False:
            print('Hello World!')
        'Handle the provided result produced after evaluating the input bundle.\n\n    Handle the provided TransformResult, produced after evaluating\n    the provided committed bundle (potentially None, if the result of a root\n    PTransform).\n\n    The result is the output of running the transform contained in the\n    TransformResult on the contents of the provided bundle.\n\n    Args:\n      completed_bundle: the bundle that was processed to produce the result.\n      completed_timers: the timers that were delivered to produce the\n                        completed_bundle.\n      result: the ``TransformResult`` of evaluating the input bundle\n\n    Returns:\n      the committed bundles contained within the handled result.\n    '
        with self._lock:
            (committed_bundles, unprocessed_bundles) = self._commit_bundles(result.uncommitted_output_bundles, result.unprocessed_bundles)
            self._metrics.commit_logical(completed_bundle, result.logical_metric_updates)
            self._update_side_inputs_container(committed_bundles, result)
            tasks = self._watermark_manager.update_watermarks(completed_bundle, result.transform, completed_timers, committed_bundles, unprocessed_bundles, result.keyed_watermark_holds, self._side_inputs_container)
            self._pending_unblocked_tasks.extend(tasks)
            if result.counters:
                for counter in result.counters:
                    merged_counter = self._counter_factory.get_counter(counter.name, counter.combine_fn)
                    merged_counter.accumulator.merge([counter.accumulator])
            existing_keyed_state = self._transform_keyed_states[result.transform]
            for (k, v) in result.partial_keyed_state.items():
                existing_keyed_state[k] = v
            return committed_bundles

    def _update_side_inputs_container(self, committed_bundles, result):
        if False:
            i = 10
            return i + 15
        "Update the side inputs container if we are outputting into a side input.\n\n    Look at the result, and if it's outputing into a PCollection that we have\n    registered as a PCollectionView, we add the result to the PCollectionView.\n    "
        if result.uncommitted_output_bundles and result.uncommitted_output_bundles[0].pcollection in self._pcollection_to_views:
            for view in self._pcollection_to_views[result.uncommitted_output_bundles[0].pcollection]:
                for committed_bundle in committed_bundles:
                    self._side_inputs_container.add_values(view, committed_bundle.get_elements_iterable(make_copy=True))

    def get_aggregator_values(self, aggregator_or_name):
        if False:
            return 10
        return self._counter_factory.get_aggregator_values(aggregator_or_name)

    def schedule_pending_unblocked_tasks(self, executor_service):
        if False:
            print('Hello World!')
        if self._pending_unblocked_tasks:
            with self._lock:
                for task in self._pending_unblocked_tasks:
                    executor_service.submit(task)
                self._pending_unblocked_tasks = []

    def _commit_bundles(self, uncommitted_bundles, unprocessed_bundles):
        if False:
            while True:
                i = 10
        'Commits bundles and returns a immutable set of committed bundles.'
        for in_progress_bundle in uncommitted_bundles:
            producing_applied_ptransform = in_progress_bundle.pcollection.producer
            watermarks = self._watermark_manager.get_watermarks(producing_applied_ptransform)
            in_progress_bundle.commit(watermarks.synchronized_processing_output_time)
        for unprocessed_bundle in unprocessed_bundles:
            unprocessed_bundle.commit(None)
        return (tuple(uncommitted_bundles), tuple(unprocessed_bundles))

    def get_execution_context(self, applied_ptransform):
        if False:
            for i in range(10):
                print('nop')
        return _ExecutionContext(self._watermark_manager.get_watermarks(applied_ptransform), self._transform_keyed_states[applied_ptransform])

    def create_bundle(self, output_pcollection):
        if False:
            for i in range(10):
                print('nop')
        'Create an uncommitted bundle for the specified PCollection.'
        return self._bundle_factory.create_bundle(output_pcollection)

    def create_empty_committed_bundle(self, output_pcollection):
        if False:
            return 10
        'Create empty bundle useful for triggering evaluation.'
        return self._bundle_factory.create_empty_committed_bundle(output_pcollection)

    def extract_all_timers(self):
        if False:
            for i in range(10):
                print('nop')
        return self._watermark_manager.extract_all_timers()

    def is_done(self, transform=None):
        if False:
            i = 10
            return i + 15
        'Checks completion of a step or the pipeline.\n\n    Args:\n      transform: AppliedPTransform to check for completion.\n\n    Returns:\n      True if the step will not produce additional output. If transform is None\n      returns true if all steps are done.\n    '
        if transform:
            return self._is_transform_done(transform)
        for applied_ptransform in self._step_names:
            if not self._is_transform_done(applied_ptransform):
                return False
        return True

    def _is_transform_done(self, transform):
        if False:
            while True:
                i = 10
        tw = self._watermark_manager.get_watermarks(transform)
        return tw.output_watermark == WatermarkManager.WATERMARK_POS_INF

    def get_value_or_block_until_ready(self, side_input, task, block_until):
        if False:
            i = 10
            return i + 15
        assert isinstance(task, TransformExecutor)
        return self._side_inputs_container.get_value_or_block_until_ready(side_input, task, block_until)

    def shutdown(self):
        if False:
            print('Hello World!')
        self.shutdown_requested = True

class DirectUnmergedState(InMemoryUnmergedState):
    """UnmergedState implementation for the DirectRunner."""

    def __init__(self):
        if False:
            return 10
        super().__init__(defensive_copy=False)

class DirectStepContext(object):
    """Context for the currently-executing step."""

    def __init__(self, existing_keyed_state):
        if False:
            print('Hello World!')
        self.existing_keyed_state = existing_keyed_state
        self.partial_keyed_state = {}

    def get_keyed_state(self, key):
        if False:
            while True:
                i = 10
        if not self.existing_keyed_state.get(key):
            self.existing_keyed_state[key] = DirectUnmergedState()
        if not self.partial_keyed_state.get(key):
            self.partial_keyed_state[key] = self.existing_keyed_state[key].copy()
        return self.partial_keyed_state[key]