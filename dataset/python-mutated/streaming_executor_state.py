"""Contains classes that encapsulate streaming executor state.

This is split out from streaming_executor.py to facilitate better unit testing.
"""
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union
import ray
from ray.data._internal.execution.autoscaling_requester import get_or_create_autoscaling_requester_actor
from ray.data._internal.execution.backpressure_policy import BackpressurePolicy
from ray.data._internal.execution.interfaces import ExecutionOptions, ExecutionResources, PhysicalOperator, RefBundle
from ray.data._internal.execution.interfaces.physical_operator import DataOpTask, MetadataOpTask, OpTask, Waitable
from ray.data._internal.execution.operators.base_physical_operator import AllToAllOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.util import memory_string
from ray.data._internal.progress_bar import ProgressBar
Topology = Dict[PhysicalOperator, 'OpState']
MaybeRefBundle = Union[RefBundle, Exception, None]
DEFAULT_OBJECT_STORE_MEMORY_LIMIT_FRACTION = 0.25
MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS = 20

@dataclass
class AutoscalingState:
    """State of the interaction between an executor and Ray autoscaler."""
    last_request_ts: int = 0

@dataclass
class TopologyResourceUsage:
    """Snapshot of resource usage in a `Topology` object.

    The stats here can be computed on the fly from any `Topology`; this class
    serves only a convenience wrapper to access the current usage snapshot.
    """
    overall: ExecutionResources
    downstream_memory_usage: Dict[PhysicalOperator, 'DownstreamMemoryInfo']

    @staticmethod
    def of(topology: Topology) -> 'TopologyResourceUsage':
        if False:
            return 10
        'Calculate the resource usage of the given topology.'
        downstream_usage = {}
        cur_usage = ExecutionResources(0, 0, 0)
        for (op, state) in reversed(topology.items()):
            cur_usage = cur_usage.add(op.current_resource_usage())
            if not isinstance(op, InputDataBuffer):
                cur_usage.object_store_memory += state.outqueue_memory_usage()
            f = (1.0 + len(downstream_usage)) / max(1.0, len(topology) - 1.0)
            downstream_usage[op] = DownstreamMemoryInfo(topology_fraction=min(1.0, f), object_store_memory=cur_usage.object_store_memory)
        return TopologyResourceUsage(cur_usage, downstream_usage)

@dataclass
class DownstreamMemoryInfo:
    """Mem stats of an operator and its downstream operators in a topology."""
    topology_fraction: float
    object_store_memory: float

class OpState:
    """The execution state tracked for each PhysicalOperator.

    This tracks state to manage input and output buffering for StreamingExecutor and
    progress bars, which is separate from execution state internal to the operators.

    Note: we use the `deque` data structure here because it is thread-safe, enabling
    operator queues to be shared across threads.
    """

    def __init__(self, op: PhysicalOperator, inqueues: List[Deque[MaybeRefBundle]]):
        if False:
            print('Hello World!')
        assert len(inqueues) == len(op.input_dependencies), (op, inqueues)
        self.inqueues: List[Deque[MaybeRefBundle]] = inqueues
        self.outqueue: Deque[MaybeRefBundle] = deque()
        self.op = op
        self.progress_bar = None
        self.num_completed_tasks = 0
        self.inputs_done_called = False
        self.input_done_called = [False] * len(op.input_dependencies)
        self.dependents_completed_called = False

    def initialize_progress_bars(self, index: int, verbose_progress: bool) -> int:
        if False:
            return 10
        'Create progress bars at the given index (line offset in console).\n\n        For AllToAllOperator, zero or more sub progress bar would be created.\n        Return the number of progress bars created for this operator.\n        '
        is_all_to_all = isinstance(self.op, AllToAllOperator)
        enabled = verbose_progress or is_all_to_all
        self.progress_bar = ProgressBar('- ' + self.op.name, self.op.num_outputs_total(), index, enabled=enabled)
        if enabled:
            num_bars = 1
            if is_all_to_all:
                num_bars += self.op.initialize_sub_progress_bars(index + 1)
        else:
            num_bars = 0
        return num_bars

    def close_progress_bars(self):
        if False:
            while True:
                i = 10
        'Close all progress bars for this operator.'
        if self.progress_bar:
            self.progress_bar.close()
            if isinstance(self.op, AllToAllOperator):
                self.op.close_sub_progress_bars()

    def num_queued(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Return the number of queued bundles across all inqueues.'
        return sum((len(q) for q in self.inqueues))

    def num_processing(self):
        if False:
            return 10
        'Return the number of bundles currently in processing for this operator.'
        return self.op.num_active_tasks() + self.op.internal_queue_size()

    def add_output(self, ref: RefBundle) -> None:
        if False:
            return 10
        'Move a bundle produced by the operator to its outqueue.'
        self.outqueue.append(ref)
        self.num_completed_tasks += 1
        if self.progress_bar:
            self.progress_bar.update(1, self.op._estimated_output_blocks)

    def refresh_progress_bar(self) -> None:
        if False:
            while True:
                i = 10
        'Update the console with the latest operator progress.'
        if self.progress_bar:
            self.progress_bar.set_description(self.summary_str())

    def summary_str(self) -> str:
        if False:
            while True:
                i = 10
        queued = self.num_queued() + self.op.internal_queue_size()
        active = self.op.num_active_tasks()
        desc = f'- {self.op.name}: {active} active, {queued} queued'
        mem = memory_string((self.op.current_resource_usage().object_store_memory or 0) + self.inqueue_memory_usage())
        desc += f', {mem} objects'
        suffix = self.op.progress_str()
        if suffix:
            desc += f', {suffix}'
        return desc

    def dispatch_next_task(self) -> None:
        if False:
            while True:
                i = 10
        'Move a bundle from the operator inqueue to the operator itself.'
        for (i, inqueue) in enumerate(self.inqueues):
            if inqueue:
                self.op.add_input(inqueue.popleft(), input_index=i)
                return
        assert False, 'Nothing to dispatch'

    def get_output_blocking(self, output_split_idx: Optional[int]) -> MaybeRefBundle:
        if False:
            i = 10
            return i + 15
        "Get an item from this node's output queue, blocking as needed.\n\n        Returns:\n            The RefBundle from the output queue, or an error / end of stream indicator.\n        "
        while True:
            try:
                if output_split_idx is None:
                    return self.outqueue.popleft()
                for i in range(len(self.outqueue)):
                    bundle = self.outqueue[i]
                    if bundle is None or isinstance(bundle, Exception):
                        return bundle
                    elif bundle.output_split_idx == output_split_idx:
                        self.outqueue.remove(bundle)
                        return bundle
            except IndexError:
                pass
            time.sleep(0.01)

    def inqueue_memory_usage(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        "Return the object store memory of this operator's inqueue."
        total = 0
        for (op, inq) in zip(self.op.input_dependencies, self.inqueues):
            if not isinstance(op, InputDataBuffer):
                total += self._queue_memory_usage(inq)
        return total

    def outqueue_memory_usage(self) -> int:
        if False:
            while True:
                i = 10
        "Return the object store memory of this operator's outqueue."
        return self._queue_memory_usage(self.outqueue)

    def _queue_memory_usage(self, queue: Deque[RefBundle]) -> int:
        if False:
            for i in range(10):
                print('nop')
        "Sum the object store memory usage in this queue.\n\n        Note: Python's deque isn't truly thread-safe since it raises RuntimeError\n        if it detects concurrent iteration. Hence we don't use its iterator but\n        manually index into it.\n        "
        object_store_memory = 0
        for i in range(len(queue)):
            try:
                bundle = queue[i]
                object_store_memory += bundle.size_bytes()
            except IndexError:
                break
        return object_store_memory

    def outqueue_num_blocks(self) -> int:
        if False:
            return 10
        "Return the number of blocks in this operator's outqueue."
        num_blocks = 0
        for i in range(len(self.outqueue)):
            try:
                bundle = self.outqueue[i]
                if isinstance(bundle, RefBundle):
                    num_blocks += len(bundle.blocks)
            except IndexError:
                break
        return len(self.outqueue)

def build_streaming_topology(dag: PhysicalOperator, options: ExecutionOptions) -> Tuple[Topology, int]:
    if False:
        while True:
            i = 10
    'Instantiate the streaming operator state topology for the given DAG.\n\n    This involves creating the operator state for each operator in the DAG,\n    registering it with this class, and wiring up the inqueues/outqueues of\n    dependent operator states.\n\n    Args:\n        dag: The operator DAG to instantiate.\n        options: The execution options to use to start operators.\n\n    Returns:\n        The topology dict holding the streaming execution state.\n        The number of progress bars initialized so far.\n    '
    topology: Topology = {}

    def setup_state(op: PhysicalOperator) -> OpState:
        if False:
            print('Hello World!')
        if op in topology:
            raise ValueError('An operator can only be present in a topology once.')
        inqueues = []
        for (i, parent) in enumerate(op.input_dependencies):
            parent_state = setup_state(parent)
            inqueues.append(parent_state.outqueue)
        op_state = OpState(op, inqueues)
        topology[op] = op_state
        op.start(options)
        return op_state
    setup_state(dag)
    i = 1
    for op_state in list(topology.values()):
        if not isinstance(op_state.op, InputDataBuffer):
            i += op_state.initialize_progress_bars(i, options.verbose_progress)
    return (topology, i)

def process_completed_tasks(topology: Topology, backpressure_policies: List[BackpressurePolicy]) -> None:
    if False:
        print('Hello World!')
    'Process any newly completed tasks. To update operator\n    states, call `update_operator_states()` afterwards.'
    active_tasks: Dict[Waitable, Tuple[OpState, OpTask]] = {}
    for (op, state) in topology.items():
        for task in op.get_active_tasks():
            active_tasks[task.get_waitable()] = (state, task)
    max_blocks_to_read_per_op: Dict[OpState, int] = {}
    for policy in backpressure_policies:
        non_empty = len(max_blocks_to_read_per_op) > 0
        max_blocks_to_read_per_op = policy.calculate_max_blocks_to_read_per_op(topology)
        if non_empty and len(max_blocks_to_read_per_op) > 0:
            raise ValueError('At most one backpressure policy that implements calculate_max_blocks_to_read_per_op() can be used at a time.')
    if active_tasks:
        (ready, _) = ray.wait(list(active_tasks.keys()), num_returns=len(active_tasks), fetch_local=False, timeout=0.1)
        for ref in ready:
            (state, task) = active_tasks.pop(ref)
            if isinstance(task, DataOpTask):
                num_blocks_read = task.on_data_ready(max_blocks_to_read_per_op.get(state, None))
                if state in max_blocks_to_read_per_op:
                    max_blocks_to_read_per_op[state] -= num_blocks_read
            else:
                assert isinstance(task, MetadataOpTask)
                task.on_task_finished()
    for (op, op_state) in topology.items():
        while op.has_next():
            op_state.add_output(op.get_next())

def update_operator_states(topology: Topology) -> None:
    if False:
        while True:
            i = 10
    'Update operator states accordingly for newly completed tasks.\n    Should be called after `process_completed_tasks()`.'
    for (op, op_state) in topology.items():
        if op_state.inputs_done_called:
            continue
        all_inputs_done = True
        for (idx, dep) in enumerate(op.input_dependencies):
            if dep.completed() and (not topology[dep].outqueue):
                if not op_state.input_done_called[idx]:
                    op.input_done(idx)
                    op_state.input_done_called[idx] = True
            else:
                all_inputs_done = False
        if all_inputs_done:
            op.all_inputs_done()
            op_state.inputs_done_called = True
    for (op, op_state) in reversed(list(topology.items())):
        if op_state.dependents_completed_called:
            continue
        dependents_completed = len(op.output_dependencies) > 0 and all((not dep.need_more_inputs() for dep in op.output_dependencies))
        if dependents_completed:
            op.all_dependents_complete()
            op_state.dependents_completed_called = True

def select_operator_to_run(topology: Topology, cur_usage: TopologyResourceUsage, limits: ExecutionResources, backpressure_policies: List[BackpressurePolicy], ensure_at_least_one_running: bool, execution_id: str, autoscaling_state: AutoscalingState) -> Optional[PhysicalOperator]:
    if False:
        return 10
    'Select an operator to run, if possible.\n\n    The objective of this function is to maximize the throughput of the overall\n    pipeline, subject to defined memory and parallelism limits.\n\n    This is currently implemented by applying backpressure on operators that are\n    producing outputs faster than they are consuming them `len(outqueue)`, as well as\n    operators with a large number of running tasks `num_processing()`.\n\n    Note that memory limits also apply to the outqueue of the output operator. This\n    provides backpressure if the consumer is slow. However, once a bundle is returned\n    to the user, it is no longer tracked.\n    '
    assert isinstance(cur_usage, TopologyResourceUsage), cur_usage
    ops = []
    for (op, state) in topology.items():
        under_resource_limits = _execution_allowed(op, cur_usage, limits)
        if op.need_more_inputs() and state.num_queued() > 0 and op.should_add_input() and under_resource_limits and (not op.completed()) and all((p.can_add_input(op) for p in backpressure_policies)):
            ops.append(op)
        op.notify_resource_usage(state.num_queued(), under_resource_limits)
    if not ops and any((state.num_queued() > 0 for state in topology.values())):
        now = time.time()
        if now > autoscaling_state.last_request_ts + MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS:
            autoscaling_state.last_request_ts = now
            _try_to_scale_up_cluster(topology, execution_id)
    if ensure_at_least_one_running and (not ops) and all((op.num_active_tasks() == 0 for op in topology)):
        ops = [op for (op, state) in topology.items() if op.need_more_inputs() and state.num_queued() > 0 and (not op.completed())]
    if not ops:
        return None
    return min(ops, key=lambda op: (not op.throttling_disabled(), len(topology[op].outqueue) + topology[op].num_processing()))

def _try_to_scale_up_cluster(topology: Topology, execution_id: str):
    if False:
        i = 10
        return i + 15
    "Try to scale up the cluster to accomodate the provided in-progress workload.\n\n    This makes a resource request to Ray's autoscaler consisting of the current,\n    aggregate usage of all operators in the DAG + the incremental usage of all operators\n    that are ready for dispatch (i.e. that have inputs queued). If the autoscaler were\n    to grant this resource request, it would allow us to dispatch one task for every\n    ready operator.\n\n    Note that this resource request does not take the global resource limits or the\n    liveness policy into account; it only tries to make the existing resource usage +\n    one more task per ready operator feasible in the cluster.\n\n    Args:\n        topology: The execution state of the in-progress workload for which we wish to\n            request more resources.\n    "
    resource_request = []

    def to_bundle(resource: ExecutionResources) -> Dict:
        if False:
            print('Hello World!')
        req = {}
        if resource.cpu:
            req['CPU'] = math.ceil(resource.cpu)
        if resource.gpu:
            req['GPU'] = math.ceil(resource.gpu)
        return req
    for (op, state) in topology.items():
        per_task_resource = op.incremental_resource_usage()
        task_bundle = to_bundle(per_task_resource)
        resource_request.extend([task_bundle] * op.num_active_tasks())
        if state.num_queued() > 0:
            resource_request.append(task_bundle)
    actor = get_or_create_autoscaling_requester_actor()
    actor.request_resources.remote(resource_request, execution_id)

def _execution_allowed(op: PhysicalOperator, global_usage: TopologyResourceUsage, global_limits: ExecutionResources) -> bool:
    if False:
        while True:
            i = 10
    "Return whether an operator is allowed to execute given resource usage.\n\n    Operators are throttled globally based on CPU and GPU limits for the stream.\n\n    For an N operator DAG, we only throttle the kth operator (in the source-to-sink\n    ordering) on object store utilization if the cumulative object store utilization\n    for the kth operator and every operator downstream from it is greater than\n    k/N * global_limit; i.e., the N - k operator sub-DAG is using more object store\n    memory than it's share.\n\n    Args:\n        op: The operator to check.\n        global_usage: Resource usage across the entire topology.\n        global_limits: Execution resource limits.\n\n    Returns:\n        Whether the op is allowed to run.\n    "
    if op.throttling_disabled():
        return True
    assert isinstance(global_usage, TopologyResourceUsage), global_usage
    global_floored = ExecutionResources(cpu=math.floor(global_usage.overall.cpu or 0), gpu=math.floor(global_usage.overall.gpu or 0), object_store_memory=global_usage.overall.object_store_memory)
    inc = op.incremental_resource_usage()
    if inc.cpu and inc.gpu:
        raise NotImplementedError('Operator incremental resource usage cannot specify both CPU and GPU at the same time, since it may cause deadlock.')
    elif inc.object_store_memory:
        raise NotImplementedError('Operator incremental resource usage must not include memory.')
    inc_indicator = ExecutionResources(cpu=1 if inc.cpu else 0, gpu=1 if inc.gpu else 0, object_store_memory=1 if inc.object_store_memory else 0)
    new_usage = global_floored.add(inc_indicator)
    if new_usage.satisfies_limit(global_limits):
        return True
    global_limits_sans_memory = ExecutionResources(cpu=global_limits.cpu, gpu=global_limits.gpu)
    global_ok_sans_memory = new_usage.satisfies_limit(global_limits_sans_memory)
    downstream_usage = global_usage.downstream_memory_usage[op]
    downstream_limit = global_limits.scale(downstream_usage.topology_fraction)
    downstream_memory_ok = ExecutionResources(object_store_memory=downstream_usage.object_store_memory).satisfies_limit(downstream_limit)
    return global_ok_sans_memory and downstream_memory_ok