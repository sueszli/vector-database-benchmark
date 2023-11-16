import asyncio
from collections import deque, defaultdict
import dataclasses
from dataclasses import field
import logging
from typing import List, Dict, Optional, Set, Deque, Callable
import ray
from ray.workflow.common import TaskID, WorkflowRef, WorkflowTaskRuntimeOptions
from ray.workflow.workflow_context import WorkflowTaskContext
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class TaskExecutionMetadata:
    submit_time: Optional[float] = None
    finish_time: Optional[float] = None
    output_size: Optional[int] = None

    @property
    def duration(self):
        if False:
            i = 10
            return i + 15
        return self.finish_time - self.submit_time

@dataclasses.dataclass
class Task:
    """Data class for a workflow task."""
    task_id: str
    options: WorkflowTaskRuntimeOptions
    user_metadata: Dict
    func_body: Optional[Callable]

    def to_dict(self) -> Dict:
        if False:
            while True:
                i = 10
        return {'task_id': self.task_id, 'task_options': self.options.to_dict(), 'user_metadata': self.user_metadata}

@dataclasses.dataclass
class WorkflowExecutionState:
    """The execution state of a workflow. This dataclass helps with observation
    and debugging."""
    upstream_dependencies: Dict[TaskID, List[TaskID]] = field(default_factory=dict)
    downstream_dependencies: Dict[TaskID, List[TaskID]] = field(default_factory=lambda : defaultdict(list))
    next_continuation: Dict[TaskID, TaskID] = field(default_factory=dict)
    prev_continuation: Dict[TaskID, TaskID] = field(default_factory=dict)
    latest_continuation: Dict[TaskID, TaskID] = field(default_factory=dict)
    continuation_root: Dict[TaskID, TaskID] = field(default_factory=dict)
    tasks: Dict[TaskID, Task] = field(default_factory=dict)
    task_input_args: Dict[TaskID, ray.ObjectRef] = field(default_factory=dict)
    task_context: Dict[TaskID, WorkflowTaskContext] = field(default_factory=dict)
    task_execution_metadata: Dict[TaskID, TaskExecutionMetadata] = field(default_factory=dict)
    task_retries: Dict[TaskID, int] = field(default_factory=lambda : defaultdict(int))
    reference_set: Dict[TaskID, Set[TaskID]] = field(default_factory=lambda : defaultdict(set))
    pending_input_set: Dict[TaskID, Set[TaskID]] = field(default_factory=dict)
    output_map: Dict[TaskID, WorkflowRef] = field(default_factory=dict)
    checkpoint_map: Dict[TaskID, WorkflowRef] = field(default_factory=dict)
    free_outputs: Set[TaskID] = field(default_factory=set)
    frontier_to_run: Deque[TaskID] = field(default_factory=deque)
    frontier_to_run_set: Set[TaskID] = field(default_factory=set)
    running_frontier: Dict[asyncio.Future, WorkflowRef] = field(default_factory=dict)
    running_frontier_set: Set[TaskID] = field(default_factory=set)
    done_tasks: Set[TaskID] = field(default_factory=set)
    output_task_id: Optional[TaskID] = None

    def get_input(self, task_id: TaskID) -> Optional[WorkflowRef]:
        if False:
            i = 10
            return i + 15
        'Get the input. It checks memory first and storage later. It returns None if\n        the input does not exist.\n        '
        return self.output_map.get(task_id, self.checkpoint_map.get(task_id))

    def pop_frontier_to_run(self) -> Optional[TaskID]:
        if False:
            return 10
        'Pop one task to run from the frontier queue.'
        try:
            t = self.frontier_to_run.popleft()
            self.frontier_to_run_set.remove(t)
            return t
        except IndexError:
            return None

    def append_frontier_to_run(self, task_id: TaskID) -> None:
        if False:
            return 10
        'Insert one task to the frontier queue.'
        if task_id not in self.frontier_to_run_set and task_id not in self.running_frontier_set:
            self.frontier_to_run.append(task_id)
            self.frontier_to_run_set.add(task_id)

    def add_dependencies(self, task_id: TaskID, in_dependencies: List[TaskID]) -> None:
        if False:
            print('Hello World!')
        'Add dependencies between a task and it input dependencies.'
        self.upstream_dependencies[task_id] = in_dependencies
        for in_task_id in in_dependencies:
            self.downstream_dependencies[in_task_id].append(task_id)

    def pop_running_frontier(self, fut: asyncio.Future) -> WorkflowRef:
        if False:
            return 10
        'Pop a task from the running frontier.'
        ref = self.running_frontier.pop(fut)
        self.running_frontier_set.remove(ref.task_id)
        return ref

    def insert_running_frontier(self, fut: asyncio.Future, ref: WorkflowRef) -> None:
        if False:
            print('Hello World!')
        'Insert a task to the running frontier.'
        self.running_frontier[fut] = ref
        self.running_frontier_set.add(ref.task_id)

    def append_continuation(self, task_id: TaskID, continuation_task_id: TaskID) -> None:
        if False:
            return 10
        'Append continuation to a task.'
        continuation_root = self.continuation_root.get(task_id, task_id)
        self.prev_continuation[continuation_task_id] = task_id
        self.next_continuation[task_id] = continuation_task_id
        self.continuation_root[continuation_task_id] = continuation_root
        self.latest_continuation[continuation_root] = continuation_task_id

    def merge_state(self, state: 'WorkflowExecutionState') -> None:
        if False:
            return 10
        'Merge with another execution state.'
        self.upstream_dependencies.update(state.upstream_dependencies)
        self.downstream_dependencies.update(state.downstream_dependencies)
        self.task_input_args.update(state.task_input_args)
        self.tasks.update(state.tasks)
        self.task_context.update(state.task_context)
        self.output_map.update(state.output_map)
        self.checkpoint_map.update(state.checkpoint_map)

    def construct_scheduling_plan(self, task_id: TaskID) -> None:
        if False:
            i = 10
            return i + 15
        'Analyze upstream dependencies of a task to construct the scheduling plan.'
        if self.get_input(task_id) is not None:
            return
        visited_nodes = set()
        dag_visit_queue = deque([task_id])
        while dag_visit_queue:
            tid = dag_visit_queue.popleft()
            if tid in visited_nodes:
                continue
            visited_nodes.add(tid)
            self.pending_input_set[tid] = set()
            for in_task_id in self.upstream_dependencies[tid]:
                self.reference_set[in_task_id].add(tid)
                task_input = self.get_input(in_task_id)
                if task_input is None:
                    self.pending_input_set[tid].add(in_task_id)
                    dag_visit_queue.append(in_task_id)
            if tid in self.latest_continuation:
                if self.pending_input_set[tid]:
                    raise ValueError('A task that already returns a continuation cannot be pending.')
                self.construct_scheduling_plan(self.latest_continuation[tid])
            elif not self.pending_input_set[tid]:
                self.append_frontier_to_run(tid)

    def init_context(self, context: WorkflowTaskContext) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize the context of all tasks.'
        for (task_id, task) in self.tasks.items():
            options = task.options
            self.task_context.setdefault(task_id, dataclasses.replace(context, task_id=task_id, creator_task_id=context.task_id, checkpoint=options.checkpoint, catch_exceptions=options.catch_exceptions))