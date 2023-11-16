from typing import Optional
from collections import deque
from ray.workflow import serialization
from ray.workflow.common import TaskID, WorkflowRef
from ray.workflow.exceptions import WorkflowTaskNotRecoverableError
from ray.workflow import workflow_storage
from ray.workflow.workflow_state import WorkflowExecutionState, Task

def workflow_state_from_storage(workflow_id: str, task_id: Optional[TaskID]) -> WorkflowExecutionState:
    if False:
        i = 10
        return i + 15
    'Try to construct a workflow (task) that recovers the workflow task.\n    If the workflow task already has an output checkpointing file, we return\n    the workflow task id instead.\n\n    Args:\n        workflow_id: The ID of the workflow.\n        task_id: The ID of the output task. If None, it will be the entrypoint of\n            the workflow.\n\n    Returns:\n        A workflow that recovers the task, or the output of the task\n            if it has been checkpointed.\n    '
    reader = workflow_storage.WorkflowStorage(workflow_id)
    if task_id is None:
        task_id = reader.get_entrypoint_task_id()
    state = WorkflowExecutionState(output_task_id=task_id)
    state.output_task_id = task_id
    visited_tasks = set()
    dag_visit_queue = deque([task_id])
    with serialization.objectref_cache():
        while dag_visit_queue:
            task_id: TaskID = dag_visit_queue.popleft()
            if task_id in visited_tasks:
                continue
            visited_tasks.add(task_id)
            r = reader.inspect_task(task_id)
            if not r.is_recoverable():
                raise WorkflowTaskNotRecoverableError(task_id)
            if r.output_object_valid:
                target = state.continuation_root.get(task_id, task_id)
                state.checkpoint_map[target] = WorkflowRef(task_id)
                continue
            if isinstance(r.output_task_id, str):
                state.upstream_dependencies[task_id] = []
                state.append_continuation(task_id, r.output_task_id)
                dag_visit_queue.append(r.output_task_id)
                continue
            state.add_dependencies(task_id, r.workflow_refs)
            state.task_input_args[task_id] = reader.load_task_args(task_id)
            state.tasks[task_id] = Task(task_id='', options=r.task_options, user_metadata={}, func_body=reader.load_task_func_body(task_id))
            dag_visit_queue.extend(r.workflow_refs)
    return state