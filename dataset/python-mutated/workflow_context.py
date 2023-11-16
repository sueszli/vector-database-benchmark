import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import ray
from ray._private.ray_logging import configure_log_file, get_worker_log_file_name
from ray.workflow.common import CheckpointModeType, WorkflowStatus
logger = logging.getLogger(__name__)

@dataclass
class WorkflowTaskContext:
    """
    The structure for saving workflow task context. The context provides
    critical info (e.g. where to checkpoint, which is its parent task)
    for the task to execute correctly.
    """
    workflow_id: Optional[str] = None
    task_id: str = ''
    creator_task_id: str = ''
    checkpoint: CheckpointModeType = True
    catch_exceptions: bool = False
_context: Optional[WorkflowTaskContext] = None

@contextmanager
def workflow_task_context(context) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Initialize the workflow task context.\n\n    Args:\n        context: The new context.\n    '
    global _context
    original_context = _context
    try:
        _context = context
        yield
    finally:
        _context = original_context

def get_workflow_task_context() -> Optional[WorkflowTaskContext]:
    if False:
        for i in range(10):
            print('nop')
    return _context

def get_current_task_id() -> str:
    if False:
        i = 10
        return i + 15
    'Get the current workflow task ID. Empty means we are in\n    the workflow job driver.'
    return get_workflow_task_context().task_id

def get_current_workflow_id() -> str:
    if False:
        i = 10
        return i + 15
    assert _context is not None
    return _context.workflow_id

def get_name() -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'{get_current_workflow_id()}@{get_current_task_id()}'

def get_task_status_info(status: WorkflowStatus) -> str:
    if False:
        while True:
            i = 10
    assert _context is not None
    return f'Task status [{status}]\t[{get_name()}]'
_in_workflow_execution = False

@contextmanager
def workflow_execution() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Scope for workflow task execution.'
    global _in_workflow_execution
    try:
        _in_workflow_execution = True
        yield
    finally:
        _in_workflow_execution = False

def in_workflow_execution() -> bool:
    if False:
        while True:
            i = 10
    'Whether we are in workflow task execution.'
    global _in_workflow_execution
    return _in_workflow_execution

@contextmanager
def workflow_logging_context(job_id) -> None:
    if False:
        i = 10
        return i + 15
    'Initialize the workflow logging context.\n\n    Workflow executions are running as remote functions from\n    WorkflowManagementActor. Without logging redirection, workflow\n    inner execution logs will be pushed to the driver that initially\n    created WorkflowManagementActor rather than the driver that\n    actually submits the current workflow execution.\n    We use this conext manager to re-configure the log files to send\n    the logs to the correct driver, and to restore the log files once\n    the execution is done.\n\n    Args:\n        job_id: The ID of the job that submits the workflow execution.\n    '
    node = ray._private.worker._global_node
    (original_out_file, original_err_file) = node.get_log_file_handles(get_worker_log_file_name('WORKER'))
    (out_file, err_file) = node.get_log_file_handles(get_worker_log_file_name('WORKER', job_id))
    try:
        configure_log_file(out_file, err_file)
        yield
    finally:
        configure_log_file(original_out_file, original_err_file)