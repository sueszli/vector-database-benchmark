import io
import logging
import sys
import warnings
from builtins import print
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Optional, Union
import prefect
from prefect.exceptions import MissingContextError
if TYPE_CHECKING:
    from prefect.client.schemas import FlowRun as ClientFlowRun
    from prefect.client.schemas.objects import FlowRun, TaskRun
    from prefect.context import RunContext
    from prefect.flows import Flow
    from prefect.tasks import Task

class PrefectLogAdapter(logging.LoggerAdapter):
    """
    Adapter that ensures extra kwargs are passed through correctly; without this
    the `extra` fields set on the adapter would overshadow any provided on a
    log-by-log basis.

    See https://bugs.python.org/issue32732 — the Python team has declared that this is
    not a bug in the LoggingAdapter and subclassing is the intended workaround.
    """

    def process(self, msg, kwargs):
        if False:
            print('Hello World!')
        kwargs['extra'] = {**(self.extra or {}), **(kwargs.get('extra') or {})}
        from prefect._internal.compatibility.deprecated import PrefectDeprecationWarning, generate_deprecation_message
        if 'send_to_orion' in kwargs['extra']:
            warnings.warn(generate_deprecation_message('The "send_to_orion" option', start_date='May 2023', help='Use "send_to_api" instead.'), PrefectDeprecationWarning, stacklevel=4)
        return (msg, kwargs)

    def getChild(self, suffix: str, extra: Optional[Dict[str, str]]=None) -> 'PrefectLogAdapter':
        if False:
            i = 10
            return i + 15
        if extra is None:
            extra = {}
        return PrefectLogAdapter(self.logger.getChild(suffix), extra={**self.extra, **extra})

@lru_cache()
def get_logger(name: str=None) -> logging.Logger:
    if False:
        print('Hello World!')
    '\n    Get a `prefect` logger. These loggers are intended for internal use within the\n    `prefect` package.\n\n    See `get_run_logger` for retrieving loggers for use within task or flow runs.\n    By default, only run-related loggers are connected to the `APILogHandler`.\n    '
    parent_logger = logging.getLogger('prefect')
    if name:
        if not name.startswith(parent_logger.name + '.'):
            logger = parent_logger.getChild(name)
        else:
            logger = logging.getLogger(name)
    else:
        logger = parent_logger
    return logger

def get_run_logger(context: 'RunContext'=None, **kwargs: str) -> Union[logging.Logger, logging.LoggerAdapter]:
    if False:
        print('Hello World!')
    '\n    Get a Prefect logger for the current task run or flow run.\n\n    The logger will be named either `prefect.task_runs` or `prefect.flow_runs`.\n    Contextual data about the run will be attached to the log records.\n\n    These loggers are connected to the `APILogHandler` by default to send log records to\n    the API.\n\n    Arguments:\n        context: A specific context may be provided as an override. By default, the\n            context is inferred from global state and this should not be needed.\n        **kwargs: Additional keyword arguments will be attached to the log records in\n            addition to the run metadata\n\n    Raises:\n        RuntimeError: If no context can be found\n    '
    task_run_context = prefect.context.TaskRunContext.get()
    flow_run_context = prefect.context.FlowRunContext.get()
    if context:
        if isinstance(context, prefect.context.FlowRunContext):
            flow_run_context = context
        elif isinstance(context, prefect.context.TaskRunContext):
            task_run_context = context
        else:
            raise TypeError(f"Received unexpected type {type(context).__name__!r} for context. Expected one of 'None', 'FlowRunContext', or 'TaskRunContext'.")
    if task_run_context:
        logger = task_run_logger(task_run=task_run_context.task_run, task=task_run_context.task, flow_run=flow_run_context.flow_run if flow_run_context else None, flow=flow_run_context.flow if flow_run_context else None, **kwargs)
    elif flow_run_context:
        logger = flow_run_logger(flow_run=flow_run_context.flow_run, flow=flow_run_context.flow, **kwargs)
    elif get_logger('prefect.flow_run').disabled and get_logger('prefect.task_run').disabled:
        logger = logging.getLogger('null')
    else:
        raise MissingContextError('There is no active flow or task run context.')
    return logger

def flow_run_logger(flow_run: Union['FlowRun', 'ClientFlowRun'], flow: Optional['Flow']=None, **kwargs: str):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a flow run logger with the run's metadata attached.\n\n    Additional keyword arguments can be provided to attach custom data to the log\n    records.\n\n    If the flow run context is available, see `get_run_logger` instead.\n    "
    return PrefectLogAdapter(get_logger('prefect.flow_runs'), extra={**{'flow_run_name': flow_run.name if flow_run else '<unknown>', 'flow_run_id': str(flow_run.id) if flow_run else '<unknown>', 'flow_name': flow.name if flow else '<unknown>'}, **kwargs})

def task_run_logger(task_run: 'TaskRun', task: 'Task'=None, flow_run: 'FlowRun'=None, flow: 'Flow'=None, **kwargs: str):
    if False:
        i = 10
        return i + 15
    "\n    Create a task run logger with the run's metadata attached.\n\n    Additional keyword arguments can be provided to attach custom data to the log\n    records.\n\n    If the task run context is available, see `get_run_logger` instead.\n\n    If only the flow run context is available, it will be used for default values\n    of `flow_run` and `flow`.\n    "
    if not flow_run or not flow:
        flow_run_context = prefect.context.FlowRunContext.get()
        if flow_run_context:
            flow_run = flow_run or flow_run_context.flow_run
            flow = flow or flow_run_context.flow
    return PrefectLogAdapter(get_logger('prefect.task_runs'), extra={**{'task_run_id': str(task_run.id), 'flow_run_id': str(task_run.flow_run_id), 'task_run_name': task_run.name, 'task_name': task.name if task else '<unknown>', 'flow_run_name': flow_run.name if flow_run else '<unknown>', 'flow_name': flow.name if flow else '<unknown>'}, **kwargs})

@contextmanager
def disable_logger(name: str):
    if False:
        return 10
    '\n    Get a logger by name and disables it within the context manager.\n    Upon exiting the context manager, the logger is returned to its\n    original state.\n    '
    logger = logging.getLogger(name=name)
    base_state = logger.disabled
    try:
        logger.disabled = True
        yield
    finally:
        logger.disabled = base_state

@contextmanager
def disable_run_logger():
    if False:
        i = 10
        return i + 15
    '\n    Gets both `prefect.flow_run` and `prefect.task_run` and disables them\n    within the context manager. Upon exiting the context manager, both loggers\n    are returned to its original state.\n    '
    with disable_logger('prefect.flow_run'), disable_logger('prefect.task_run'):
        yield

def print_as_log(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    A patch for `print` to send printed messages to the Prefect run logger.\n\n    If no run is active, `print` will behave as if it were not patched.\n\n    If `print` sends data to a file other than `sys.stdout` or `sys.stderr`, it will\n    not be forwarded to the Prefect logger either.\n    '
    from prefect.context import FlowRunContext, TaskRunContext
    context = TaskRunContext.get() or FlowRunContext.get()
    if not context or not context.log_prints or kwargs.get('file') not in {None, sys.stdout, sys.stderr}:
        return print(*args, **kwargs)
    logger = get_run_logger()
    buffer = io.StringIO()
    kwargs['file'] = buffer
    print(*args, **kwargs)
    logger.info(buffer.getvalue().rstrip())

@contextmanager
def patch_print():
    if False:
        while True:
            i = 10
    '\n    Patches the Python builtin `print` method to use `print_as_log`\n    '
    import builtins
    original = builtins.print
    try:
        builtins.print = print_as_log
        yield
    finally:
        builtins.print = original