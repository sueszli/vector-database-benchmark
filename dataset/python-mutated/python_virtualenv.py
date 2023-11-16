from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from airflow.decorators.base import task_decorator_factory
from airflow.decorators.python import _PythonDecoratedOperator
from airflow.operators.python import PythonVirtualenvOperator
if TYPE_CHECKING:
    from airflow.decorators.base import TaskDecorator

class _PythonVirtualenvDecoratedOperator(_PythonDecoratedOperator, PythonVirtualenvOperator):
    """Wraps a Python callable and captures args/kwargs when called for execution."""
    custom_operator_name: str = '@task.virtualenv'

def virtualenv_task(python_callable: Callable | None=None, multiple_outputs: bool | None=None, **kwargs) -> TaskDecorator:
    if False:
        while True:
            i = 10
    "\n    Wrap a callable into an Airflow operator to run via a Python virtual environment.\n\n    Accepts kwargs for operator kwarg. Can be reused in a single DAG.\n\n    This function is only used only used during type checking or auto-completion.\n\n    :meta private:\n\n    :param python_callable: Function to decorate\n    :param multiple_outputs: If set to True, the decorated function's return value will be unrolled to\n        multiple XCom values. Dict will unroll to XCom values with its keys as XCom keys.\n        Defaults to False.\n    "
    return task_decorator_factory(python_callable=python_callable, multiple_outputs=multiple_outputs, decorated_operator_class=_PythonVirtualenvDecoratedOperator, **kwargs)