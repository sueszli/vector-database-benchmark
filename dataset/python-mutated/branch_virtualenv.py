from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from airflow.decorators.base import task_decorator_factory
from airflow.decorators.python import _PythonDecoratedOperator
from airflow.operators.python import BranchPythonVirtualenvOperator
if TYPE_CHECKING:
    from airflow.decorators.base import TaskDecorator

class _BranchPythonVirtualenvDecoratedOperator(_PythonDecoratedOperator, BranchPythonVirtualenvOperator):
    """Wraps a Python callable and captures args/kwargs when called for execution."""
    custom_operator_name: str = '@task.branch_virtualenv'

def branch_virtualenv_task(python_callable: Callable | None=None, multiple_outputs: bool | None=None, **kwargs) -> TaskDecorator:
    if False:
        while True:
            i = 10
    '\n    Wrap a python function into a BranchPythonVirtualenvOperator.\n\n    For more information on how to use this operator, take a look at the guide:\n    :ref:`concepts:branching`\n\n    Accepts kwargs for operator kwarg. Can be reused in a single DAG.\n\n    :param python_callable: Function to decorate\n    :param multiple_outputs: if set, function return value will be\n        unrolled to multiple XCom values. Dict will unroll to xcom values with keys as XCom keys.\n        Defaults to False.\n    '
    return task_decorator_factory(python_callable=python_callable, multiple_outputs=multiple_outputs, decorated_operator_class=_BranchPythonVirtualenvDecoratedOperator, **kwargs)