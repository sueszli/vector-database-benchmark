from __future__ import annotations
import types
from typing import TYPE_CHECKING, Callable
from airflow.decorators import python_task
from airflow.decorators.task_group import _TaskGroupFactory
from airflow.exceptions import AirflowException
from airflow.models import BaseOperator
from airflow.utils.setup_teardown import SetupTeardownContext
if TYPE_CHECKING:
    from airflow import XComArg

def setup_task(func: Callable) -> Callable:
    if False:
        i = 10
        return i + 15
    if isinstance(func, types.FunctionType):
        func = python_task(func)
    if isinstance(func, _TaskGroupFactory):
        raise AirflowException('Task groups cannot be marked as setup or teardown.')
    func.is_setup = True
    return func

def teardown_task(_func=None, *, on_failure_fail_dagrun: bool=False) -> Callable:
    if False:
        print('Hello World!')

    def teardown(func: Callable) -> Callable:
        if False:
            return 10
        if isinstance(func, types.FunctionType):
            func = python_task(func)
        if isinstance(func, _TaskGroupFactory):
            raise AirflowException('Task groups cannot be marked as setup or teardown.')
        func.is_teardown = True
        func.on_failure_fail_dagrun = on_failure_fail_dagrun
        return func
    if _func is None:
        return teardown
    return teardown(_func)

class ContextWrapper(list):
    """A list subclass that has a context manager that pushes setup/teardown tasks to the context."""

    def __init__(self, tasks: list[BaseOperator | XComArg]):
        if False:
            return 10
        self.tasks = tasks
        super().__init__(tasks)

    def __enter__(self):
        if False:
            print('Hello World!')
        operators = []
        for task in self.tasks:
            if isinstance(task, BaseOperator):
                operators.append(task)
                if not task.is_setup and (not task.is_teardown):
                    raise AirflowException('Only setup/teardown tasks can be used as context managers.')
            elif not task.operator.is_setup and (not task.operator.is_teardown):
                raise AirflowException('Only setup/teardown tasks can be used as context managers.')
        if not operators:
            operators = [task.operator for task in self.tasks]
        SetupTeardownContext.push_setup_teardown_task(operators)
        return SetupTeardownContext

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        SetupTeardownContext.set_work_task_roots_and_leaves()
context_wrapper = ContextWrapper