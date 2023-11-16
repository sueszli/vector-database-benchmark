from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Sequence
from airflow.decorators.base import get_unique_task_id, task_decorator_factory
from airflow.sensors.python import PythonSensor
if TYPE_CHECKING:
    from airflow.decorators.base import TaskDecorator

class DecoratedSensorOperator(PythonSensor):
    """
    Wraps a Python callable and captures args/kwargs when called for execution.

    :param python_callable: A reference to an object that is callable
    :param task_id: task Id
    :param op_args: a list of positional arguments that will get unpacked when
        calling your callable (templated)
    :param op_kwargs: a dictionary of keyword arguments that will get unpacked
        in your function (templated)
    :param kwargs_to_upstream: For certain operators, we might need to upstream certain arguments
        that would otherwise be absorbed by the DecoratedOperator (for example python_callable for the
        PythonOperator). This gives a user the option to upstream kwargs as needed.
    """
    template_fields: Sequence[str] = ('op_args', 'op_kwargs')
    template_fields_renderers: dict[str, str] = {'op_args': 'py', 'op_kwargs': 'py'}
    custom_operator_name = '@task.sensor'
    shallow_copy_attrs: Sequence[str] = ('python_callable',)

    def __init__(self, *, task_id: str, **kwargs) -> None:
        if False:
            print('Hello World!')
        kwargs.pop('multiple_outputs')
        kwargs['task_id'] = get_unique_task_id(task_id, kwargs.get('dag'), kwargs.get('task_group'))
        super().__init__(**kwargs)

def sensor_task(python_callable: Callable | None=None, **kwargs) -> TaskDecorator:
    if False:
        while True:
            i = 10
    '\n    Wrap a function into an Airflow operator.\n\n    Accepts kwargs for operator kwarg. Can be reused in a single DAG.\n    :param python_callable: Function to decorate\n    '
    return task_decorator_factory(python_callable=python_callable, multiple_outputs=False, decorated_operator_class=DecoratedSensorOperator, **kwargs)