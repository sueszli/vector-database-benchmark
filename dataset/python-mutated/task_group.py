"""Implements the ``@task_group`` function decorator.

When the decorated function is called, a task group will be created to represent
a collection of closely related tasks on the same DAG that should be grouped
together when the DAG is displayed graphically.
"""
from __future__ import annotations
import functools
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Mapping, Sequence, TypeVar, overload
import attr
from airflow.decorators.base import ExpandableFactory
from airflow.models.expandinput import DictOfListsExpandInput, ListOfDictsExpandInput, MappedArgument
from airflow.models.taskmixin import DAGNode
from airflow.models.xcom_arg import XComArg
from airflow.typing_compat import ParamSpec
from airflow.utils.helpers import prevent_duplicates
from airflow.utils.task_group import MappedTaskGroup, TaskGroup
if TYPE_CHECKING:
    from airflow.models.dag import DAG
    from airflow.models.expandinput import OperatorExpandArgument, OperatorExpandKwargsArgument
FParams = ParamSpec('FParams')
FReturn = TypeVar('FReturn', None, DAGNode)
task_group_sig = inspect.signature(TaskGroup.__init__)

@attr.define()
class _TaskGroupFactory(ExpandableFactory, Generic[FParams, FReturn]):
    function: Callable[FParams, FReturn] = attr.ib(validator=attr.validators.is_callable())
    tg_kwargs: dict[str, Any] = attr.ib(factory=dict)
    partial_kwargs: dict[str, Any] = attr.ib(factory=dict)
    _task_group_created: bool = attr.ib(False, init=False)
    tg_class: ClassVar[type[TaskGroup]] = TaskGroup

    @tg_kwargs.validator
    def _validate(self, _, kwargs):
        if False:
            print('Hello World!')
        task_group_sig.bind_partial(**kwargs)

    def __attrs_post_init__(self):
        if False:
            print('Hello World!')
        self.tg_kwargs.setdefault('group_id', self.function.__name__)

    def __del__(self):
        if False:
            i = 10
            return i + 15
        if self.partial_kwargs and (not self._task_group_created):
            try:
                group_id = repr(self.tg_kwargs['group_id'])
            except KeyError:
                group_id = f'at {hex(id(self))}'
            warnings.warn(f'Partial task group {group_id} was never mapped!')

    def __call__(self, *args: FParams.args, **kwargs: FParams.kwargs) -> DAGNode:
        if False:
            return 10
        'Instantiate the task group.\n\n        This uses the wrapped function to create a task group. Depending on the\n        return type of the wrapped function, this either returns the last task\n        in the group, or the group itself, to support task chaining.\n        '
        return self._create_task_group(TaskGroup, *args, **kwargs)

    def _create_task_group(self, tg_factory: Callable[..., TaskGroup], *args: Any, **kwargs: Any) -> DAGNode:
        if False:
            print('Hello World!')
        with tg_factory(add_suffix_on_collision=True, **self.tg_kwargs) as task_group:
            if self.function.__doc__ and (not task_group.tooltip):
                task_group.tooltip = self.function.__doc__
            retval = self.function(*args, **kwargs)
        self._task_group_created = True
        if retval is not None:
            return retval
        return task_group

    def override(self, **kwargs: Any) -> _TaskGroupFactory[FParams, FReturn]:
        if False:
            while True:
                i = 10
        return attr.evolve(self, tg_kwargs={**self.tg_kwargs, **kwargs})

    def partial(self, **kwargs: Any) -> _TaskGroupFactory[FParams, FReturn]:
        if False:
            return 10
        self._validate_arg_names('partial', kwargs)
        prevent_duplicates(self.partial_kwargs, kwargs, fail_reason='duplicate partial')
        kwargs.update(self.partial_kwargs)
        return attr.evolve(self, partial_kwargs=kwargs)

    def expand(self, **kwargs: OperatorExpandArgument) -> DAGNode:
        if False:
            for i in range(10):
                print('nop')
        if not kwargs:
            raise TypeError('no arguments to expand against')
        self._validate_arg_names('expand', kwargs)
        prevent_duplicates(self.partial_kwargs, kwargs, fail_reason='mapping already partial')
        expand_input = DictOfListsExpandInput(kwargs)
        return self._create_task_group(functools.partial(MappedTaskGroup, expand_input=expand_input), **self.partial_kwargs, **{k: MappedArgument(input=expand_input, key=k) for k in kwargs})

    def expand_kwargs(self, kwargs: OperatorExpandKwargsArgument) -> DAGNode:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(kwargs, Sequence):
            for item in kwargs:
                if not isinstance(item, (XComArg, Mapping)):
                    raise TypeError(f'expected XComArg or list[dict], not {type(kwargs).__name__}')
        elif not isinstance(kwargs, XComArg):
            raise TypeError(f'expected XComArg or list[dict], not {type(kwargs).__name__}')
        function_has_vararg = any((v.kind == inspect.Parameter.VAR_POSITIONAL or v.kind == inspect.Parameter.VAR_KEYWORD for v in self.function_signature.parameters.values()))
        if function_has_vararg:
            raise TypeError('calling expand_kwargs() on task group function with * or ** is not supported')
        map_kwargs = (k for k in self.function_signature.parameters if k not in self.partial_kwargs)
        expand_input = ListOfDictsExpandInput(kwargs)
        return self._create_task_group(functools.partial(MappedTaskGroup, expand_input=expand_input), **self.partial_kwargs, **{k: MappedArgument(input=expand_input, key=k) for k in map_kwargs})

@overload
def task_group(group_id: str | None=None, prefix_group_id: bool=True, parent_group: TaskGroup | None=None, dag: DAG | None=None, default_args: dict[str, Any] | None=None, tooltip: str='', ui_color: str='CornflowerBlue', ui_fgcolor: str='#000', add_suffix_on_collision: bool=False) -> Callable[[Callable[FParams, FReturn]], _TaskGroupFactory[FParams, FReturn]]:
    if False:
        print('Hello World!')
    ...

@overload
def task_group(python_callable: Callable[FParams, FReturn]) -> _TaskGroupFactory[FParams, FReturn]:
    if False:
        print('Hello World!')
    ...

def task_group(python_callable=None, **tg_kwargs):
    if False:
        print('Hello World!')
    'Python TaskGroup decorator.\n\n    This wraps a function into an Airflow TaskGroup. When used as the\n    ``@task_group()`` form, all arguments are forwarded to the underlying\n    TaskGroup class. Can be used to parametrize TaskGroup.\n\n    :param python_callable: Function to decorate.\n    :param tg_kwargs: Keyword arguments for the TaskGroup object.\n    '
    if callable(python_callable) and (not tg_kwargs):
        return _TaskGroupFactory(function=python_callable, tg_kwargs=tg_kwargs)
    return functools.partial(_TaskGroupFactory, tg_kwargs=tg_kwargs)