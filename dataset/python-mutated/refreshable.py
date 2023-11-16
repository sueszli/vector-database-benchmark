from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, ClassVar, Dict, Generic, List, Optional, Tuple, TypeVar, Union, cast
from typing_extensions import ParamSpec, Self
from .. import background_tasks, core
from ..client import Client
from ..dataclasses import KWONLY_SLOTS
from ..element import Element
from ..helpers import is_coroutine_function
_T = TypeVar('_T')
_P = ParamSpec('_P')

@dataclass(**KWONLY_SLOTS)
class RefreshableTarget:
    container: RefreshableContainer
    refreshable: refreshable
    instance: Any
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    current_target: ClassVar[Optional[RefreshableTarget]] = None
    locals: List[Any] = field(default_factory=list)
    next_index: int = 0

    def run(self, func: Callable[..., Union[_T, Awaitable[_T]]]) -> Union[_T, Awaitable[_T]]:
        if False:
            print('Hello World!')
        'Run the function and return the result.'
        RefreshableTarget.current_target = self
        self.next_index = 0
        if is_coroutine_function(func):

            async def wait_for_result() -> Any:
                with self.container:
                    if self.instance is None:
                        result = func(*self.args, **self.kwargs)
                    else:
                        result = func(self.instance, *self.args, **self.kwargs)
                    assert isinstance(result, Awaitable)
                    return await result
            return wait_for_result()
        else:
            with self.container:
                if self.instance is None:
                    return func(*self.args, **self.kwargs)
                else:
                    return func(self.instance, *self.args, **self.kwargs)

class RefreshableContainer(Element, component='refreshable.js'):
    pass

class refreshable(Generic[_P, _T]):

    def __init__(self, func: Callable[_P, Union[_T, Awaitable[_T]]]) -> None:
        if False:
            while True:
                i = 10
        'Refreshable UI functions\n\n        The `@ui.refreshable` decorator allows you to create functions that have a `refresh` method.\n        This method will automatically delete all elements created by the function and recreate them.\n        '
        self.func = func
        self.instance = None
        self.targets: List[RefreshableTarget] = []

    def __get__(self, instance, _) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.instance = instance
        return self

    def __getattribute__(self, __name: str) -> Any:
        if False:
            print('Hello World!')
        attribute = object.__getattribute__(self, __name)
        if __name == 'refresh':

            def refresh(*args: Any, _instance=self.instance, **kwargs: Any) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.instance = _instance
                attribute(*args, **kwargs)
            return refresh
        return attribute

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> Union[_T, Awaitable[_T]]:
        if False:
            return 10
        self.prune()
        target = RefreshableTarget(container=RefreshableContainer(), refreshable=self, instance=self.instance, args=args, kwargs=kwargs)
        self.targets.append(target)
        return target.run(self.func)

    def refresh(self, *args: _P.args, **kwargs: _P.kwargs) -> None:
        if False:
            print('Hello World!')
        'Refresh the UI elements created by this function.'
        self.prune()
        for target in self.targets:
            if target.instance != self.instance:
                continue
            target.container.clear()
            target.args = args or target.args
            target.kwargs.update(kwargs)
            try:
                result = target.run(self.func)
            except TypeError as e:
                if 'got multiple values for argument' in str(e):
                    function = str(e).split()[0].split('.')[-1]
                    parameter = str(e).split()[-1]
                    raise TypeError(f'{parameter} needs to be consistently passed to {function} either as positional or as keyword argument') from e
                raise
            if is_coroutine_function(self.func):
                assert isinstance(result, Awaitable)
                if core.loop and core.loop.is_running():
                    background_tasks.create(result)
                else:
                    core.app.on_startup(result)

    def prune(self) -> None:
        if False:
            print('Hello World!')
        'Remove all targets that are no longer on a page with a client connection.\n\n        This method is called automatically before each refresh.\n        '
        self.targets = [target for target in self.targets if target.container.client.id in Client.instances and target.container.id in target.container.client.elements]

def state(value: Any) -> Tuple[Any, Callable[[Any], None]]:
    if False:
        print('Hello World!')
    'Create a state variable that automatically updates its refreshable UI container.\n\n    :param value: The initial value of the state variable.\n\n    :return: A tuple containing the current value and a function to update the value.\n    '
    target = cast(RefreshableTarget, RefreshableTarget.current_target)
    if target.next_index >= len(target.locals):
        target.locals.append(value)
    else:
        value = target.locals[target.next_index]

    def set_value(new_value: Any, index=target.next_index) -> None:
        if False:
            while True:
                i = 10
        target.locals[index] = new_value
        target.refreshable.refresh()
    target.next_index += 1
    return (value, set_value)