"""
A decorator used to create [workers](/guide/workers).
"""
from __future__ import annotations
from functools import partial, wraps
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Callable, Coroutine, TypeVar, Union, cast, overload
from typing_extensions import ParamSpec, TypeAlias
if TYPE_CHECKING:
    from .worker import Worker
FactoryParamSpec = ParamSpec('FactoryParamSpec')
DecoratorParamSpec = ParamSpec('DecoratorParamSpec')
ReturnType = TypeVar('ReturnType')
Decorator: TypeAlias = Callable[[Union[Callable[DecoratorParamSpec, ReturnType], Callable[DecoratorParamSpec, Coroutine[None, None, ReturnType]]]], Callable[DecoratorParamSpec, 'Worker[ReturnType]']]

class WorkerDeclarationError(Exception):
    """An error in the declaration of a worker method."""

@overload
def work(method: Callable[FactoryParamSpec, Coroutine[None, None, ReturnType]], *, name: str='', group: str='default', exit_on_error: bool=True, exclusive: bool=False, description: str | None=None, thread: bool=False) -> Callable[FactoryParamSpec, 'Worker[ReturnType]']:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def work(method: Callable[FactoryParamSpec, ReturnType], *, name: str='', group: str='default', exit_on_error: bool=True, exclusive: bool=False, description: str | None=None, thread: bool=False) -> Callable[FactoryParamSpec, 'Worker[ReturnType]']:
    if False:
        return 10
    ...

@overload
def work(*, name: str='', group: str='default', exit_on_error: bool=True, exclusive: bool=False, description: str | None=None, thread: bool=False) -> Decorator[..., ReturnType]:
    if False:
        print('Hello World!')
    ...

def work(method: Callable[FactoryParamSpec, ReturnType] | Callable[FactoryParamSpec, Coroutine[None, None, ReturnType]] | None=None, *, name: str='', group: str='default', exit_on_error: bool=True, exclusive: bool=False, description: str | None=None, thread: bool=False) -> Callable[FactoryParamSpec, Worker[ReturnType]] | Decorator:
    if False:
        while True:
            i = 10
    'A decorator used to create [workers](/guide/workers).\n\n    Args:\n        method: A function or coroutine.\n        name: A short string to identify the worker (in logs and debugging).\n        group: A short string to identify a group of workers.\n        exit_on_error: Exit the app if the worker raises an error. Set to `False` to suppress exceptions.\n        exclusive: Cancel all workers in the same group.\n        description: Readable description of the worker for debugging purposes.\n            By default, it uses a string representation of the decorated method\n            and its arguments.\n        thread: Mark the method as a thread worker.\n    '

    def decorator(method: Callable[DecoratorParamSpec, ReturnType] | Callable[DecoratorParamSpec, Coroutine[None, None, ReturnType]]) -> Callable[DecoratorParamSpec, Worker[ReturnType]]:
        if False:
            for i in range(10):
                print('nop')
        'The decorator.'
        if not iscoroutinefunction(method) and (not thread):
            raise WorkerDeclarationError('Can not create a worker from a non-async function unless `thread=True` is set on the work decorator.')

        @wraps(method)
        def decorated(*args: DecoratorParamSpec.args, **kwargs: DecoratorParamSpec.kwargs) -> Worker[ReturnType]:
            if False:
                while True:
                    i = 10
            'The replaced callable.'
            from .dom import DOMNode
            self = args[0]
            assert isinstance(self, DOMNode)
            if description is not None:
                debug_description = description
            else:
                try:
                    positional_arguments = ', '.join((repr(arg) for arg in args[1:]))
                    keyword_arguments = ', '.join((f'{name}={value!r}' for (name, value) in kwargs.items()))
                    tokens = [positional_arguments, keyword_arguments]
                    debug_description = f"{method.__name__}({', '.join((token for token in tokens if token))})"
                except Exception:
                    debug_description = '<worker>'
            worker = cast('Worker[ReturnType]', self.run_worker(partial(method, *args, **kwargs), name=name or method.__name__, group=group, description=debug_description, exclusive=exclusive, exit_on_error=exit_on_error, thread=thread))
            return worker
        return decorated
    if method is None:
        return decorator
    else:
        return decorator(method)