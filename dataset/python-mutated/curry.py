from functools import partial as _partial
from functools import wraps
from inspect import BoundArguments, Signature
from typing import Any, Callable, Tuple, TypeVar, Union
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ReturnType = TypeVar('_ReturnType')

def partial(func: Callable[..., _ReturnType], *args: Any, **kwargs: Any) -> Callable[..., _ReturnType]:
    if False:
        print('Hello World!')
    '\n    Typed partial application.\n\n    It is just a ``functools.partial`` wrapper with better typing support.\n\n    We use a custom ``mypy`` plugin to make sure types are correct.\n    Otherwise, it is currently impossible to properly type this function.\n\n    .. code:: python\n\n      >>> from returns.curry import partial\n\n      >>> def sum_two_numbers(first: int, second: int) -> int:\n      ...     return first + second\n\n      >>> sum_with_ten = partial(sum_two_numbers, 10)\n      >>> assert sum_with_ten(2) == 12\n      >>> assert sum_with_ten(-5) == 5\n\n    See also:\n        - https://docs.python.org/3/library/functools.html#functools.partial\n\n    '
    return _partial(func, *args, **kwargs)

def curry(function: Callable[..., _ReturnType]) -> Callable[..., _ReturnType]:
    if False:
        i = 10
        return i + 15
    "\n    Typed currying decorator.\n\n    Currying is a conception from functional languages that does partial\n    applying. That means that if we pass one argument in a function that\n    gets 2 or more arguments, we'll get a new function that remembers all\n    previously passed arguments. Then we can pass remaining arguments, and\n    the function will be executed.\n\n    :func:`~partial` function does a similar thing,\n    but it does partial application exactly once.\n    ``curry`` is a bit smarter and will do partial\n    application until enough arguments passed.\n\n    If wrong arguments are passed, ``TypeError`` will be raised immediately.\n\n    We use a custom ``mypy`` plugin to make sure types are correct.\n    Otherwise, it is currently impossible to properly type this function.\n\n    .. code:: pycon\n\n      >>> from returns.curry import curry\n\n      >>> @curry\n      ... def divide(number: int, by: int) -> float:\n      ...     return number / by\n\n      >>> divide(1)  # doesn't call the func and remembers arguments\n      <function divide at ...>\n      >>> assert divide(1)(by=10) == 0.1  # calls the func when possible\n      >>> assert divide(1)(10) == 0.1  # calls the func when possible\n      >>> assert divide(1, by=10) == 0.1  # or call the func like always\n\n    Here are several examples with wrong arguments:\n\n    .. code:: pycon\n\n      >>> divide(1, 2, 3)\n      Traceback (most recent call last):\n        ...\n      TypeError: too many positional arguments\n\n      >>> divide(a=1)\n      Traceback (most recent call last):\n        ...\n      TypeError: got an unexpected keyword argument 'a'\n\n    Limitations:\n\n    - It is kinda slow. Like 100 times slower than a regular function call.\n    - It does not work with several builtins like ``str``, ``int``,\n      and possibly other ``C`` defined callables\n    - ``*args`` and ``**kwargs`` are not supported\n      and we use ``Any`` as a fallback\n    - Support of arguments with default values is very limited,\n      because we cannot be totally sure which case we are using:\n      with the default value or without it, be careful\n    - We use a custom ``mypy`` plugin to make types correct,\n      otherwise, it is currently impossible\n    - It might not work as expected with curried ``Klass().method``,\n      it might generate invalid method signature\n      (looks like a bug in ``mypy``)\n    - It is probably a bad idea to ``curry`` a function with lots of arguments,\n      because you will end up with lots of overload functions,\n      that you won't be able to understand.\n      It might also be slow during the typecheck\n    - Currying of ``__init__`` does not work because of the bug in ``mypy``:\n      https://github.com/python/mypy/issues/8801\n\n    We expect people to use this tool responsibly\n    when they know that they are doing.\n\n    See also:\n        - https://en.wikipedia.org/wiki/Currying\n        - https://stackoverflow.com/questions/218025/\n\n    "
    argspec = Signature.from_callable(function).bind_partial()

    def decorator(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return _eager_curry(function, argspec, args, kwargs)
    return wraps(function)(decorator)

def _eager_curry(function: Callable[..., _ReturnType], argspec, args: tuple, kwargs: dict) -> Union[_ReturnType, Callable[..., _ReturnType]]:
    if False:
        return 10
    '\n    Internal ``curry`` implementation.\n\n    The interesting part about it is that it return the result\n    or a new callable that will return a result at some point.\n    '
    (intermediate, full_args) = _intermediate_argspec(argspec, args, kwargs)
    if full_args is not None:
        return function(*full_args[0], **full_args[1])

    def decorator(*inner_args, **inner_kwargs):
        if False:
            return 10
        return _eager_curry(function, intermediate, inner_args, inner_kwargs)
    return wraps(function)(decorator)
_ArgSpec = Union[Tuple[None, Tuple[tuple, dict]], Tuple[BoundArguments, None]]

def _intermediate_argspec(argspec: BoundArguments, args: tuple, kwargs: dict) -> _ArgSpec:
    if False:
        return 10
    "\n    That's where ``curry`` magic happens.\n\n    We use ``Signature`` objects from ``inspect`` to bind existing arguments.\n\n    If there's a ``TypeError`` while we ``bind`` the arguments we try again.\n    The second time we try to ``bind_partial`` arguments. It can fail too!\n    It fails when there are invalid arguments\n    or more arguments than we can fit in a function.\n\n    This function is slow. Any optimization ideas are welcome!\n    "
    full_args = argspec.args + args
    full_kwargs = {**argspec.kwargs, **kwargs}
    try:
        argspec.signature.bind(*full_args, **full_kwargs)
    except TypeError:
        return (argspec.signature.bind_partial(*full_args, **full_kwargs), None)
    return (None, (full_args, full_kwargs))