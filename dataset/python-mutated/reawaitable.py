from typing import Awaitable, Callable, Generator, NewType, TypeVar, Union, cast, final
_ValueType = TypeVar('_ValueType')
_FunctionCoroType = TypeVar('_FunctionCoroType', bound=Callable[..., Awaitable])
_Sentinel = NewType('_Sentinel', object)
_sentinel: _Sentinel = cast(_Sentinel, object())

@final
class ReAwaitable(object):
    """
    Allows to write coroutines that can be awaited multiple times.

    It works by actually caching the ``await`` result and reusing it.
    So, in reality we still ``await`` once,
    but pretending to do it multiple times.

    Why is that required? Because otherwise,
    ``Future`` containers would be unusable:

    .. code:: python

      >>> import anyio
      >>> from returns.future import Future
      >>> from returns.io import IO

      >>> async def example(arg: int) -> int:
      ...     return arg

      >>> instance = Future(example(1))
      >>> two = instance.map(lambda x: x + 1)
      >>> zero = instance.map(lambda x: x - 1)

      >>> assert anyio.run(two.awaitable) == IO(2)
      >>> assert anyio.run(zero.awaitable) == IO(0)

    In this example we ``await`` our ``Future`` twice.
    It happens in each ``.map`` call.
    Without this class (that is used inside ``Future``)
    it would result in ``RuntimeError: cannot reuse already awaited coroutine``.

    We try to make this type transparent.
    It should not actually be visible to any of its users.

    """
    __slots__ = ('_coro', '_cache')

    def __init__(self, coro: Awaitable[_ValueType]) -> None:
        if False:
            return 10
        'We need just an awaitable to work with.'
        self._coro = coro
        self._cache: Union[_ValueType, _Sentinel] = _sentinel

    def __await__(self) -> Generator[None, None, _ValueType]:
        if False:
            while True:
                i = 10
        "\n        Allows to use ``await`` multiple times.\n\n        .. code:: python\n\n          >>> import anyio\n          >>> from returns.primitives.reawaitable import ReAwaitable\n\n          >>> async def say_hello() -> str:\n          ...    return 'Hello'\n\n          >>> async def main():\n          ...    instance = ReAwaitable(say_hello())\n          ...    print(await instance)\n          ...    print(await instance)\n          ...    print(await instance)\n\n          >>> anyio.run(main)\n          Hello\n          Hello\n          Hello\n\n        "
        return self._awaitable().__await__()

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        "\n        Formats this type the same way as the coroutine underneath.\n\n        .. code:: python\n\n          >>> from returns.primitives.reawaitable import ReAwaitable\n\n          >>> async def test() -> int:\n          ...    return 1\n\n          >>> assert repr(test) == repr(ReAwaitable(test))\n          >>> repr(ReAwaitable(test))\n          '<function test at 0x...>'\n\n        "
        return repr(self._coro)

    async def _awaitable(self) -> _ValueType:
        """Caches the once awaited value forever."""
        if self._cache is _sentinel:
            self._cache = await self._coro
        return self._cache

def reawaitable(coro: _FunctionCoroType) -> _FunctionCoroType:
    if False:
        for i in range(10):
            print('nop')
    '\n    Allows to decorate coroutine functions to be awaitable multiple times.\n\n    .. code:: python\n\n      >>> import anyio\n      >>> from returns.primitives.reawaitable import reawaitable\n\n      >>> @reawaitable\n      ... async def return_int() -> int:\n      ...    return 1\n\n      >>> async def main():\n      ...    instance = return_int()\n      ...    return await instance + await instance + await instance\n\n      >>> assert anyio.run(main) == 3\n\n    '
    return lambda *args, **kwargs: ReAwaitable(coro(*args, **kwargs))