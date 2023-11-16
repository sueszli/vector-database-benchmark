from asyncio import Future
from typing import Callable, TypeVar
from reactivex import Observable, from_future, throw
_T = TypeVar('_T')

def start_async_(function_async: Callable[[], 'Future[_T]']) -> Observable[_T]:
    if False:
        return 10
    try:
        future = function_async()
    except Exception as ex:
        return throw(ex)
    return from_future(future)
__all__ = ['start_async_']