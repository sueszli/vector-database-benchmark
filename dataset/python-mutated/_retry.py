from typing import Callable, Optional, TypeVar
import reactivex
from reactivex import Observable
from reactivex.internal.utils import infinite
_T = TypeVar('_T')

def retry_(retry_count: Optional[int]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')
    'Repeats the source observable sequence the specified number of\n    times or until it successfully terminates. If the retry count is\n    not specified, it retries indefinitely.\n\n    Examples:\n        >>> retried = retry()\n        >>> retried = retry(42)\n\n    Args:\n        retry_count: [Optional] Number of times to retry the sequence.\n            If not provided, retry the sequence indefinitely.\n\n    Returns:\n        An observable sequence producing the elements of the given\n        sequence repeatedly until it terminates successfully.\n    '
    if retry_count is None:
        gen = infinite()
    else:
        gen = range(retry_count)

    def retry(source: Observable[_T]) -> Observable[_T]:
        if False:
            print('Hello World!')
        return reactivex.catch_with_iterable((source for _ in gen))
    return retry
__all__ = ['retry_']