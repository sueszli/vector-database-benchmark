from typing import Any, Callable, Tuple
import reactivex
from reactivex import Observable

def fork_join_(*args: Observable[Any]) -> Callable[[Observable[Any]], Observable[Tuple[Any, ...]]]:
    if False:
        print('Hello World!')

    def fork_join(source: Observable[Any]) -> Observable[Tuple[Any, ...]]:
        if False:
            i = 10
            return i + 15
        'Wait for observables to complete and then combine last values\n        they emitted into a tuple. Whenever any of that observables\n        completes without emitting any value, result sequence will\n        complete at that moment as well.\n\n        Examples:\n            >>> obs = fork_join(source)\n\n        Returns:\n            An observable sequence containing the result of combining\n            last element from each source in given sequence.\n        '
        return reactivex.fork_join(source, *args)
    return fork_join
__all__ = ['fork_join_']