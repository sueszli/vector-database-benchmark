from typing import Any, Callable, Optional, TypeVar
from reactivex import Observable, operators
from reactivex.typing import Predicate
from ._lastordefault import last_or_default_async
_T = TypeVar('_T')

def last_(predicate: Optional[Predicate[_T]]=None) -> Callable[[Observable[_T]], Observable[Any]]:
    if False:
        return 10

    def last(source: Observable[_T]) -> Observable[Any]:
        if False:
            print('Hello World!')
        'Partially applied last operator.\n\n        Returns the last element of an observable sequence that\n        satisfies the condition in the predicate if specified, else\n        the last element.\n\n        Examples:\n            >>> res = last(source)\n\n        Args:\n            source: Source observable to get last item from.\n\n        Returns:\n            An observable sequence containing the last element in the\n            observable sequence that satisfies the condition in the\n            predicate.\n        '
        if predicate:
            return source.pipe(operators.filter(predicate), operators.last())
        return last_or_default_async(source, False)
    return last
__all__ = ['last_']