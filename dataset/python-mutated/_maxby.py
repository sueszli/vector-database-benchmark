from typing import Callable, List, Optional, TypeVar
from reactivex import Observable, typing
from reactivex.internal.basic import default_sub_comparer
from ._minby import extrema_by
_T = TypeVar('_T')
_TKey = TypeVar('_TKey')

def max_by_(key_mapper: typing.Mapper[_T, _TKey], comparer: Optional[typing.SubComparer[_TKey]]=None) -> Callable[[Observable[_T]], Observable[List[_T]]]:
    if False:
        i = 10
        return i + 15
    cmp = comparer or default_sub_comparer

    def max_by(source: Observable[_T]) -> Observable[List[_T]]:
        if False:
            for i in range(10):
                print('nop')
        'Partially applied max_by operator.\n\n        Returns the elements in an observable sequence with the maximum\n        key value.\n\n        Examples:\n            >>> res = max_by(source)\n\n        Args:\n            source: The source observable sequence to.\n\n        Returns:\n            An observable sequence containing a list of zero or more\n            elements that have a maximum key value.\n        '
        return extrema_by(source, key_mapper, cmp)
    return max_by
__all__ = ['max_by_']