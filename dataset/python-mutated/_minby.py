from typing import Callable, List, Optional, TypeVar, cast
from reactivex import Observable, abc, typing
from reactivex.internal.basic import default_sub_comparer
_T = TypeVar('_T')
_TKey = TypeVar('_TKey')

def extrema_by(source: Observable[_T], key_mapper: typing.Mapper[_T, _TKey], comparer: typing.SubComparer[_TKey]) -> Observable[List[_T]]:
    if False:
        i = 10
        return i + 15

    def subscribe(observer: abc.ObserverBase[List[_T]], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        has_value = False
        last_key: _TKey = cast(_TKey, None)
        items: List[_T] = []

        def on_next(x: _T) -> None:
            if False:
                return 10
            nonlocal has_value, last_key
            try:
                key = key_mapper(x)
            except Exception as ex:
                observer.on_error(ex)
                return
            comparison = 0
            if not has_value:
                has_value = True
                last_key = key
            else:
                try:
                    comparison = comparer(key, last_key)
                except Exception as ex1:
                    observer.on_error(ex1)
                    return
            if comparison > 0:
                last_key = key
                items[:] = []
            if comparison >= 0:
                items.append(x)

        def on_completed():
            if False:
                while True:
                    i = 10
            observer.on_next(items)
            observer.on_completed()
        return source.subscribe(on_next, observer.on_error, on_completed, scheduler=scheduler)
    return Observable(subscribe)

def min_by_(key_mapper: typing.Mapper[_T, _TKey], comparer: Optional[typing.SubComparer[_TKey]]=None) -> Callable[[Observable[_T]], Observable[List[_T]]]:
    if False:
        while True:
            i = 10
    'The `min_by` operator.\n\n    Returns the elements in an observable sequence with the minimum key\n    value according to the specified comparer.\n\n    Examples:\n        >>> res = min_by(lambda x: x.value)\n        >>> res = min_by(lambda x: x.value, lambda x, y: x - y)\n\n    Args:\n        key_mapper: Key mapper function.\n        comparer: [Optional] Comparer used to compare key values.\n\n    Returns:\n        An observable sequence containing a list of zero or more\n        elements that have a minimum key value.\n    '
    cmp = comparer or default_sub_comparer

    def min_by(source: Observable[_T]) -> Observable[List[_T]]:
        if False:
            i = 10
            return i + 15
        return extrema_by(source, key_mapper, lambda x, y: -cmp(x, y))
    return min_by
__all__ = ['min_by_']