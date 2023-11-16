from typing import Callable, Optional, TypeVar, cast
from reactivex import Observable, abc, compose
from reactivex import operators as ops
from reactivex import typing
from reactivex.internal.basic import identity
from reactivex.internal.utils import infinite
from reactivex.typing import Mapper, MapperIndexed
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')

def map_(mapper: Optional[Mapper[_T1, _T2]]=None) -> Callable[[Observable[_T1]], Observable[_T2]]:
    if False:
        i = 10
        return i + 15
    _mapper = mapper or cast(Mapper[_T1, _T2], identity)

    def map(source: Observable[_T1]) -> Observable[_T2]:
        if False:
            for i in range(10):
                print('nop')
        "Partially applied map operator.\n\n        Project each element of an observable sequence into a new form\n        by incorporating the element's index.\n\n        Example:\n            >>> map(source)\n\n        Args:\n            source: The observable source to transform.\n\n        Returns:\n            Returns an observable sequence whose elements are the\n            result of invoking the transform function on each element\n            of the source.\n        "

        def subscribe(obv: abc.ObserverBase[_T2], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                return 10

            def on_next(value: _T1) -> None:
                if False:
                    while True:
                        i = 10
                try:
                    result = _mapper(value)
                except Exception as err:
                    obv.on_error(err)
                else:
                    obv.on_next(result)
            return source.subscribe(on_next, obv.on_error, obv.on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return map

def map_indexed_(mapper_indexed: Optional[MapperIndexed[_T1, _T2]]=None) -> Callable[[Observable[_T1]], Observable[_T2]]:
    if False:
        i = 10
        return i + 15

    def _identity(value: _T1, _: int) -> _T2:
        if False:
            i = 10
            return i + 15
        return cast(_T2, value)
    _mapper_indexed = mapper_indexed or cast(typing.MapperIndexed[_T1, _T2], _identity)
    return compose(ops.zip_with_iterable(infinite()), ops.starmap_indexed(_mapper_indexed))
__all__ = ['map_', 'map_indexed_']