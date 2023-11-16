from asyncio import Future
from typing import Any, Callable, Optional, TypeVar, Union, cast
from reactivex import Observable, from_, from_future
from reactivex import operators as ops
from reactivex.internal.basic import identity
from reactivex.typing import Mapper, MapperIndexed
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')

def _flat_map_internal(source: Observable[_T1], mapper: Optional[Mapper[_T1, Any]]=None, mapper_indexed: Optional[MapperIndexed[_T1, Any]]=None) -> Observable[Any]:
    if False:
        print('Hello World!')

    def projection(x: _T1, i: int) -> Observable[Any]:
        if False:
            while True:
                i = 10
        mapper_result: Any = mapper(x) if mapper else mapper_indexed(x, i) if mapper_indexed else identity
        if isinstance(mapper_result, Future):
            result: Observable[Any] = from_future(cast('Future[Any]', mapper_result))
        elif isinstance(mapper_result, Observable):
            result = mapper_result
        else:
            result = from_(mapper_result)
        return result
    return source.pipe(ops.map_indexed(projection), ops.merge_all())

def flat_map_(mapper: Optional[Mapper[_T1, Observable[_T2]]]=None) -> Callable[[Observable[_T1]], Observable[_T2]]:
    if False:
        for i in range(10):
            print('nop')

    def flat_map(source: Observable[_T1]) -> Observable[_T2]:
        if False:
            i = 10
            return i + 15
        'One of the Following:\n        Projects each element of an observable sequence to an observable\n        sequence and merges the resulting observable sequences into one\n        observable sequence.\n\n        Example:\n            >>> flat_map(source)\n\n        Args:\n            source: Source observable to flat map.\n\n        Returns:\n            An operator function that takes a source observable and returns\n            an observable sequence whose elements are the result of invoking\n            the one-to-many transform function on each element of the\n            input sequence .\n        '
        if callable(mapper):
            ret = _flat_map_internal(source, mapper=mapper)
        else:
            ret = _flat_map_internal(source, mapper=lambda _: mapper)
        return ret
    return flat_map

def flat_map_indexed_(mapper_indexed: Optional[Any]=None) -> Callable[[Observable[Any]], Observable[Any]]:
    if False:
        print('Hello World!')

    def flat_map_indexed(source: Observable[Any]) -> Observable[Any]:
        if False:
            i = 10
            return i + 15
        'One of the Following:\n        Projects each element of an observable sequence to an observable\n        sequence and merges the resulting observable sequences into one\n        observable sequence.\n\n        Example:\n            >>> flat_map_indexed(source)\n\n        Args:\n            source: Source observable to flat map.\n\n        Returns:\n            An observable sequence whose elements are the result of invoking\n            the one-to-many transform function on each element of the input\n            sequence.\n        '
        if callable(mapper_indexed):
            ret = _flat_map_internal(source, mapper_indexed=mapper_indexed)
        else:
            ret = _flat_map_internal(source, mapper=lambda _: mapper_indexed)
        return ret
    return flat_map_indexed

def flat_map_latest_(mapper: Mapper[_T1, Union[Observable[_T2], 'Future[_T2]']]) -> Callable[[Observable[_T1]], Observable[_T2]]:
    if False:
        for i in range(10):
            print('nop')

    def flat_map_latest(source: Observable[_T1]) -> Observable[_T2]:
        if False:
            return 10
        "Projects each element of an observable sequence into a new\n        sequence of observable sequences by incorporating the element's\n        index and then transforms an observable sequence of observable\n        sequences into an observable sequence producing values only\n        from the most recent observable sequence.\n\n        Args:\n            source: Source observable to flat map latest.\n\n        Returns:\n            An observable sequence whose elements are the result of\n            invoking the transform function on each element of source\n            producing an observable of Observable sequences and that at\n            any point in time produces the elements of the most recent\n            inner observable sequence that has been received.\n        "
        return source.pipe(ops.map(mapper), ops.switch_latest())
    return flat_map_latest
__all__ = ['flat_map_', 'flat_map_latest_', 'flat_map_indexed_']