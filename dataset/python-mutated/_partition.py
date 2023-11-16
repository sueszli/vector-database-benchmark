from typing import Callable, List, TypeVar
from reactivex import Observable
from reactivex import operators as ops
from reactivex.typing import Predicate, PredicateIndexed
_T = TypeVar('_T')

def partition_(predicate: Predicate[_T]) -> Callable[[Observable[_T]], List[Observable[_T]]]:
    if False:
        i = 10
        return i + 15

    def partition(source: Observable[_T]) -> List[Observable[_T]]:
        if False:
            for i in range(10):
                print('nop')
        'The partially applied `partition` operator.\n\n        Returns two observables which partition the observations of the\n        source by the given function. The first will trigger\n        observations for those values for which the predicate returns\n        true. The second will trigger observations for those values\n        where the predicate returns false. The predicate is executed\n        once for each subscribed observer. Both also propagate all\n        error observations arising from the source and each completes\n        when the source completes.\n\n        Args:\n            source: Source observable to partition.\n\n        Returns:\n            A list of observables. The first triggers when the\n            predicate returns True, and the second triggers when the\n            predicate returns False.\n        '

        def not_predicate(x: _T) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return not predicate(x)
        published = source.pipe(ops.publish(), ops.ref_count())
        return [published.pipe(ops.filter(predicate)), published.pipe(ops.filter(not_predicate))]
    return partition

def partition_indexed_(predicate_indexed: PredicateIndexed[_T]) -> Callable[[Observable[_T]], List[Observable[_T]]]:
    if False:
        for i in range(10):
            print('nop')

    def partition_indexed(source: Observable[_T]) -> List[Observable[_T]]:
        if False:
            return 10
        'The partially applied indexed partition operator.\n\n        Returns two observables which partition the observations of the\n        source by the given function. The first will trigger\n        observations for those values for which the predicate returns\n        true. The second will trigger observations for those values\n        where the predicate returns false. The predicate is executed\n        once for each subscribed observer. Both also propagate all\n        error observations arising from the source and each completes\n        when the source completes.\n\n        Args:\n            source: Source observable to partition.\n\n        Returns:\n            A list of observables. The first triggers when the\n            predicate returns True, and the second triggers when the\n            predicate returns False.\n        '

        def not_predicate_indexed(x: _T, i: int) -> bool:
            if False:
                while True:
                    i = 10
            return not predicate_indexed(x, i)
        published = source.pipe(ops.publish(), ops.ref_count())
        return [published.pipe(ops.filter_indexed(predicate_indexed)), published.pipe(ops.filter_indexed(not_predicate_indexed))]
    return partition_indexed
__all__ = ['partition_', 'partition_indexed_']