from typing import Callable, Type, TypeVar, Union, cast
from reactivex import Observable, abc, defer
from reactivex import operators as ops
from reactivex.internal.utils import NotSet
from reactivex.typing import Accumulator
_T = TypeVar('_T')
_TState = TypeVar('_TState')

def scan_(accumulator: Accumulator[_TState, _T], seed: Union[_TState, Type[NotSet]]=NotSet) -> Callable[[Observable[_T]], Observable[_TState]]:
    if False:
        print('Hello World!')
    has_seed = seed is not NotSet

    def scan(source: Observable[_T]) -> Observable[_TState]:
        if False:
            print('Hello World!')
        'Partially applied scan operator.\n\n        Applies an accumulator function over an observable sequence and\n        returns each intermediate result.\n\n        Examples:\n            >>> scanned = scan(source)\n\n        Args:\n            source: The observable source to scan.\n\n        Returns:\n            An observable sequence containing the accumulated values.\n        '

        def factory(scheduler: abc.SchedulerBase) -> Observable[_TState]:
            if False:
                while True:
                    i = 10
            has_accumulation = False
            accumulation: _TState = cast(_TState, None)

            def projection(x: _T) -> _TState:
                if False:
                    return 10
                nonlocal has_accumulation
                nonlocal accumulation
                if has_accumulation:
                    accumulation = accumulator(accumulation, x)
                else:
                    accumulation = accumulator(cast(_TState, seed), x) if has_seed else cast(_TState, x)
                    has_accumulation = True
                return accumulation
            return source.pipe(ops.map(projection))
        return defer(factory)
    return scan
__all__ = ['scan_']