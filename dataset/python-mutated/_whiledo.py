import itertools
from asyncio import Future
from typing import Callable, TypeVar, Union
import reactivex
from reactivex import Observable
from reactivex.internal.utils import infinite
from reactivex.typing import Predicate
_T = TypeVar('_T')

def while_do_(condition: Predicate[Observable[_T]]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10

    def while_do(source: Union[Observable[_T], 'Future[_T]']) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')
        'Repeats source as long as condition holds emulating a while\n        loop.\n\n        Args:\n            source: The observable sequence that will be run if the\n                condition function returns true.\n\n        Returns:\n            An observable sequence which is repeated as long as the\n            condition holds.\n        '
        if isinstance(source, Future):
            obs = reactivex.from_future(source)
        else:
            obs = source
        it = itertools.takewhile(condition, (obs for _ in infinite()))
        return reactivex.concat_with_iterable(it)
    return while_do
__all__ = ['while_do_']