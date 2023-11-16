from typing import Callable, TypeVar
from reactivex import Observable
from reactivex import operators as ops
_T = TypeVar('_T')

def do_while_(condition: Callable[[Observable[_T]], bool]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10
    'Repeats source as long as condition holds emulating a do while\n    loop.\n\n    Args:\n        condition: The condition which determines if the source will be\n            repeated.\n\n    Returns:\n        An observable sequence which is repeated as long\n        as the condition holds.\n    '

    def do_while(source: Observable[_T]) -> Observable[_T]:
        if False:
            print('Hello World!')
        return source.pipe(ops.concat(source.pipe(ops.while_do(condition))))
    return do_while
__all__ = ['do_while_']