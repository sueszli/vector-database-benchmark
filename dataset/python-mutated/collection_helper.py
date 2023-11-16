from typing import Iterable, TypeVar, Callable
_IterType = TypeVar('_IterType')
_IterTargetType = TypeVar('_IterTargetType')

def iter_mapping(iter_: Iterable[_IterType], mapping: Callable[[_IterType], _IterTargetType]):
    if False:
        print('Hello World!')
    '\n    Overview:\n        Map a list of iterable elements to input iteration callable\n    Arguments:\n        - iter_(:obj:`_IterType list`): The list for iteration\n        - mapping (:obj:`Callable [[_IterType], _IterTargetType]`): A callable that maps iterable elements function.\n    Return:\n        - (:obj:`iter_mapping object`): Iteration results\n    Example:\n        >>> iterable_list = [1, 2, 3, 4, 5]\n        >>> _iter = iter_mapping(iterable_list, lambda x: x ** 2)\n        >>> print(list(_iter))\n        [1, 4, 9, 16, 25]\n    '
    for item in iter_:
        yield mapping(item)