import numpy as np
from typing import Any, Iterator, List, Tuple, Union

class ArrayView:
    """Array of homogeneous elements with sparse indices.
    Interface for working with array as a non-sparse array is available for cases
    when index values are not important.
    """

    def __iter__(self) -> Iterator[Any]:
        if False:
            return 10
        ...

    def keys(self) -> Iterator[int]:
        if False:
            i = 10
            return i + 15
        "Return sparse indices iterator.\n\n        Yields:\n             Array's next sparse index.\n        "
        ...

    def indices(self) -> Iterator[int]:
        if False:
            for i in range(10):
                print('nop')
        "Return sparse indices iterator.\n\n        Yields:\n             Array's next sparse index.\n        "
        ...

    def values(self) -> Iterator[Any]:
        if False:
            while True:
                i = 10
        "Return values iterator.\n\n        Yields:\n             Array's next value.\n        "
        ...

    def items(self) -> Iterator[Tuple[int, Any]]:
        if False:
            i = 10
            return i + 15
        "Return items iterator.\n\n        Yields:\n            Tuple of array's next sparse index and value.\n        "
        ...

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        ...

    def __getitem__(self, idx: Union[int, slice]):
        if False:
            return 10
        ...

    def __setitem__(self, idx: int, val: Any):
        if False:
            for i in range(10):
                print('nop')
        ...

    def sparse_list(self) -> Tuple[List[int], List[Any]]:
        if False:
            while True:
                i = 10
        'Get sparse indices and values as :obj:`list`s.\n        '
        ...

    def indices_list(self) -> List[int]:
        if False:
            print('Hello World!')
        'Get sparse indices as a :obj:`list`.\n        '
        ...

    def values_list(self) -> List[Any]:
        if False:
            i = 10
            return i + 15
        'Get values as a :obj:`list`.\n        '
        ...

    def sparse_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            while True:
                i = 10
        'Get sparse indices and values as numpy arrays.\n        '
        ...

    def indices_numpy(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        'Get sparse indices as numpy array.\n        '
        ...

    def values_numpy(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        'Get values as numpy array.\n        '
        ...

    def tolist(self) -> List[Any]:
        if False:
            return 10
        'Convert to values list'
        ...

    def first(self) -> Tuple[int, Any]:
        if False:
            i = 10
            return i + 15
        'First index and value of the array.\n        '
        ...

    def first_idx(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'First index of the array.\n        '
        ...

    def first_value(self) -> Any:
        if False:
            while True:
                i = 10
        'First value of the array.\n        '
        ...

    def last(self) -> Tuple[int, Any]:
        if False:
            print('Hello World!')
        'Last index and value of the array.\n        '
        ...

    def last_idx(self) -> int:
        if False:
            print('Hello World!')
        'Last index of the array.\n        '
        ...

    def last_value(self) -> Any:
        if False:
            i = 10
            return i + 15
        'Last value of the array.\n        '
        ...