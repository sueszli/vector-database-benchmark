import numpy as np
from typing import Any, Iterator, List, Tuple, Union
from aim.storage.treeview import TreeView
from aim.storage.arrayview import ArrayView

class TreeArrayView(ArrayView):

    def __init__(self, tree: 'TreeView', dtype: Any=None):
        if False:
            print('Hello World!')
        self.tree = tree
        self.dtype = dtype

    def allocate(self):
        if False:
            while True:
                i = 10
        self.tree.make_array()
        return self

    def __iter__(self) -> Iterator[Any]:
        if False:
            for i in range(10):
                print('nop')
        yield from self.values()

    def keys(self) -> Iterator[int]:
        if False:
            return 10
        yield from self.tree.keys()

    def indices(self) -> Iterator[int]:
        if False:
            while True:
                i = 10
        yield from self.keys()

    def values(self) -> Iterator[Any]:
        if False:
            i = 10
            return i + 15
        for (k, v) in self.tree.items():
            yield v

    def items(self) -> Iterator[Tuple[int, Any]]:
        if False:
            i = 10
            return i + 15
        yield from self.tree.items()

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        try:
            last_idx = self.last_idx()
        except KeyError:
            return 0
        return last_idx + 1

    def __bool__(self) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(len(self))

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if False:
            return 10
        if isinstance(idx, slice):
            raise NotImplementedError
        return self.tree[idx]

    def __setitem__(self, idx: int, val: Any):
        if False:
            return 10
        assert isinstance(idx, int)
        self.tree[idx] = val

    def sparse_list(self) -> Tuple[List[int], List[Any]]:
        if False:
            print('Hello World!')
        indices = []
        values = []
        for (k, v) in self.items():
            indices.append(k)
            values.append(v)
        return (indices, values)

    def indices_list(self) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        return list(self.indices())

    def values_list(self) -> List[Any]:
        if False:
            print('Hello World!')
        return list(self.values())

    def sparse_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            while True:
                i = 10
        (indices_list, values_list) = self.sparse_list()
        indices_array = np.array(indices_list, dtype=np.intp)
        values_array = np.array(values_list, dtype=self.dtype)
        return (indices_array, values_array)

    def indices_numpy(self) -> np.ndarray:
        if False:
            print('Hello World!')
        return np.array(self.indices_list(), dtype=np.intp)

    def values_numpy(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        return np.array(self.values_list(), dtype=self.dtype)

    def tolist(self) -> List[Any]:
        if False:
            i = 10
            return i + 15
        arr = self.tree[...]
        assert isinstance(arr, list)
        return arr

    def first(self) -> Tuple[int, Any]:
        if False:
            while True:
                i = 10
        idx = self.first_idx()
        return (idx, self[idx])

    def first_idx(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.tree.first_key()

    def first_value(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return self[self.first_idx()]

    def last(self) -> Tuple[int, Any]:
        if False:
            i = 10
            return i + 15
        idx = self.last_idx()
        return (idx, self[idx])

    def last_idx(self) -> int:
        if False:
            while True:
                i = 10
        return self.tree.last_key()

    def last_value(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return self[self.last_idx()]