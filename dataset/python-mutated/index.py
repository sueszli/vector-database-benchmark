from typing import Union, List, Tuple, Iterable, Optional
from collections.abc import Iterable
import numpy as np
IndexValue = Union[int, slice, Tuple[int, ...]]

def has_negatives(s: slice) -> bool:
    if False:
        while True:
            i = 10
    if s.start and s.start < 0:
        return True
    elif s.stop and s.stop < 0:
        return True
    elif s.step and s.step < 0:
        return True
    else:
        return False

def merge_slices(existing_slice: slice, new_slice: slice) -> slice:
    if False:
        for i in range(10):
            print('nop')
    'Compose two slice objects\n\n    Given an iterable x, the following should be equivalent:\n\n    ``x[existing_slice][new_slice] == x[merge_slices(existing_slice, new_slice)]``\n\n    Args:\n        existing_slice (slice): The existing slice to be restricted.\n        new_slice (slice): The new slice to be applied to the existing slice.\n\n    Returns:\n        slice: the composition of the given slices\n\n    Raises:\n        NotImplementedError: Composing slices with negative values is not supported.\n            Negative indexing for slices is only supported for the first slice.\n    '
    if existing_slice == slice(None):
        return new_slice
    elif new_slice == slice(None):
        return existing_slice
    if has_negatives(existing_slice) or has_negatives(new_slice):
        raise NotImplementedError('Multiple subscripting for slices with negative values is not supported.')
    step1 = existing_slice.step if existing_slice.step is not None else 1
    step2 = new_slice.step if new_slice.step is not None else 1
    step = step1 * step2
    start1 = existing_slice.start if existing_slice.start is not None else 0
    start2 = new_slice.start if new_slice.start is not None else 0
    start = start1 + start2 * step1
    stop1 = existing_slice.stop
    stop2 = new_slice.stop
    if stop2 is None:
        stop = stop1
    else:
        stop = start + (stop2 - start2) * step1
        if stop1 is not None:
            stop = min(stop, stop1)
    return slice(start, stop, step)

def slice_at_int(s: slice, i: int):
    if False:
        i = 10
        return i + 15
    'Returns the ``i`` th element of a slice ``s``.\n\n    Examples:\n        >>> slice_at_int(slice(None), 10)\n        10\n\n        >>> slice_at_int(slice(10, 20, 2), 3)\n        16\n\n    Args:\n        s (slice): The slice to index into.\n        i (int): The integer offset into the slice.\n\n    Returns:\n        int: The index corresponding to the offset into the slice.\n\n    Raises:\n        NotImplementedError: Nontrivial slices should not be indexed with negative integers.\n        IndexError: If step is negative and start is not greater than stop.\n    '
    if s == slice(None):
        return i
    if i < 0:
        raise NotImplementedError('Subscripting slices with negative integers is not supported.')
    step = s.step if s.step is not None else 1
    if step < 0:
        if (s.start and s.stop) and s.stop > s.start:
            raise IndexError(f'index {i} out of bounds.')
    start = s.start
    if start is None:
        start = -1 if step < 0 else 0
    return start + i * step

def slice_length(s: slice, parent_length: int) -> int:
    if False:
        i = 10
        return i + 15
    'Returns the length of a slice given the length of its parent.'
    (start, stop, step) = s.indices(parent_length)
    step_offset = step - (1 if step > 0 else -1)
    slice_length = stop - start
    total_length = (slice_length + step_offset) // step
    return max(0, total_length)

def replace_ellipsis_with_slices(items, ndim: int):
    if False:
        return 10
    if items is Ellipsis:
        return (slice(None),) * ndim
    try:
        idx = items.index(Ellipsis)
    except ValueError:
        return items
    nslices = ndim - len(items) + 1
    if Ellipsis in items[idx + 1:]:
        raise IndexError("an index can only have a single ellipsis ('...')")
    items = items[:idx] + (slice(None),) * nslices + items[idx + 1:]
    return items

class IndexEntry:

    def __init__(self, value: IndexValue=slice(None)):
        if False:
            for i in range(10):
                print('nop')
        self.value = value

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'IndexEntry({self.value})'

    def __getitem__(self, item: IndexValue):
        if False:
            return 10
        'Combines the given ``item`` and this IndexEntry.\n        Returns a new IndexEntry representing the composition of the two.\n\n        Examples:\n            >>> IndexEntry()[0:100]\n            IndexEntry(slice(0, 100, None))\n\n            >>> IndexEntry()[100:200][5]\n            IndexEntry(105)\n\n            >>> IndexEntry()[(0, 1, 2, 3)]\n            IndexEntry((0, 1, 2, 3))\n\n            >>> IndexEntry()[1, 2, 3]\n            IndexEntry((0, 1, 2, 3))\n\n        Args:\n            item: The desired sub-index to be composed with this IndexEntry.\n                Can be an int, a slice, or a tuple of ints.\n\n        Returns:\n            IndexEntry: The new IndexEntry object.\n\n        Raises:\n            TypeError: An integer IndexEntry should not be indexed further.\n        '
        if not self.subscriptable():
            raise TypeError("Subscripting IndexEntry after 'int' is not allowed. Use Index instead.")
        elif isinstance(self.value, slice):
            if isinstance(item, int):
                new_value = slice_at_int(self.value, item)
                return IndexEntry(new_value)
            elif isinstance(item, slice):
                return IndexEntry(merge_slices(self.value, item))
            elif isinstance(item, (tuple, list)):
                if self.is_trivial():
                    new_value = tuple(item)
                else:
                    new_value = tuple((slice_at_int(self.value, idx) for idx in item))
                return IndexEntry(new_value)
        elif isinstance(self.value, (tuple, list)):
            if isinstance(item, int) or isinstance(item, slice):
                return IndexEntry(self.value[item])
            elif isinstance(item, (tuple, list)):
                new_value = tuple((self.value[idx] for idx in item))
                return IndexEntry(new_value)
        raise TypeError(f'Value {item} is of unrecognized type {type(item)}.')

    def subscriptable(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether an IndexEntry can be further subscripted.'
        return not isinstance(self.value, int)

    def indices(self, length: int):
        if False:
            print('Hello World!')
        'Generates the sequence of integer indices for a target of a given length.'
        parse_int = lambda i: i if i >= 0 else length + i
        if isinstance(self.value, int):
            yield parse_int(self.value)
        elif isinstance(self.value, slice):
            yield from range(*self.value.indices(length))
        elif isinstance(self.value, Iterable):
            yield from map(parse_int, self.value)
        elif callable(self.value):
            yield from self.value()

    def is_trivial(self):
        if False:
            i = 10
            return i + 15
        'Checks if an IndexEntry represents the entire slice'
        return isinstance(self.value, slice) and (not self.value.start) and (self.value.stop is None) and ((self.value.step or 1) == 1)

    def length(self, parent_length: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns the length of an IndexEntry given the length of the parent it is indexing.\n\n        Examples:\n            >>> IndexEntry(slice(5, 10)).length(100)\n            5\n            >>> len(list(range(100))[5:10])\n            5\n            >>> IndexEntry(slice(5, 100)).length(50)\n            45\n            >>> len(list(range(50))[5:100])\n            45\n            >>> IndexEntry(0).length(10)\n            1\n\n        Args:\n            parent_length (int): The length of the target that this IndexEntry is indexing.\n\n        Returns:\n            int: The length of the index if it were applied to a parent of the given length.\n        '
        if parent_length == 0:
            return 0
        elif not self.subscriptable():
            return 1
        elif isinstance(self.value, slice):
            return slice_length(self.value, parent_length)
        lenf = getattr(self.value, '__len__', None)
        if lenf is None:
            return 0
        return lenf()

    def validate(self, parent_length: int):
        if False:
            i = 10
            return i + 15
        'Checks that the index is not accessing values outside the range of the parent.'
        if isinstance(self.value, slice):
            return
        value_to_check = self.value
        if isinstance(value_to_check, int):
            value_to_check = (value_to_check,)
        if isinstance(value_to_check, tuple):
            value_arr = np.array(value_to_check)
            if np.any((value_arr >= parent_length) | (value_arr < -parent_length)):
                raise IndexError(f'Index {value_to_check} is out of range for tensors with length {parent_length}')

    def downsample(self, factor: int, length: int):
        if False:
            for i in range(10):
                print('nop')
        'Downsamples an IndexEntry by a given factor.\n\n        Args:\n            factor (int): The factor by which to downsample.\n            length (int): The length of the downsampled IndexEntry.\n\n        Returns:\n            IndexEntry: The downsampled IndexEntry.\n\n        Raises:\n            TypeError: If the IndexEntry cannot be downsampled.\n        '
        if isinstance(self.value, slice):
            start = self.value.start or 0
            stop = self.value.stop
            step = self.value.step or 1
            assert step == 1, 'Cannot downsample with step != 1'
            downsampled_start = start // factor
            downsampled_stop = stop // factor if stop is not None else None
            if downsampled_stop is None or downsampled_stop - downsampled_start != length:
                downsampled_stop = downsampled_start + length
            return IndexEntry(slice(downsampled_start, downsampled_stop, 1))
        else:
            raise TypeError(f'Cannot downsample IndexEntry with value {self.value} of type {type(self.value)}')

class Index:

    def __init__(self, item: Union[IndexValue, 'Index', List[IndexEntry]]=slice(None)):
        if False:
            print('Hello World!')
        'Initializes an Index from an IndexValue, another Index, or the values from another Index.\n\n        Represents a list of IndexEntry objects corresponding to indexes into each axis of an ndarray.\n        '
        if isinstance(item, Index):
            item = item.values
        elif item in ((), [], None):
            item = slice(None)
        if isinstance(item, tuple):
            item = list(map(IndexEntry, item))
        if not (isinstance(item, list) and isinstance(item[0], IndexEntry)):
            item = [IndexEntry(item)]
        self.values: List[IndexEntry] = item

    def find_axis(self, offset: int=0):
        if False:
            print('Hello World!')
        'Returns the index for the nth subscriptable axis in the values of an Index.\n\n        Args:\n            offset (int): The number of subscriptable axes to skip before returning.\n                Defaults to 0, meaning that the first valid axis is returned.\n\n        Returns:\n            int: The index of the found axis, or None if no match is found.\n        '
        matches = 0
        for (idx, entry) in enumerate(self.values):
            if entry.subscriptable():
                if matches == offset:
                    return idx
                else:
                    matches += 1
        return None

    def compose_at(self, item: IndexValue, i: Optional[int]=None):
        if False:
            return 10
        'Returns a new Index representing the addition of an IndexValue,\n        or the composition with a given axis.\n\n        Examples:\n            >>> Index([slice(None), slice(None)]).compose_at(5)\n            Index([slice(None), slice(None), 5])\n\n            >>> Index([slice(None), slice(5, 10), slice(None)]).compose_at(3, 1)\n            Index([slice(None), 8, slice(None)])\n\n        Args:\n            item (IndexValue): The value to append or compose with the Index.\n            i (int, optional): The axis to compose with the given item.\n                Defaults to None, meaning that the item will be appended instead.\n\n        Returns:\n            Index: The result of the addition or composition.\n        '
        if i is None or i >= len(self.values):
            return Index(self.values + [IndexEntry(item)])
        else:
            new_values = self.values[:i] + [self.values[i][item]] + self.values[i + 1:]
            return Index(new_values)

    def __getitem__(self, item: Union[int, slice, List[int], Tuple[IndexValue], 'Index']):
        if False:
            print('Hello World!')
        "Returns a new Index representing a subscripting with the given item.\n        Modeled after NumPy's advanced integer indexing.\n\n        See: https://numpy.org/doc/stable/reference/arrays.indexing.html\n\n        Examples:\n            >>> Index([5, slice(None)])[5]\n            Index([5, 5])\n\n            >>> Index([5])[5:6]\n            Index([5, slice(5, 6)])\n\n            >>> Index()[0, 1, 2:5, 3]\n            Index([0, 1, slice(2, 5), 3])\n\n            >>> Index([slice(5, 6)])[(0, 1, 2:5, 3),]\n            Index([(5, 1, slice(2, 5), 3)])\n\n        Args:\n            item: The contents of the subscript expression to add to this Index.\n\n        Returns:\n            Index: The Index representing the result of the subscript operation.\n\n        Raises:\n            TypeError: Given item should be another Index,\n                or compatible with NumPy's advanced integer indexing.\n        "
        if isinstance(item, int) or isinstance(item, slice):
            ax = self.find_axis()
            return self.compose_at(item, ax)
        elif isinstance(item, tuple):
            new_index = self
            for (idx, sub_item) in enumerate(item):
                ax = new_index.find_axis(offset=idx)
                new_index = new_index.compose_at(sub_item, ax)
            return new_index
        elif isinstance(item, list):
            return self[tuple(item),]
        elif isinstance(item, Index):
            return self[tuple((v.value for v in item.values))]
        else:
            raise TypeError(f'Value {item} is of unrecognized type {type(item)}.')

    def apply(self, samples: List[np.ndarray]):
        if False:
            for i in range(10):
                print('nop')
        'Applies an Index to a list of ndarray samples with the same number of entries\n        as the first entry in the Index.\n        '
        index_values = tuple((item.value for item in self.values[1:]))
        if index_values:
            samples = [arr[index_values] for arr in samples]
        else:
            samples = list(samples)
        return samples

    def apply_squeeze(self, samples: List[np.ndarray]):
        if False:
            while True:
                i = 10
        'Applies the primary axis of an Index to a list of ndarray samples.\n        Will either return the list as given, or return the first sample.\n        '
        if self.values[0].subscriptable():
            return samples
        else:
            return samples[0]

    def is_trivial(self):
        if False:
            i = 10
            return i + 15
        'Checks if an Index is equivalent to the trivial slice `[:]`, aka slice(None).'
        return len(self.values) == 1 and self.values[0].is_trivial()

    def length(self, parent_length: int):
        if False:
            print('Hello World!')
        'Returns the primary length of an Index given the length of the parent it is indexing.\n        See: :meth:`IndexEntry.length`'
        return self.values[0].length(parent_length)

    def validate(self, parent_length):
        if False:
            print('Hello World!')
        'Checks that the index is not accessing values outside the range of the parent.'
        self.values[0].validate(parent_length)

    def __str__(self):
        if False:
            print('Hello World!')
        eval_f = lambda v: list(v()) if callable(v) else v
        values = [eval_f(entry.value) for entry in self.values]
        return f'Index({values})'

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'Index(values={self.values})'

    def to_json(self):
        if False:
            while True:
                i = 10
        ret = []
        for e in self.values:
            v = e.value
            if isinstance(v, slice):
                ret.append({'start': v.start, 'stop': v.stop, 'step': v.step})
            elif isinstance(v, Iterable):
                ret.append(list(v))
            elif callable(v):
                ret.append(list(v()))
            else:
                ret.append(v)
        return ret

    @classmethod
    def from_json(cls, idxs):
        if False:
            print('Hello World!')
        entries = []
        for idx in idxs:
            if isinstance(idx, dict):
                idx = slice(idx['start'], idx['stop'], idx['step'])
            entries.append(IndexEntry(idx))
        return cls(entries)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.values)

    def subscriptable_at(self, i: int) -> bool:
        if False:
            print('Hello World!')
        try:
            return self.values[i].subscriptable()
        except IndexError:
            return True

    def length_at(self, i: int, parent_length: int) -> int:
        if False:
            while True:
                i = 10
        try:
            return self.values[i].length(parent_length)
        except IndexError:
            return parent_length

    def trivial_at(self, i: int) -> bool:
        if False:
            while True:
                i = 10
        try:
            return self.values[i].is_trivial()
        except IndexError:
            return True

    def downsample(self, factor: int, shape: Tuple[int, ...]):
        if False:
            while True:
                i = 10
        'Downsamples an Index by the given factor.\n\n        Args:\n            factor (int): The factor to downsample by.\n            shape (Tuple[int, ...]): The shape of the downsampled data.\n\n        Returns:\n            Index: The downsampled Index.\n        '
        new_values = [v.downsample(factor, length) for (v, length) in zip(self.values[:2], shape)]
        new_values += self.values[2:]
        return Index(new_values)