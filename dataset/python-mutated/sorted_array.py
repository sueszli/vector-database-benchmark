import numpy as np

def _searchsorted(array, val, side='left'):
    if False:
        i = 10
        return i + 15
    '\n    Call np.searchsorted or use a custom binary\n    search if necessary.\n    '
    if hasattr(array, 'searchsorted'):
        return array.searchsorted(val, side=side)
    begin = 0
    end = len(array)
    while begin < end:
        mid = (begin + end) // 2
        if val > array[mid]:
            begin = mid + 1
        elif val < array[mid]:
            end = mid
        elif side == 'right':
            begin = mid + 1
        else:
            end = mid
    return begin

class SortedArray:
    """
    Implements a sorted array container using
    a list of numpy arrays.

    Parameters
    ----------
    data : Table
        Sorted columns of the original table
    row_index : Column object
        Row numbers corresponding to data columns
    unique : bool
        Whether the values of the index must be unique.
        Defaults to False.
    """

    def __init__(self, data, row_index, unique=False):
        if False:
            i = 10
            return i + 15
        self.data = data
        self.row_index = row_index
        self.num_cols = len(getattr(data, 'colnames', []))
        self.unique = unique

    @property
    def cols(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.data.columns.values())

    def add(self, key, row):
        if False:
            print('Hello World!')
        '\n        Add a new entry to the sorted array.\n\n        Parameters\n        ----------\n        key : tuple\n            Column values at the given row\n        row : int\n            Row number\n        '
        pos = self.find_pos(key, row)
        if self.unique and 0 <= pos < len(self.row_index) and all((self.data[pos][i] == key[i] for i in range(len(key)))):
            raise ValueError(f'Cannot add duplicate value "{key}" in a unique index')
        self.data.insert_row(pos, key)
        self.row_index = self.row_index.insert(pos, row)

    def _get_key_slice(self, i, begin, end):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve the ith slice of the sorted array\n        from begin to end.\n        '
        if i < self.num_cols:
            return self.cols[i][begin:end]
        else:
            return self.row_index[begin:end]

    def find_pos(self, key, data, exact=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the index of the largest key in data greater than or\n        equal to the given key, data pair.\n\n        Parameters\n        ----------\n        key : tuple\n            Column key\n        data : int\n            Row number\n        exact : bool\n            If True, return the index of the given key in data\n            or -1 if the key is not present.\n        '
        begin = 0
        end = len(self.row_index)
        num_cols = self.num_cols
        if not self.unique:
            key = key + (data,)
            num_cols += 1
        for i in range(num_cols):
            key_slice = self._get_key_slice(i, begin, end)
            t = _searchsorted(key_slice, key[i])
            if exact and (t == len(key_slice) or key_slice[t] != key[i]):
                return -1
            elif t == len(key_slice) or (t == 0 and len(key_slice) > 0 and (key[i] < key_slice[0])):
                return begin + t
            end = begin + _searchsorted(key_slice, key[i], side='right')
            begin += t
            if begin >= len(self.row_index):
                return begin
        return begin

    def find(self, key):
        if False:
            while True:
                i = 10
        '\n        Find all rows matching the given key.\n\n        Parameters\n        ----------\n        key : tuple\n            Column values\n\n        Returns\n        -------\n        matching_rows : list\n            List of rows matching the input key\n        '
        begin = 0
        end = len(self.row_index)
        for i in range(self.num_cols):
            key_slice = self._get_key_slice(i, begin, end)
            t = _searchsorted(key_slice, key[i])
            if t == len(key_slice) or key_slice[t] != key[i]:
                return []
            elif t == 0 and len(key_slice) > 0 and (key[i] < key_slice[0]):
                return []
            end = begin + _searchsorted(key_slice, key[i], side='right')
            begin += t
            if begin >= len(self.row_index):
                return []
        return self.row_index[begin:end]

    def range(self, lower, upper, bounds):
        if False:
            return 10
        '\n        Find values in the given range.\n\n        Parameters\n        ----------\n        lower : tuple\n            Lower search bound\n        upper : tuple\n            Upper search bound\n        bounds : (2,) tuple of bool\n            Indicates whether the search should be inclusive or\n            exclusive with respect to the endpoints. The first\n            argument corresponds to an inclusive lower bound,\n            and the second argument to an inclusive upper bound.\n        '
        lower_pos = self.find_pos(lower, 0)
        upper_pos = self.find_pos(upper, 0)
        if lower_pos == len(self.row_index):
            return []
        lower_bound = tuple((col[lower_pos] for col in self.cols))
        if not bounds[0] and lower_bound == lower:
            lower_pos += 1
        if upper_pos < len(self.row_index):
            upper_bound = tuple((col[upper_pos] for col in self.cols))
            if not bounds[1] and upper_bound == upper:
                upper_pos -= 1
            elif upper_bound > upper:
                upper_pos -= 1
        return self.row_index[lower_pos:upper_pos + 1]

    def remove(self, key, data):
        if False:
            while True:
                i = 10
        '\n        Remove the given entry from the sorted array.\n\n        Parameters\n        ----------\n        key : tuple\n            Column values\n        data : int\n            Row number\n\n        Returns\n        -------\n        successful : bool\n            Whether the entry was successfully removed\n        '
        pos = self.find_pos(key, data, exact=True)
        if pos == -1:
            return False
        self.data.remove_row(pos)
        keep_mask = np.ones(len(self.row_index), dtype=bool)
        keep_mask[pos] = False
        self.row_index = self.row_index[keep_mask]
        return True

    def shift_left(self, row):
        if False:
            for i in range(10):
                print('nop')
        '\n        Decrement all row numbers greater than the input row.\n\n        Parameters\n        ----------\n        row : int\n            Input row number\n        '
        self.row_index[self.row_index > row] -= 1

    def shift_right(self, row):
        if False:
            for i in range(10):
                print('nop')
        '\n        Increment all row numbers greater than or equal to the input row.\n\n        Parameters\n        ----------\n        row : int\n            Input row number\n        '
        self.row_index[self.row_index >= row] += 1

    def replace_rows(self, row_map):
        if False:
            for i in range(10):
                print('nop')
        '\n        Replace all rows with the values they map to in the\n        given dictionary. Any rows not present as keys in\n        the dictionary will have their entries deleted.\n\n        Parameters\n        ----------\n        row_map : dict\n            Mapping of row numbers to new row numbers\n        '
        num_rows = len(row_map)
        keep_rows = np.zeros(len(self.row_index), dtype=bool)
        tagged = 0
        for (i, row) in enumerate(self.row_index):
            if row in row_map:
                keep_rows[i] = True
                tagged += 1
                if tagged == num_rows:
                    break
        self.data = self.data[keep_rows]
        self.row_index = np.array([row_map[x] for x in self.row_index[keep_rows]])

    def items(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve all array items as a list of pairs of the form\n        [(key, [row 1, row 2, ...]), ...].\n        '
        array = []
        last_key = None
        for (i, key) in enumerate(zip(*self.data.columns.values())):
            row = self.row_index[i]
            if key == last_key:
                array[-1][1].append(row)
            else:
                last_key = key
                array.append((key, [row]))
        return array

    def sort(self):
        if False:
            return 10
        '\n        Make row order align with key order.\n        '
        self.row_index = np.arange(len(self.row_index))

    def sorted_data(self):
        if False:
            print('Hello World!')
        '\n        Return rows in sorted order.\n        '
        return self.row_index

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        '\n        Return a sliced reference to this sorted array.\n\n        Parameters\n        ----------\n        item : slice\n            Slice to use for referencing\n        '
        return SortedArray(self.data[item], self.row_index[item])

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        t = self.data.copy()
        t['rows'] = self.row_index
        return f'<{self.__class__.__name__} length={len(t)}>\n{t}'