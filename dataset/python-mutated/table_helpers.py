"""
Helper functions for table development, mostly creating useful
tables for testing.
"""
import string
from itertools import cycle
import numpy as np
from astropy.utils.data_info import ParentDtypeInfo
from .table import Column, Table

class TimingTables:
    """
    Object which contains two tables and various other attributes that
    are useful for timing and other API tests.
    """

    def __init__(self, size=1000, masked=False):
        if False:
            i = 10
            return i + 15
        self.masked = masked
        self.table = Table(masked=self.masked)
        np.random.seed(12345)
        self.table['i'] = np.arange(size)
        self.table['a'] = np.random.random(size)
        self.table['b'] = np.random.random(size) > 0.5
        self.table['c'] = np.random.random((size, 10))
        self.table['d'] = np.random.choice(np.array(list(string.ascii_letters)), size)
        self.extra_row = {'a': 1.2, 'b': True, 'c': np.repeat(1, 10), 'd': 'Z'}
        self.extra_column = np.random.randint(0, 100, size)
        self.row_indices = np.where(self.table['a'] > 0.9)[0]
        self.table_grouped = self.table.group_by('d')
        self.other_table = Table(masked=self.masked)
        self.other_table['i'] = np.arange(1, size, 3)
        self.other_table['f'] = np.random.random()
        self.other_table.sort('f')
        self.other_table_2 = Table(masked=self.masked)
        self.other_table_2['g'] = np.random.random(size)
        self.other_table_2['h'] = np.random.random((size, 10))
        self.bool_mask = self.table['a'] > 0.6

def simple_table(size=3, cols=None, kinds='ifS', masked=False):
    if False:
        while True:
            i = 10
    "\n    Return a simple table for testing.\n\n    Example\n    --------\n    ::\n\n      >>> from astropy.table.table_helpers import simple_table\n      >>> print(simple_table(3, 6, masked=True, kinds='ifOS'))\n       a   b     c      d   e   f\n      --- --- -------- --- --- ---\n       -- 1.0 {'c': 2}  --   5 5.0\n        2 2.0       --   e   6  --\n        3  -- {'e': 4}   f  -- 7.0\n\n    Parameters\n    ----------\n    size : int\n        Number of table rows\n    cols : int, optional\n        Number of table columns. Defaults to number of kinds.\n    kinds : str\n        String consisting of the column dtype.kinds.  This string\n        will be cycled through to generate the column dtype.\n        The allowed values are 'i', 'f', 'S', 'O'.\n\n    Returns\n    -------\n    out : `Table`\n        New table with appropriate characteristics\n    "
    if cols is None:
        cols = len(kinds)
    if cols > 26:
        raise ValueError('Max 26 columns in SimpleTable')
    columns = []
    names = [chr(ord('a') + ii) for ii in range(cols)]
    letters = np.array(list(string.ascii_letters))
    for (jj, kind) in zip(range(cols), cycle(kinds)):
        if kind == 'i':
            data = np.arange(1, size + 1, dtype=np.int64) + jj
        elif kind == 'f':
            data = np.arange(size, dtype=np.float64) + jj
        elif kind == 'S':
            indices = (np.arange(size) + jj) % len(letters)
            data = letters[indices]
        elif kind == 'O':
            indices = (np.arange(size) + jj) % len(letters)
            vals = letters[indices]
            data = [{val: index} for (val, index) in zip(vals, indices)]
        else:
            raise ValueError('Unknown data kind')
        columns.append(Column(data))
    table = Table(columns, names=names, masked=masked)
    if masked:
        for (ii, col) in enumerate(table.columns.values()):
            mask = np.array((np.arange(size) + ii) % 3, dtype=bool)
            col.mask = ~mask
    return table

def complex_table():
    if False:
        print('Hello World!')
    '\n    Return a masked table from the io.votable test set that has a wide variety\n    of stressing types.\n    '
    import warnings
    from astropy.io.votable.table import parse
    from astropy.utils.data import get_pkg_data_filename
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        votable = parse(get_pkg_data_filename('../io/votable/tests/data/regression.xml'), pedantic=False)
    first_table = votable.get_first_table()
    table = first_table.to_table()
    return table

class ArrayWrapperInfo(ParentDtypeInfo):
    _represent_as_dict_primary_data = 'data'

    def _represent_as_dict(self):
        if False:
            print('Hello World!')
        'Represent Column as a dict that can be serialized.'
        col = self._parent
        out = {'data': col.data}
        return out

    def _construct_from_dict(self, map):
        if False:
            while True:
                i = 10
        'Construct Column from ``map``.'
        data = map.pop('data')
        out = self._parent_cls(data, **map)
        return out

class ArrayWrapper:
    """
    Minimal mixin using a simple wrapper around a numpy array.

    TODO: think about the future of this class as it is mostly for demonstration
    purposes (of the mixin protocol). Consider taking it out of core and putting
    it into a tutorial. One advantage of having this in core is that it is
    getting tested in the mixin testing though it doesn't work for multidim
    data.
    """
    info = ArrayWrapperInfo()

    def __init__(self, data, copy=True):
        if False:
            while True:
                i = 10
        self.data = np.array(data, copy=copy)
        if 'info' in getattr(data, '__dict__', ()):
            self.info = data.info

    def __getitem__(self, item):
        if False:
            return 10
        if isinstance(item, (int, np.integer)):
            out = self.data[item]
        else:
            out = self.__class__(self.data[item], copy=False)
            if 'info' in self.__dict__:
                out.info = self.info
        return out

    def __setitem__(self, item, value):
        if False:
            print('Hello World!')
        self.data[item] = value

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.data)

    def __eq__(self, other):
        if False:
            return 10
        'Minimal equality testing, mostly for mixin unit tests.'
        if isinstance(other, ArrayWrapper):
            return self.data == other.data
        else:
            return self.data == other

    @property
    def dtype(self):
        if False:
            while True:
                i = 10
        return self.data.dtype

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        return self.data.shape

    def __repr__(self):
        if False:
            return 10
        return f"<{self.__class__.__name__} name='{self.info.name}' data={self.data}>"