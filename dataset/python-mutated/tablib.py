from collections import OrderedDict
from .format import registry

class Row:
    """Internal Row object. Mainly used for filtering."""
    __slots__ = ['_row', 'tags']

    def __init__(self, row=None, tags=None):
        if False:
            while True:
                i = 10
        if tags is None:
            tags = list()
        if row is None:
            row = list()
        self._row = list(row)
        self.tags = list(tags)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return (col for col in self._row)

    def __len__(self):
        if False:
            return 10
        return len(self._row)

    def __repr__(self):
        if False:
            return 10
        return repr(self._row)

    def __getitem__(self, i):
        if False:
            print('Hello World!')
        return self._row[i]

    def __setitem__(self, i, value):
        if False:
            for i in range(10):
                print('nop')
        self._row[i] = value

    def __delitem__(self, i):
        if False:
            i = 10
            return i + 15
        del self._row[i]

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        slots = dict()
        for slot in self.__slots__:
            attribute = getattr(self, slot)
            slots[slot] = attribute
        return slots

    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        for (k, v) in list(state.items()):
            setattr(self, k, v)

    def rpush(self, value):
        if False:
            while True:
                i = 10
        self.insert(len(self._row), value)

    def append(self, value):
        if False:
            return 10
        self.rpush(value)

    def insert(self, index, value):
        if False:
            return 10
        self._row.insert(index, value)

    def __contains__(self, item):
        if False:
            while True:
                i = 10
        return item in self._row

    @property
    def tuple(self):
        if False:
            i = 10
            return i + 15
        'Tuple representation of :class:`Row`.'
        return tuple(self._row)

class Dataset:
    """The :class:`Dataset` object is the heart of Tablib. It provides all core
    functionality.

    Usually you create a :class:`Dataset` instance in your main module, and append
    rows as you collect data. ::

        data = tablib.Dataset()
        data.headers = ('name', 'age')

        for (name, age) in some_collector():
            data.append((name, age))


    Setting columns is similar. The column data length must equal the
    current height of the data and headers must be set. ::

        data = tablib.Dataset()
        data.headers = ('first_name', 'last_name')

        data.append(('John', 'Adams'))
        data.append(('George', 'Washington'))

        data.append_col((90, 67), header='age')


    You can also set rows and headers upon instantiation. This is useful if
    dealing with dozens or hundreds of :class:`Dataset` objects. ::

        headers = ('first_name', 'last_name')
        data = [('John', 'Adams'), ('George', 'Washington')]

        data = tablib.Dataset(*data, headers=headers)

    :param \\*args: (optional) list of rows to populate Dataset
    :param headers: (optional) list strings for Dataset header row
    :param title: (optional) string to use as title of the Dataset


    .. admonition:: Format Attributes Definition

     If you look at the code, the various output/import formats are not
     defined within the :class:`Dataset` object. To add support for a new format, see
     :ref:`Adding New Formats <newformats>`.

    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._data = list((Row(arg) for arg in args))
        self.__headers = None
        self._separators = []
        self._formatters = []
        self.headers = kwargs.get('headers')
        self.title = kwargs.get('title')

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.height

    def _validate(self, row=None, col=None, safety=False):
        if False:
            for i in range(10):
                print('nop')
        'Assures size of every row in dataset is of proper proportions.'
        if row:
            is_valid = len(row) == self.width if self.width else True
        elif col:
            if len(col) < 1:
                is_valid = True
            else:
                is_valid = len(col) == self.height if self.height else True
        else:
            is_valid = all((len(x) == self.width for x in self._data))
        if is_valid:
            return True
        if not safety:
            raise InvalidDimensions
        return False

    def _package(self, dicts=True, ordered=True):
        if False:
            while True:
                i = 10
        'Packages Dataset into lists of dictionaries for transmission.'
        _data = list(self._data)
        if ordered:
            dict_pack = OrderedDict
        else:
            dict_pack = dict
        if self._formatters:
            for (row_i, row) in enumerate(_data):
                for (col, callback) in self._formatters:
                    try:
                        if col is None:
                            for (j, c) in enumerate(row):
                                _data[row_i][j] = callback(c)
                        else:
                            _data[row_i][col] = callback(row[col])
                    except IndexError:
                        raise InvalidDatasetIndex
        if self.headers:
            if dicts:
                data = [dict_pack(list(zip(self.headers, data_row))) for data_row in _data]
            else:
                data = [list(self.headers)] + list(_data)
        else:
            data = [list(row) for row in _data]
        return data

    def _get_headers(self):
        if False:
            while True:
                i = 10
        'An *optional* list of strings to be used for header rows and attribute names.\n\n        This must be set manually. The given list length must equal :class:`Dataset.width`.\n\n        '
        return self.__headers

    def _set_headers(self, collection):
        if False:
            print('Hello World!')
        'Validating headers setter.'
        self._validate(collection)
        if collection:
            try:
                self.__headers = list(collection)
            except TypeError:
                raise TypeError
        else:
            self.__headers = None
    headers = property(_get_headers, _set_headers)

    def _get_dict(self):
        if False:
            return 10
        "A native Python representation of the :class:`Dataset` object. If headers have\n        been set, a list of Python dictionaries will be returned. If no headers have been\n        set, a list of tuples (rows) will be returned instead.\n\n        A dataset object can also be imported by setting the `Dataset.dict` attribute: ::\n\n            data = tablib.Dataset()\n            data.dict = [{'age': 90, 'first_name': 'Kenneth', 'last_name': 'Reitz'}]\n\n        "
        return self._package()

    def _set_dict(self, pickle):
        if False:
            print('Hello World!')
        "A native Python representation of the Dataset object. If headers have been\n        set, a list of Python dictionaries will be returned. If no headers have been\n        set, a list of tuples (rows) will be returned instead.\n\n        A dataset object can also be imported by setting the :class:`Dataset.dict` attribute. ::\n\n            data = tablib.Dataset()\n            data.dict = [{'age': 90, 'first_name': 'Kenneth', 'last_name': 'Reitz'}]\n\n        "
        if not len(pickle):
            return
        if isinstance(pickle[0], list):
            self.wipe()
            for row in pickle:
                self.append(Row(row))
        elif isinstance(pickle[0], dict):
            self.wipe()
            self.headers = list(pickle[0].keys())
            for row in pickle:
                self.append(Row(list(row.values())))
        else:
            raise UnsupportedFormat
    dict = property(_get_dict, _set_dict)

    @property
    def height(self):
        if False:
            return 10
        'The number of rows currently in the :class:`Dataset`.\n           Cannot be directly modified.\n        '
        return len(self._data)

    @property
    def width(self):
        if False:
            i = 10
            return i + 15
        'The number of columns currently in the :class:`Dataset`.\n           Cannot be directly modified.\n        '
        try:
            return len(self._data[0])
        except IndexError:
            try:
                return len(self.headers)
            except TypeError:
                return 0

    def export(self, format, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Export :class:`Dataset` object to `format`.\n\n        :param format: export format\n        :param kwargs: (optional) custom configuration to the format `export_set`.\n        '
        fmt = registry.get_format(format)
        if not hasattr(fmt, 'export_set'):
            raise Exception('Format {} cannot be exported.'.format(format))
        return fmt.export_set(self, **kwargs)

    def insert(self, index, row, tags=None):
        if False:
            i = 10
            return i + 15
        'Inserts a row to the :class:`Dataset` at the given index.\n\n        Rows inserted must be the correct size (height or width).\n\n        The default behaviour is to insert the given row to the :class:`Dataset`\n        object at the given index.\n       '
        if tags is None:
            tags = list()
        self._validate(row)
        self._data.insert(index, Row(row, tags=tags))

    def rpush(self, row, tags=None):
        if False:
            print('Hello World!')
        'Adds a row to the end of the :class:`Dataset`.\n        See :class:`Dataset.insert` for additional documentation.\n        '
        if tags is None:
            tags = list()
        self.insert(self.height, row=row, tags=tags)

    def append(self, row, tags=None):
        if False:
            while True:
                i = 10
        'Adds a row to the :class:`Dataset`.\n        See :class:`Dataset.insert` for additional documentation.\n        '
        if tags is None:
            tags = list()
        self.rpush(row, tags)

    def extend(self, rows, tags=None):
        if False:
            while True:
                i = 10
        'Adds a list of rows to the :class:`Dataset` using\n        :class:`Dataset.append`\n        '
        if tags is None:
            tags = list()
        for row in rows:
            self.append(row, tags)

    def remove_duplicates(self):
        if False:
            i = 10
            return i + 15
        'Removes all duplicate rows from the :class:`Dataset` object\n        while maintaining the original order.'
        seen = set()
        self._data[:] = [row for row in self._data if not (tuple(row) in seen or seen.add(tuple(row)))]

    def wipe(self):
        if False:
            print('Hello World!')
        'Removes all content and headers from the :class:`Dataset` object.'
        self._data = list()
        self.__headers = None
registry.register_builtins()

class InvalidDimensions(Exception):
    """Invalid size"""

class InvalidDatasetIndex(Exception):
    """Outside of Dataset size"""

class UnsupportedFormat(NotImplementedError):
    """Format is not supported"""