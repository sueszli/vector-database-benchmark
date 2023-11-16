"""If there is one recurring theme in ``boltons``, it is that Python
has excellent datastructures that constitute a good foundation for
most quick manipulations, as well as building applications. However,
Python usage has grown much faster than builtin data structure
power. Python has a growing need for more advanced general-purpose
data structures which behave intuitively.

The :class:`Table` class is one example. When handed one- or
two-dimensional data, it can provide useful, if basic, text and HTML
renditions of small to medium sized data. It also heuristically
handles recursive data of various formats (lists, dicts, namedtuples,
objects).

For more advanced :class:`Table`-style manipulation check out the
`pandas`_ DataFrame.

.. _pandas: http://pandas.pydata.org/

"""
from __future__ import print_function
try:
    from html import escape as html_escape
except ImportError:
    from cgi import escape as html_escape
import types
from itertools import islice
try:
    from collections.abc import Sequence, Mapping, MutableSequence
except ImportError:
    from collections import Sequence, Mapping, MutableSequence
try:
    (string_types, integer_types) = ((str, unicode), (int, long))
    from cgi import escape as html_escape
except NameError:
    unicode = str
    (string_types, integer_types) = ((str, bytes), (int,))
    from html import escape as html_escape
try:
    from .typeutils import make_sentinel
    _MISSING = make_sentinel(var_name='_MISSING')
except ImportError:
    _MISSING = object()
"\nSome idle feature thoughts:\n\n* shift around column order without rearranging data\n* gotta make it so you can add additional items, not just initialize with\n* maybe a shortcut would be to allow adding of Tables to other Tables\n* what's the perf of preallocating lists and overwriting items versus\n  starting from empty?\n* is it possible to effectively tell the difference between when a\n  Table is from_data()'d with a single row (list) or with a list of lists?\n* CSS: white-space pre-line or pre-wrap maybe?\n* Would be nice to support different backends (currently uses lists\n  exclusively). Sometimes large datasets come in list-of-dicts and\n  list-of-tuples format and it's desirable to cut down processing overhead.\n\nTODO: make iterable on rows?\n"
__all__ = ['Table']

def to_text(obj, maxlen=None):
    if False:
        while True:
            i = 10
    try:
        text = unicode(obj)
    except Exception:
        try:
            text = unicode(repr(obj))
        except Exception:
            text = unicode(object.__repr__(obj))
    if maxlen and len(text) > maxlen:
        text = text[:maxlen - 3] + '...'
    return text

def escape_html(obj, maxlen=None):
    if False:
        print('Hello World!')
    text = to_text(obj, maxlen=maxlen)
    return html_escape(text, quote=True)
_DNR = set((type(None), bool, complex, float, type(NotImplemented), slice, types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.GeneratorType) + string_types + integer_types)

class UnsupportedData(TypeError):
    pass

class InputType(object):

    def __init__(self, *a, **kw):
        if False:
            return 10
        pass

    def get_entry_seq(self, data_seq, headers):
        if False:
            while True:
                i = 10
        return [self.get_entry(entry, headers) for entry in data_seq]

class DictInputType(InputType):

    def check_type(self, obj):
        if False:
            while True:
                i = 10
        return isinstance(obj, Mapping)

    def guess_headers(self, obj):
        if False:
            return 10
        return sorted(obj.keys())

    def get_entry(self, obj, headers):
        if False:
            i = 10
            return i + 15
        return [obj.get(h) for h in headers]

    def get_entry_seq(self, obj, headers):
        if False:
            print('Hello World!')
        return [[ci.get(h) for h in headers] for ci in obj]

class ObjectInputType(InputType):

    def check_type(self, obj):
        if False:
            print('Hello World!')
        return type(obj) not in _DNR and hasattr(obj, '__class__')

    def guess_headers(self, obj):
        if False:
            while True:
                i = 10
        headers = []
        for attr in dir(obj):
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            if callable(val):
                continue
            headers.append(attr)
        return headers

    def get_entry(self, obj, headers):
        if False:
            return 10
        values = []
        for h in headers:
            try:
                values.append(getattr(obj, h))
            except Exception:
                values.append(None)
        return values

class ListInputType(InputType):

    def check_type(self, obj):
        if False:
            print('Hello World!')
        return isinstance(obj, MutableSequence)

    def guess_headers(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return None

    def get_entry(self, obj, headers):
        if False:
            return 10
        return obj

    def get_entry_seq(self, obj_seq, headers):
        if False:
            print('Hello World!')
        return obj_seq

class TupleInputType(InputType):

    def check_type(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(obj, tuple)

    def guess_headers(self, obj):
        if False:
            i = 10
            return i + 15
        return None

    def get_entry(self, obj, headers):
        if False:
            i = 10
            return i + 15
        return list(obj)

    def get_entry_seq(self, obj_seq, headers):
        if False:
            while True:
                i = 10
        return [list(t) for t in obj_seq]

class NamedTupleInputType(InputType):

    def check_type(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(obj, '_fields') and isinstance(obj, tuple)

    def guess_headers(self, obj):
        if False:
            i = 10
            return i + 15
        return list(obj._fields)

    def get_entry(self, obj, headers):
        if False:
            return 10
        return [getattr(obj, h, None) for h in headers]

    def get_entry_seq(self, obj_seq, headers):
        if False:
            print('Hello World!')
        return [[getattr(obj, h, None) for h in headers] for obj in obj_seq]

class Table(object):
    """
    This Table class is meant to be simple, low-overhead, and extensible. Its
    most common use would be for translation between in-memory data
    structures and serialization formats, such as HTML and console-ready text.

    As such, it stores data in list-of-lists format, and *does not* copy
    lists passed in. It also reserves the right to modify those lists in a
    "filling" process, whereby short lists are extended to the width of
    the table (usually determined by number of headers). This greatly
    reduces overhead and processing/validation that would have to occur
    otherwise.

    General description of headers behavior:

    Headers describe the columns, but are not part of the data, however,
    if the *headers* argument is omitted, Table tries to infer header
    names from the data. It is possible to have a table with no headers,
    just pass in ``headers=None``.

    Supported inputs:

    * :class:`list` of :class:`list` objects
    * :class:`dict` (list/single)
    * :class:`object` (list/single)
    * :class:`collections.namedtuple` (list/single)
    * TODO: DB API cursor?
    * TODO: json

    Supported outputs:

    * HTML
    * Pretty text (also usable as GF Markdown)
    * TODO: CSV
    * TODO: json
    * TODO: json lines

    To minimize resident size, the Table data is stored as a list of lists.
    """
    _input_types = [DictInputType(), ListInputType(), NamedTupleInputType(), TupleInputType(), ObjectInputType()]
    (_html_tr, _html_tr_close) = ('<tr>', '</tr>')
    (_html_th, _html_th_close) = ('<th>', '</th>')
    (_html_td, _html_td_close) = ('<td>', '</td>')
    (_html_thead, _html_thead_close) = ('<thead>', '</thead>')
    (_html_tbody, _html_tbody_close) = ('<tbody>', '</tbody>')
    (_html_table_tag, _html_table_tag_close) = ('<table>', '</table>')

    def __init__(self, data=None, headers=_MISSING, metadata=None):
        if False:
            print('Hello World!')
        if headers is _MISSING:
            headers = []
            if data:
                (headers, data) = (list(data[0]), islice(data, 1, None))
        self.headers = headers or []
        self.metadata = metadata or {}
        self._data = []
        self._width = 0
        self.extend(data)

    def extend(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Append the given data to the end of the Table.\n        '
        if not data:
            return
        self._data.extend(data)
        self._set_width()
        self._fill()

    def _set_width(self, reset=False):
        if False:
            return 10
        if reset:
            self._width = 0
        if self._width:
            return
        if self.headers:
            self._width = len(self.headers)
            return
        self._width = max([len(d) for d in self._data])

    def _fill(self):
        if False:
            for i in range(10):
                print('nop')
        (width, filler) = (self._width, [None])
        if not width:
            return
        for d in self._data:
            rem = width - len(d)
            if rem > 0:
                d.extend(filler * rem)
        return

    @classmethod
    def from_dict(cls, data, headers=_MISSING, max_depth=1, metadata=None):
        if False:
            while True:
                i = 10
        'Create a Table from a :class:`dict`. Operates the same as\n        :meth:`from_data`, but forces interpretation of the data as a\n        Mapping.\n        '
        return cls.from_data(data=data, headers=headers, max_depth=max_depth, _data_type=DictInputType(), metadata=metadata)

    @classmethod
    def from_list(cls, data, headers=_MISSING, max_depth=1, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a Table from a :class:`list`. Operates the same as\n        :meth:`from_data`, but forces the interpretation of the data\n        as a Sequence.\n        '
        return cls.from_data(data=data, headers=headers, max_depth=max_depth, _data_type=ListInputType(), metadata=metadata)

    @classmethod
    def from_object(cls, data, headers=_MISSING, max_depth=1, metadata=None):
        if False:
            return 10
        'Create a Table from an :class:`object`. Operates the same as\n        :meth:`from_data`, but forces the interpretation of the data\n        as an object. May be useful for some :class:`dict` and\n        :class:`list` subtypes.\n        '
        return cls.from_data(data=data, headers=headers, max_depth=max_depth, _data_type=ObjectInputType(), metadata=metadata)

    @classmethod
    def from_data(cls, data, headers=_MISSING, max_depth=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create a Table from any supported data, heuristically\n        selecting how to represent the data in Table format.\n\n        Args:\n            data (object): Any object or iterable with data to be\n                imported to the Table.\n\n            headers (iterable): An iterable of headers to be matched\n                to the data. If not explicitly passed, headers will be\n                guessed for certain datatypes.\n\n            max_depth (int): The level to which nested Tables should\n                be created (default: 1).\n\n            _data_type (InputType subclass): For advanced use cases,\n                do not guess the type of the input data, use this data\n                type instead.\n        '
        metadata = kwargs.pop('metadata', None)
        _data_type = kwargs.pop('_data_type', None)
        if max_depth < 1:
            return cls(headers=headers, metadata=metadata)
        is_seq = isinstance(data, Sequence)
        if is_seq:
            if not data:
                return cls(headers=headers, metadata=metadata)
            to_check = data[0]
            if not _data_type:
                for it in cls._input_types:
                    if it.check_type(to_check):
                        _data_type = it
                        break
                else:
                    is_seq = False
                    to_check = data
        else:
            if type(data) in _DNR:
                return cls([[data]], headers=headers, metadata=metadata)
            to_check = data
        if not _data_type:
            for it in cls._input_types:
                if it.check_type(to_check):
                    _data_type = it
                    break
            else:
                raise UnsupportedData('unsupported data type %r' % type(data))
        if headers is _MISSING:
            headers = _data_type.guess_headers(to_check)
        if is_seq:
            entries = _data_type.get_entry_seq(data, headers)
        else:
            entries = [_data_type.get_entry(data, headers)]
        if max_depth > 1:
            new_max_depth = max_depth - 1
            for (i, entry) in enumerate(entries):
                for (j, cell) in enumerate(entry):
                    if type(cell) in _DNR:
                        continue
                    try:
                        entries[i][j] = cls.from_data(cell, max_depth=new_max_depth)
                    except UnsupportedData:
                        continue
        return cls(entries, headers=headers, metadata=metadata)

    def __len__(self):
        if False:
            return 10
        return len(self._data)

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        return self._data[idx]

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        cn = self.__class__.__name__
        if self.headers:
            return '%s(headers=%r, data=%r)' % (cn, self.headers, self._data)
        else:
            return '%s(%r)' % (cn, self._data)

    def to_html(self, orientation=None, wrapped=True, with_headers=True, with_newlines=True, with_metadata=False, max_depth=1):
        if False:
            print('Hello World!')
        "Render this Table to HTML. Configure the structure of Table\n        HTML by subclassing and overriding ``_html_*`` class\n        attributes.\n\n        Args:\n            orientation (str): one of 'auto', 'horizontal', or\n                'vertical' (or the first letter of any of\n                those). Default 'auto'.\n            wrapped (bool): whether or not to include the wrapping\n                '<table></table>' tags. Default ``True``, set to\n                ``False`` if appending multiple Table outputs or an\n                otherwise customized HTML wrapping tag is needed.\n            with_newlines (bool): Set to ``True`` if output should\n                include added newlines to make the HTML more\n                readable. Default ``False``.\n            with_metadata (bool/str): Set to ``True`` if output should\n                be preceded with a Table of preset metadata, if it\n                exists. Set to special value ``'bottom'`` if the\n                metadata Table HTML should come *after* the main HTML output.\n            max_depth (int): Indicate how deeply to nest HTML tables\n                before simply reverting to :func:`repr`-ing the nested\n                data.\n\n        Returns:\n            A text string of the HTML of the rendered table.\n\n        "
        lines = []
        headers = []
        if with_metadata and self.metadata:
            metadata_table = Table.from_data(self.metadata, max_depth=max_depth)
            metadata_html = metadata_table.to_html(with_headers=True, with_newlines=with_newlines, with_metadata=False, max_depth=max_depth)
            if with_metadata != 'bottom':
                lines.append(metadata_html)
                lines.append('<br />')
        if with_headers and self.headers:
            headers.extend(self.headers)
            headers.extend([None] * (self._width - len(self.headers)))
        if wrapped:
            lines.append(self._html_table_tag)
        orientation = orientation or 'auto'
        ol = orientation[0].lower()
        if ol == 'a':
            ol = 'h' if len(self) > 1 else 'v'
        if ol == 'h':
            self._add_horizontal_html_lines(lines, headers=headers, max_depth=max_depth)
        elif ol == 'v':
            self._add_vertical_html_lines(lines, headers=headers, max_depth=max_depth)
        else:
            raise ValueError("expected one of 'auto', 'vertical', or 'horizontal', not %r" % orientation)
        if with_metadata and self.metadata and (with_metadata == 'bottom'):
            lines.append('<br />')
            lines.append(metadata_html)
        if wrapped:
            lines.append(self._html_table_tag_close)
        sep = '\n' if with_newlines else ''
        return sep.join(lines)

    def get_cell_html(self, value):
        if False:
            print('Hello World!')
        'Called on each value in an HTML table. By default it simply escapes\n        the HTML. Override this method to add additional conditions\n        and behaviors, but take care to ensure the final output is\n        HTML escaped.\n        '
        return escape_html(value)

    def _add_horizontal_html_lines(self, lines, headers, max_depth):
        if False:
            return 10
        esc = self.get_cell_html
        new_depth = max_depth - 1 if max_depth > 1 else max_depth
        if max_depth > 1:
            new_depth = max_depth - 1
        if headers:
            _thth = self._html_th_close + self._html_th
            lines.append(self._html_thead)
            lines.append(self._html_tr + self._html_th + _thth.join([esc(h) for h in headers]) + self._html_th_close + self._html_tr_close)
            lines.append(self._html_thead_close)
        (trtd, _tdtd, _td_tr) = (self._html_tr + self._html_td, self._html_td_close + self._html_td, self._html_td_close + self._html_tr_close)
        lines.append(self._html_tbody)
        for row in self._data:
            if max_depth > 1:
                _fill_parts = []
                for cell in row:
                    if isinstance(cell, Table):
                        _fill_parts.append(cell.to_html(max_depth=new_depth))
                    else:
                        _fill_parts.append(esc(cell))
            else:
                _fill_parts = [esc(c) for c in row]
            lines.append(''.join([trtd, _tdtd.join(_fill_parts), _td_tr]))
        lines.append(self._html_tbody_close)

    def _add_vertical_html_lines(self, lines, headers, max_depth):
        if False:
            print('Hello World!')
        esc = self.get_cell_html
        new_depth = max_depth - 1 if max_depth > 1 else max_depth
        (tr, th, _th) = (self._html_tr, self._html_th, self._html_th_close)
        (td, _tdtd) = (self._html_td, self._html_td_close + self._html_td)
        _td_tr = self._html_td_close + self._html_tr_close
        for i in range(self._width):
            line_parts = [tr]
            if headers:
                line_parts.extend([th, esc(headers[i]), _th])
            if max_depth > 1:
                new_depth = max_depth - 1
                _fill_parts = []
                for row in self._data:
                    cell = row[i]
                    if isinstance(cell, Table):
                        _fill_parts.append(cell.to_html(max_depth=new_depth))
                    else:
                        _fill_parts.append(esc(row[i]))
            else:
                _fill_parts = [esc(row[i]) for row in self._data]
            line_parts.extend([td, _tdtd.join(_fill_parts), _td_tr])
            lines.append(''.join(line_parts))

    def to_text(self, with_headers=True, maxlen=None):
        if False:
            for i in range(10):
                print('nop')
        "Get the Table's textual representation. Only works well\n        for Tables with non-recursive data.\n\n        Args:\n            with_headers (bool): Whether to include a header row at the top.\n            maxlen (int): Max length of data in each cell.\n        "
        lines = []
        widths = []
        headers = list(self.headers)
        text_data = [[to_text(cell, maxlen=maxlen) for cell in row] for row in self._data]
        for idx in range(self._width):
            cur_widths = [len(cur) for cur in text_data]
            if with_headers:
                cur_widths.append(len(to_text(headers[idx], maxlen=maxlen)))
            widths.append(max(cur_widths))
        if with_headers:
            lines.append(' | '.join([h.center(widths[i]) for (i, h) in enumerate(headers)]))
            lines.append('-|-'.join(['-' * w for w in widths]))
        for row in text_data:
            lines.append(' | '.join([cell.center(widths[j]) for (j, cell) in enumerate(row)]))
        return '\n'.join(lines)