"""
Display output in a table format
=================================

.. versionadded:: 2017.7.0

The ``table`` outputter displays a sequence of rows as table.

Example output:

.. code-block:: text

    edge01.bjm01:
    ----------
        comment:
        ----------
        out:
        ----------
            ______________________________________________________________________________
            | Active | Interface | Last Move |        Mac        | Moves | Static | Vlan |
            ______________________________________________________________________________
            |  True  |  ae1.900  |    0.0    | 40:A6:77:5A:50:01 |   0   | False  | 111  |
            ______________________________________________________________________________
            |  True  |  ae1.111  |    0.0    | 64:16:8D:32:26:58 |   0   | False  | 111  |
            ______________________________________________________________________________
            |  True  |  ae1.111  |    0.0    | 8C:60:4F:73:2D:57 |   0   | False  | 111  |
            ______________________________________________________________________________
            |  True  |  ae1.111  |    0.0    | 8C:60:4F:73:2D:7C |   0   | False  | 111  |
            ______________________________________________________________________________
            |  True  |  ae1.222  |    0.0    | 8C:60:4F:73:2D:57 |   0   | False  | 222  |
            ______________________________________________________________________________
            |  True  |  ae1.222  |    0.0    | F4:0F:1B:76:9D:97 |   0   | False  | 222  |
            ______________________________________________________________________________
        result:
        ----------


CLI Example:

.. code-block:: bash

    salt '*' foo.bar --out=table
"""
import operator
from functools import reduce
import salt.output
import salt.utils.color
import salt.utils.data
__virtualname__ = 'table'

def __virtual__():
    if False:
        return 10
    return __virtualname__

class TableDisplay:
    """
    Manage the table display content.
    """
    _JUSTIFY_MAP = {'center': str.center, 'right': str.rjust, 'left': str.ljust}

    def __init__(self, has_header=True, row_delimiter='-', delim=' | ', justify='center', separate_rows=True, prefix='| ', suffix=' |', width=50, wrapfunc=None):
        if False:
            i = 10
            return i + 15
        self.__dict__.update(salt.utils.color.get_colors(__opts__.get('color'), __opts__.get('color_theme')))
        self.strip_colors = __opts__.get('strip_colors', True)
        self.has_header = has_header
        self.row_delimiter = row_delimiter
        self.delim = delim
        self.justify = justify
        self.separate_rows = separate_rows
        self.prefix = prefix
        self.suffix = suffix
        self.width = width
        if not (wrapfunc and callable(wrapfunc)):
            self.wrapfunc = self.wrap_onspace
        else:
            self.wrapfunc = wrapfunc

    def ustring(self, indent, color, msg, prefix='', suffix='', endc=None):
        if False:
            i = 10
            return i + 15
        'Build the unicode string to be displayed.'
        if endc is None:
            endc = self.ENDC
        indent *= ' '
        fmt = '{0}{1}{2}{3}{4}{5}'
        try:
            return fmt.format(indent, color, prefix, msg, endc, suffix)
        except UnicodeDecodeError:
            return fmt.format(indent, color, prefix, salt.utils.data.decode(msg), endc, suffix)

    def wrap_onspace(self, text):
        if False:
            i = 10
            return i + 15
        '\n        When the text inside the column is longer then the width, will split by space and continue on the next line.'

        def _truncate(line, word):
            if False:
                while True:
                    i = 10
            return '{line}{part}{word}'.format(line=line, part=' \n'[len(line[line.rfind('\n') + 1:]) + len(word.split('\n', 1)[0]) >= self.width], word=word)
        return reduce(_truncate, text.split(' '))

    def prepare_rows(self, rows, indent, has_header):
        if False:
            while True:
                i = 10
        'Prepare rows content to be displayed.'
        out = []

        def row_wrapper(row):
            if False:
                print('Hello World!')
            new_rows = [self.wrapfunc(item).split('\n') for item in row]
            rows = []
            for item in map(lambda *args: args, *new_rows):
                if isinstance(item, (tuple, list)):
                    rows.append([substr or '' for substr in item])
                else:
                    rows.append([item])
            return rows
        logical_rows = [row_wrapper(row) for row in rows]
        columns = map(lambda *args: args, *reduce(operator.add, logical_rows))
        max_widths = [max((len(str(item)) for item in column)) for column in columns]
        row_separator = self.row_delimiter * (len(self.prefix) + len(self.suffix) + sum(max_widths) + len(self.delim) * (len(max_widths) - 1))
        justify = self._JUSTIFY_MAP[self.justify.lower()]
        if self.separate_rows:
            out.append(self.ustring(indent, self.LIGHT_GRAY, row_separator))
        for physical_rows in logical_rows:
            for row in physical_rows:
                line = self.prefix + self.delim.join([justify(str(item), width) for (item, width) in zip(row, max_widths)]) + self.suffix
                out.append(self.ustring(indent, self.WHITE, line))
            if self.separate_rows or has_header:
                out.append(self.ustring(indent, self.LIGHT_GRAY, row_separator))
                has_header = False
        return out

    def display_rows(self, rows, labels, indent):
        if False:
            i = 10
            return i + 15
        'Prepares row content and displays.'
        out = []
        if not rows:
            return out
        first_row_type = type(rows[0])
        consistent = True
        for row in rows[1:]:
            if type(row) != first_row_type:
                consistent = False
        if not consistent:
            return out
        if isinstance(labels, dict):
            labels_temp = []
            for key in sorted(labels):
                labels_temp.append(labels[key])
            labels = labels_temp
        if first_row_type is dict:
            temp_rows = []
            if not labels:
                labels = [str(label).replace('_', ' ').title() for label in sorted(rows[0])]
            for row in rows:
                temp_row = []
                for key in sorted(row):
                    temp_row.append(str(row[key]))
                temp_rows.append(temp_row)
            rows = temp_rows
        elif isinstance(rows[0], str):
            rows = [[row] for row in rows]
        labels_and_rows = [labels] + rows if labels else rows
        has_header = self.has_header and labels
        return self.prepare_rows(labels_and_rows, indent + 4, has_header)

    def display(self, ret, indent, out, rows_key=None, labels_key=None):
        if False:
            while True:
                i = 10
        'Display table(s).'
        rows = []
        labels = None
        if isinstance(ret, dict):
            if not rows_key or (rows_key and rows_key in list(ret.keys())):
                for key in sorted(ret):
                    if rows_key and key != rows_key:
                        continue
                    val = ret[key]
                    if not rows_key:
                        out.append(self.ustring(indent, self.DARK_GRAY, key, suffix=':'))
                        out.append(self.ustring(indent, self.DARK_GRAY, '----------'))
                    if isinstance(val, (list, tuple)):
                        rows = val
                        if labels_key:
                            labels = ret.get(labels_key)
                        out.extend(self.display_rows(rows, labels, indent))
                    else:
                        self.display(val, indent + 4, out, rows_key=rows_key, labels_key=labels_key)
            elif rows_key:
                for key in sorted(ret):
                    val = ret[key]
                    self.display(val, indent, out, rows_key=rows_key, labels_key=labels_key)
        elif isinstance(ret, (list, tuple)):
            if not rows_key:
                rows = ret
                out.extend(self.display_rows(rows, labels, indent))
        return out

def output(ret, **kwargs):
    if False:
        return 10
    '\n    Display the output as table.\n\n    Args:\n\n        * nested_indent: integer, specify the left alignment.\n        * has_header: boolean specifying if header should be displayed. Default: True.\n        * row_delimiter: character to separate rows. Default: ``_``.\n        * delim: character to separate columns. Default: ``" | "``.\n        * justify: text alignment. Default: ``center``.\n        * separate_rows: boolean specifying if row separator will be displayed between consecutive rows. Default: True.\n        * prefix: character at the beginning of the row. Default: ``"| "``.\n        * suffix: character at the end of the row. Default: ``" |"``.\n        * width: column max width. Default: ``50``.\n        * rows_key: display the rows under a specific key.\n        * labels_key: use the labels under a certain key. Otherwise will try to use the dictionary keys (if any).\n        * title: display title when only one table is selected (using the ``rows_key`` argument).\n    '
    if 'opts' in kwargs:
        global __opts__
        __opts__ = kwargs.pop('opts')
    base_indent = kwargs.get('nested_indent', 0) or __opts__.get('out.table.nested_indent', 0)
    rows_key = kwargs.get('rows_key') or __opts__.get('out.table.rows_key')
    labels_key = kwargs.get('labels_key') or __opts__.get('out.table.labels_key')
    title = kwargs.get('title') or __opts__.get('out.table.title')
    class_kvargs = {}
    argks = ('has_header', 'row_delimiter', 'delim', 'justify', 'separate_rows', 'prefix', 'suffix', 'width')
    for argk in argks:
        argv = kwargs.get(argk) or __opts__.get('out.table.{key}'.format(key=argk))
        if argv is not None:
            class_kvargs[argk] = argv
    table = TableDisplay(**class_kvargs)
    out = []
    if title and rows_key:
        out.append(table.ustring(base_indent, title, table.WHITE, suffix='\n'))
    return '\n'.join(table.display(salt.utils.data.decode(ret), base_indent, out, rows_key=rows_key, labels_key=labels_key))