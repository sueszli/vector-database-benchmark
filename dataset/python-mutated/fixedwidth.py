"""An extensible ASCII table reader and writer.

fixedwidth.py:
  Read or write a table with fixed width columns.

:Copyright: Smithsonian Astrophysical Observatory (2011)
:Author: Tom Aldcroft (aldcroft@head.cfa.harvard.edu)
"""
from . import basic, core
from .core import DefaultSplitter, InconsistentTableError

class FixedWidthSplitter(core.BaseSplitter):
    """
    Split line based on fixed start and end positions for each ``col`` in
    ``self.cols``.

    This class requires that the Header class will have defined ``col.start``
    and ``col.end`` for each column.  The reference to the ``header.cols`` gets
    put in the splitter object by the base Reader.read() function just in time
    for splitting data lines by a ``data`` object.

    Note that the ``start`` and ``end`` positions are defined in the pythonic
    style so line[start:end] is the desired substring for a column.  This splitter
    class does not have a hook for ``process_lines`` since that is generally not
    useful for fixed-width input.

    """
    delimiter_pad = ''
    bookend = False
    delimiter = '|'

    def __call__(self, lines):
        if False:
            while True:
                i = 10
        for line in lines:
            vals = [line[x.start:x.end] for x in self.cols]
            if self.process_val:
                yield [self.process_val(x) for x in vals]
            else:
                yield vals

    def join(self, vals, widths):
        if False:
            for i in range(10):
                print('nop')
        pad = self.delimiter_pad or ''
        delimiter = self.delimiter or ''
        padded_delim = pad + delimiter + pad
        if self.bookend:
            bookend_left = delimiter + pad
            bookend_right = pad + delimiter
        else:
            bookend_left = ''
            bookend_right = ''
        vals = [' ' * (width - len(val)) + val for (val, width) in zip(vals, widths)]
        return bookend_left + padded_delim.join(vals) + bookend_right

class FixedWidthHeaderSplitter(DefaultSplitter):
    """Splitter class that splits on ``|``."""
    delimiter = '|'

class FixedWidthHeader(basic.BasicHeader):
    """
    Fixed width table header reader.
    """
    splitter_class = FixedWidthHeaderSplitter
    ' Splitter class for splitting data lines into columns '
    position_line = None
    ' row index of line that specifies position (default = 1) '
    set_of_position_line_characters = set('`~!#$%^&*-_+=\\|":\'')

    def get_line(self, lines, index):
        if False:
            for i in range(10):
                print('nop')
        for (i, line) in enumerate(self.process_lines(lines)):
            if i == index:
                break
        else:
            raise InconsistentTableError('No header line found in table')
        return line

    def get_cols(self, lines):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the header Column objects from the table ``lines``.\n\n        Based on the previously set Header attributes find or create the column names.\n        Sets ``self.cols`` with the list of Columns.\n\n        Parameters\n        ----------\n        lines : list\n            List of table lines\n\n        '
        header_rows = getattr(self, 'header_rows', ['name'])
        start_line = core._get_line_index(self.start_line, self.process_lines(lines))
        position_line = core._get_line_index(self.position_line, self.process_lines(lines))
        if start_line is None:
            if position_line is not None:
                raise ValueError('Cannot set position_line without also setting header_start')
            data_lines = self.data.data_lines
            if not data_lines:
                raise InconsistentTableError('No data lines found so cannot autogenerate column names')
            (vals, starts, ends) = self.get_fixedwidth_params(data_lines[0])
            self.names = [self.auto_format.format(i) for i in range(1, len(vals) + 1)]
        else:
            if position_line is not None:
                line = self.get_line(lines, position_line)
                if len(set(line) - {self.splitter.delimiter, ' '}) != 1:
                    raise InconsistentTableError('Position line should only contain delimiters and one other character, e.g. "--- ------- ---".')
                charset = self.set_of_position_line_characters.union({self.splitter.delimiter, ' '})
                if not set(line).issubset(charset):
                    raise InconsistentTableError(f'Characters in position line must be part of {charset}')
                (vals, self.col_starts, col_ends) = self.get_fixedwidth_params(line)
                self.col_ends = [x - 1 if x is not None else None for x in col_ends]
            line = self.get_line(lines, start_line + header_rows.index('name'))
            (self.names, starts, ends) = self.get_fixedwidth_params(line)
        self._set_cols_from_names()
        for (ii, attr) in enumerate(header_rows):
            if attr != 'name':
                line = self.get_line(lines, start_line + ii)
                vals = self.get_fixedwidth_params(line)[0]
                for (col, val) in zip(self.cols, vals):
                    if val:
                        setattr(col, attr, val)
        for (i, col) in enumerate(self.cols):
            col.start = starts[i]
            col.end = ends[i]

    def get_fixedwidth_params(self, line):
        if False:
            return 10
        '\n        Split ``line`` on the delimiter and determine column values and\n        column start and end positions.  This might include null columns with\n        zero length (e.g. for ``header row = "| col1 || col2 | col3 |"`` or\n        ``header2_row = "----- ------- -----"``).  The null columns are\n        stripped out.  Returns the values between delimiters and the\n        corresponding start and end positions.\n\n        Parameters\n        ----------\n        line : str\n            Input line\n\n        Returns\n        -------\n        vals : list\n            List of values.\n        starts : list\n            List of starting indices.\n        ends : list\n            List of ending indices.\n\n        '
        if self.col_starts is not None and self.col_ends is not None:
            starts = list(self.col_starts)
            ends = [x + 1 if x is not None else None for x in self.col_ends]
            if len(starts) != len(ends):
                raise ValueError('Fixed width col_starts and col_ends must have the same length')
            vals = [line[start:end].strip() for (start, end) in zip(starts, ends)]
        elif self.col_starts is None and self.col_ends is None:
            vals = line.split(self.splitter.delimiter)
            starts = [0]
            ends = []
            for val in vals:
                if val:
                    ends.append(starts[-1] + len(val))
                    starts.append(ends[-1] + 1)
                else:
                    starts[-1] += 1
            starts = starts[:-1]
            vals = [x.strip() for x in vals if x]
            if len(vals) != len(starts) or len(vals) != len(ends):
                raise InconsistentTableError('Error parsing fixed width header')
        else:
            if self.col_starts is not None:
                starts = list(self.col_starts)
                ends = starts[1:] + [None]
            else:
                ends = [x + 1 for x in self.col_ends]
                starts = [0] + ends[:-1]
            vals = [line[start:end].strip() for (start, end) in zip(starts, ends)]
        return (vals, starts, ends)

    def write(self, lines):
        if False:
            i = 10
            return i + 15
        pass

class FixedWidthData(basic.BasicData):
    """
    Base table data reader.
    """
    splitter_class = FixedWidthSplitter
    ' Splitter class for splitting data lines into columns '
    start_line = None

    def write(self, lines):
        if False:
            return 10
        default_header_rows = [] if self.header.start_line is None else ['name']
        header_rows = getattr(self, 'header_rows', default_header_rows)
        vals_list = list(zip(*self.str_vals()))
        hdrs_list = []
        for col_attr in header_rows:
            vals = ['' if (val := getattr(col.info, col_attr)) is None else str(val) for col in self.cols]
            hdrs_list.append(vals)
        widths = [max((len(vals[i_col]) for vals in vals_list)) for i_col in range(len(self.cols))]
        if hdrs_list:
            for i_col in range(len(self.cols)):
                widths[i_col] = max(widths[i_col], *(len(vals[i_col]) for vals in hdrs_list))
        for vals in hdrs_list:
            lines.append(self.splitter.join(vals, widths))
        if self.header.position_line is not None:
            vals = [self.header.position_char * width for width in widths]
            lines.append(self.splitter.join(vals, widths))
        for vals in vals_list:
            lines.append(self.splitter.join(vals, widths))
        return lines

class FixedWidth(basic.Basic):
    """Fixed width table with single header line defining column names and positions.

    Examples::

      # Bar delimiter in header and data

      |  Col1 |   Col2      |  Col3 |
      |  1.2  | hello there |     3 |
      |  2.4  | many words  |     7 |

      # Bar delimiter in header only

      Col1 |   Col2      | Col3
      1.2    hello there    3
      2.4    many words     7

      # No delimiter with column positions specified as input

      Col1       Col2Col3
       1.2hello there   3
       2.4many words    7

    See the :ref:`astropy:fixed_width_gallery` for specific usage examples.

    """
    _format_name = 'fixed_width'
    _description = 'Fixed width'
    header_class = FixedWidthHeader
    data_class = FixedWidthData

    def __init__(self, col_starts=None, col_ends=None, delimiter_pad=' ', bookend=True, header_rows=None):
        if False:
            print('Hello World!')
        if header_rows is None:
            header_rows = ['name']
        super().__init__()
        self.data.splitter.delimiter_pad = delimiter_pad
        self.data.splitter.bookend = bookend
        self.header.col_starts = col_starts
        self.header.col_ends = col_ends
        self.header.header_rows = header_rows
        self.data.header_rows = header_rows
        if self.data.start_line is None:
            self.data.start_line = len(header_rows)

class FixedWidthNoHeaderHeader(FixedWidthHeader):
    """Header reader for fixed with tables with no header line."""
    start_line = None

class FixedWidthNoHeaderData(FixedWidthData):
    """Data reader for fixed width tables with no header line."""
    start_line = 0

class FixedWidthNoHeader(FixedWidth):
    """Fixed width table which has no header line.

    When reading, column names are either input (``names`` keyword) or
    auto-generated.  Column positions are determined either by input
    (``col_starts`` and ``col_stops`` keywords) or by splitting the first data
    line.  In the latter case a ``delimiter`` is required to split the data
    line.

    Examples::

      # Bar delimiter in header and data

      |  1.2  | hello there |     3 |
      |  2.4  | many words  |     7 |

      # Compact table having no delimiter and column positions specified as input

      1.2hello there3
      2.4many words 7

    This class is just a convenience wrapper around the ``FixedWidth`` reader
    but with ``header_start=None`` and ``data_start=0``.

    See the :ref:`astropy:fixed_width_gallery` for specific usage examples.

    """
    _format_name = 'fixed_width_no_header'
    _description = 'Fixed width with no header'
    header_class = FixedWidthNoHeaderHeader
    data_class = FixedWidthNoHeaderData

    def __init__(self, col_starts=None, col_ends=None, delimiter_pad=' ', bookend=True):
        if False:
            print('Hello World!')
        super().__init__(col_starts, col_ends, delimiter_pad=delimiter_pad, bookend=bookend, header_rows=[])

class FixedWidthTwoLineHeader(FixedWidthHeader):
    """Header reader for fixed width tables splitting on whitespace.

    For fixed width tables with several header lines, there is typically
    a white-space delimited format line, so splitting on white space is
    needed.
    """
    splitter_class = DefaultSplitter

class FixedWidthTwoLineDataSplitter(FixedWidthSplitter):
    """Splitter for fixed width tables splitting on ``' '``."""
    delimiter = ' '

class FixedWidthTwoLineData(FixedWidthData):
    """Data reader for fixed with tables with two header lines."""
    splitter_class = FixedWidthTwoLineDataSplitter

class FixedWidthTwoLine(FixedWidth):
    """Fixed width table which has two header lines.

    The first header line defines the column names and the second implicitly
    defines the column positions.

    Examples::

      # Typical case with column extent defined by ---- under column names.

       col1    col2         <== header_start = 0
      -----  ------------   <== position_line = 1, position_char = "-"
        1     bee flies     <== data_start = 2
        2     fish swims

      # Pretty-printed table

      +------+------------+
      | Col1 |   Col2     |
      +------+------------+
      |  1.2 | "hello"    |
      |  2.4 | there world|
      +------+------------+

    See the :ref:`astropy:fixed_width_gallery` for specific usage examples.

    """
    _format_name = 'fixed_width_two_line'
    _description = 'Fixed width with second header line'
    data_class = FixedWidthTwoLineData
    header_class = FixedWidthTwoLineHeader

    def __init__(self, position_line=None, position_char='-', delimiter_pad=None, bookend=False, header_rows=None):
        if False:
            return 10
        if len(position_char) != 1:
            raise ValueError(f'Position_char="{position_char}" must be a single character')
        super().__init__(delimiter_pad=delimiter_pad, bookend=bookend, header_rows=header_rows)
        if position_line is None:
            position_line = len(self.header.header_rows)
        self.header.position_line = position_line
        self.header.position_char = position_char
        self.data.start_line = position_line + 1