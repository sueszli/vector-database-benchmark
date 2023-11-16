"""An extensible ASCII table reader and writer.

ipac.py:
  Classes to read IPAC table format

:Copyright: Smithsonian Astrophysical Observatory (2011)
:Author: Tom Aldcroft (aldcroft@head.cfa.harvard.edu)
"""
import re
from collections import OrderedDict, defaultdict
from textwrap import wrap
from warnings import warn
from astropy.table.pprint import get_auto_format_func
from astropy.utils.exceptions import AstropyUserWarning
from . import basic, core, fixedwidth

class IpacFormatErrorDBMS(Exception):

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '{}\nSee {}'.format(super().__str__(), 'https://irsa.ipac.caltech.edu/applications/DDGEN/Doc/DBMSrestriction.html')

class IpacFormatError(Exception):

    def __str__(self):
        if False:
            return 10
        return '{}\nSee {}'.format(super().__str__(), 'https://irsa.ipac.caltech.edu/applications/DDGEN/Doc/ipac_tbl.html')

class IpacHeaderSplitter(core.BaseSplitter):
    """Splitter for Ipac Headers.

    This splitter is similar its parent when reading, but supports a
    fixed width format (as required for Ipac table headers) for writing.
    """
    process_line = None
    process_val = None
    delimiter = '|'
    delimiter_pad = ''
    skipinitialspace = False
    comment = '\\s*\\\\'
    write_comment = '\\\\'
    col_starts = None
    col_ends = None

    def join(self, vals, widths):
        if False:
            for i in range(10):
                print('nop')
        pad = self.delimiter_pad or ''
        delimiter = self.delimiter or ''
        padded_delim = pad + delimiter + pad
        bookend_left = delimiter + pad
        bookend_right = pad + delimiter
        vals = [' ' * (width - len(val)) + val for (val, width) in zip(vals, widths)]
        return bookend_left + padded_delim.join(vals) + bookend_right

class IpacHeader(fixedwidth.FixedWidthHeader):
    """IPAC table header."""
    splitter_class = IpacHeaderSplitter
    col_type_list = (('integer', core.IntType), ('long', core.IntType), ('double', core.FloatType), ('float', core.FloatType), ('real', core.FloatType), ('char', core.StrType), ('date', core.StrType))
    definition = 'ignore'
    start_line = None

    def process_lines(self, lines):
        if False:
            while True:
                i = 10
        'Generator to yield IPAC header lines, i.e. those starting and ending with\n        delimiter character (with trailing whitespace stripped).\n        '
        delim = self.splitter.delimiter
        for line in lines:
            line = line.rstrip()
            if line.startswith(delim) and line.endswith(delim):
                yield line.strip(delim)

    def update_meta(self, lines, meta):
        if False:
            while True:
                i = 10
        '\n        Extract table-level comments and keywords for IPAC table.  See:\n        https://irsa.ipac.caltech.edu/applications/DDGEN/Doc/ipac_tbl.html#kw.\n        '

        def process_keyword_value(val):
            if False:
                while True:
                    i = 10
            '\n            Take a string value and convert to float, int or str, and strip quotes\n            as needed.\n            '
            val = val.strip()
            try:
                val = int(val)
            except Exception:
                try:
                    val = float(val)
                except Exception:
                    for quote in ('"', "'"):
                        if val.startswith(quote) and val.endswith(quote):
                            val = val[1:-1]
                            break
            return val
        table_meta = meta['table']
        table_meta['comments'] = []
        table_meta['keywords'] = OrderedDict()
        keywords = table_meta['keywords']
        re_keyword = re.compile('\\\\(?P<name> \\w+)\\s* = (?P<value> .+) $', re.VERBOSE)
        for line in lines:
            if not line.startswith('\\'):
                break
            m = re_keyword.match(line)
            if m:
                name = m.group('name')
                val = process_keyword_value(m.group('value'))
                if name in keywords and isinstance(val, str):
                    prev_val = keywords[name]['value']
                    if isinstance(prev_val, str):
                        val = prev_val + val
                keywords[name] = {'value': val}
            elif line.startswith('\\ '):
                val = line[2:].strip()
                if val:
                    table_meta['comments'].append(val)

    def get_col_type(self, col):
        if False:
            return 10
        for (col_type_key, col_type) in self.col_type_list:
            if col_type_key.startswith(col.raw_type.lower()):
                return col_type
        raise ValueError(f'Unknown data type ""{col.raw_type}"" for column "{col.name}"')

    def get_cols(self, lines):
        if False:
            return 10
        '\n        Initialize the header Column objects from the table ``lines``.\n\n        Based on the previously set Header attributes find or create the column names.\n        Sets ``self.cols`` with the list of Columns.\n\n        Parameters\n        ----------\n        lines : list\n            List of table lines\n\n        '
        header_lines = self.process_lines(lines)
        header_vals = list(self.splitter(header_lines))
        if len(header_vals) == 0:
            raise ValueError('At least one header line beginning and ending with delimiter required')
        elif len(header_vals) > 4:
            raise ValueError('More than four header lines were found')
        cols = []
        start = 1
        for (i, name) in enumerate(header_vals[0]):
            col = core.Column(name=name.strip(' -'))
            col.start = start
            col.end = start + len(name)
            if len(header_vals) > 1:
                col.raw_type = header_vals[1][i].strip(' -')
                col.type = self.get_col_type(col)
            if len(header_vals) > 2:
                col.unit = header_vals[2][i].strip() or None
            if len(header_vals) > 3:
                null = header_vals[3][i].strip()
                fillval = '' if issubclass(col.type, core.StrType) else '0'
                self.data.fill_values.append((null, fillval, col.name))
            start = col.end + 1
            cols.append(col)
            if self.ipac_definition == 'right':
                col.start -= 1
            elif self.ipac_definition == 'left':
                col.end += 1
        self.names = [x.name for x in cols]
        self.cols = cols

    def str_vals(self):
        if False:
            for i in range(10):
                print('nop')
        if self.DBMS:
            IpacFormatE = IpacFormatErrorDBMS
        else:
            IpacFormatE = IpacFormatError
        namelist = self.colnames
        if self.DBMS:
            countnamelist = defaultdict(int)
            for name in self.colnames:
                countnamelist[name.lower()] += 1
            doublenames = [x for x in countnamelist if countnamelist[x] > 1]
            if doublenames != []:
                raise IpacFormatE(f'IPAC DBMS tables are not case sensitive. This causes duplicate column names: {doublenames}')
        for name in namelist:
            m = re.match('\\w+', name)
            if m.end() != len(name):
                raise IpacFormatE(f'{name} - Only alphanumeric characters and _ are allowed in column names.')
            if self.DBMS and (not (name[0].isalpha() or name[0] == '_')):
                raise IpacFormatE(f'Column name cannot start with numbers: {name}')
            if self.DBMS:
                if name in ['x', 'y', 'z', 'X', 'Y', 'Z']:
                    raise IpacFormatE(f'{name} - x, y, z, X, Y, Z are reserved names and cannot be used as column names.')
                if len(name) > 16:
                    raise IpacFormatE(f'{name} - Maximum length for column name is 16 characters')
            elif len(name) > 40:
                raise IpacFormatE(f'{name} - Maximum length for column name is 40 characters.')
        dtypelist = []
        unitlist = []
        nullist = []
        for col in self.cols:
            col_dtype = col.info.dtype
            col_unit = col.info.unit
            col_format = col.info.format
            if col_dtype.kind in ['i', 'u']:
                if col_dtype.itemsize <= 2:
                    dtypelist.append('int')
                else:
                    dtypelist.append('long')
            elif col_dtype.kind == 'f':
                if col_dtype.itemsize <= 4:
                    dtypelist.append('float')
                else:
                    dtypelist.append('double')
            else:
                dtypelist.append('char')
            if col_unit is None:
                unitlist.append('')
            else:
                unitlist.append(str(col.info.unit))
            null = col.fill_values[core.masked]
            try:
                auto_format_func = get_auto_format_func(col)
                format_func = col.info._format_funcs.get(col_format, auto_format_func)
                nullist.append(format_func(col_format, null).strip())
            except Exception:
                nullist.append(str(null).strip())
        return [namelist, dtypelist, unitlist, nullist]

    def write(self, lines, widths):
        if False:
            while True:
                i = 10
        'Write header.\n\n        The width of each column is determined in Ipac.write. Writing the header\n        must be delayed until that time.\n        This function is called from there, once the width information is\n        available.\n        '
        for vals in self.str_vals():
            lines.append(self.splitter.join(vals, widths))
        return lines

class IpacDataSplitter(fixedwidth.FixedWidthSplitter):
    delimiter = ' '
    delimiter_pad = ''
    bookend = True

class IpacData(fixedwidth.FixedWidthData):
    """IPAC table data reader."""
    comment = '[|\\\\]'
    start_line = 0
    splitter_class = IpacDataSplitter
    fill_values = [(core.masked, 'null')]

    def write(self, lines, widths, vals_list):
        if False:
            i = 10
            return i + 15
        'IPAC writer, modified from FixedWidth writer.'
        for vals in vals_list:
            lines.append(self.splitter.join(vals, widths))
        return lines

class Ipac(basic.Basic):
    """IPAC format table.

    See: https://irsa.ipac.caltech.edu/applications/DDGEN/Doc/ipac_tbl.html

    Example::

      \\\\name=value
      \\\\ Comment
      |  column1 |   column2 | column3 | column4  |    column5    |
      |  double  |   double  |   int   |   double |     char      |
      |  unit    |   unit    |   unit  |    unit  |     unit      |
      |  null    |   null    |   null  |    null  |     null      |
       2.0978     29.09056    73765     2.06000    B8IVpMnHg

    Or::

      |-----ra---|----dec---|---sao---|------v---|----sptype--------|
        2.09708   29.09056     73765   2.06000    B8IVpMnHg

    The comments and keywords defined in the header are available via the output
    table ``meta`` attribute::

      >>> import os
      >>> from astropy.io import ascii
      >>> filename = os.path.join(ascii.__path__[0], 'tests/data/ipac.dat')
      >>> data = ascii.read(filename)
      >>> print(data.meta['comments'])
      ['This is an example of a valid comment']
      >>> for name, keyword in data.meta['keywords'].items():
      ...     print(name, keyword['value'])
      ...
      intval 1
      floatval 2300.0
      date Wed Sp 20 09:48:36 1995
      key_continue IPAC keywords can continue across lines

    Note that there are different conventions for characters occurring below the
    position of the ``|`` symbol in IPAC tables. By default, any character
    below a ``|`` will be ignored (since this is the current standard),
    but if you need to read files that assume characters below the ``|``
    symbols belong to the column before or after the ``|``, you can specify
    ``definition='left'`` or ``definition='right'`` respectively when reading
    the table (the default is ``definition='ignore'``). The following examples
    demonstrate the different conventions:

    * ``definition='ignore'``::

        |   ra  |  dec  |
        | float | float |
          1.2345  6.7890

    * ``definition='left'``::

        |   ra  |  dec  |
        | float | float |
           1.2345  6.7890

    * ``definition='right'``::

        |   ra  |  dec  |
        | float | float |
        1.2345  6.7890

    IPAC tables can specify a null value in the header that is shown in place
    of missing or bad data. On writing, this value defaults to ``null``.
    To specify a different null value, use the ``fill_values`` option to
    replace masked values with a string or number of your choice as
    described in :ref:`astropy:io_ascii_write_parameters`::

        >>> from astropy.io.ascii import masked
        >>> fill = [(masked, 'N/A', 'ra'), (masked, -999, 'sptype')]
        >>> ascii.write(data, format='ipac', fill_values=fill)
        \\ This is an example of a valid comment
        ...
        |          ra|         dec|      sai|          v2|            sptype|
        |      double|      double|     long|      double|              char|
        |        unit|        unit|     unit|        unit|              ergs|
        |         N/A|        null|     null|        null|              -999|
                  N/A     29.09056      null         2.06               -999
         2345678901.0 3456789012.0 456789012 4567890123.0 567890123456789012

    When writing a table with a column of integers, the data type is output
    as ``int`` when the column ``dtype.itemsize`` is less than or equal to 2;
    otherwise the data type is ``long``. For a column of floating-point values,
    the data type is ``float`` when ``dtype.itemsize`` is less than or equal
    to 4; otherwise the data type is ``double``.

    Parameters
    ----------
    definition : str, optional
        Specify the convention for characters in the data table that occur
        directly below the pipe (``|``) symbol in the header column definition:

          * 'ignore' - Any character beneath a pipe symbol is ignored (default)
          * 'right' - Character is associated with the column to the right
          * 'left' - Character is associated with the column to the left

    DBMS : bool, optional
        If true, this verifies that written tables adhere (semantically)
        to the `IPAC/DBMS
        <https://irsa.ipac.caltech.edu/applications/DDGEN/Doc/DBMSrestriction.html>`_
        definition of IPAC tables. If 'False' it only checks for the (less strict)
        `IPAC <https://irsa.ipac.caltech.edu/applications/DDGEN/Doc/ipac_tbl.html>`_
        definition.
    """
    _format_name = 'ipac'
    _io_registry_format_aliases = ['ipac']
    _io_registry_can_write = True
    _description = 'IPAC format table'
    data_class = IpacData
    header_class = IpacHeader

    def __init__(self, definition='ignore', DBMS=False):
        if False:
            return 10
        super().__init__()
        if definition in ['ignore', 'left', 'right']:
            self.header.ipac_definition = definition
        else:
            raise ValueError('definition should be one of ignore/left/right')
        self.header.DBMS = DBMS

    def write(self, table):
        if False:
            while True:
                i = 10
        '\n        Write ``table`` as list of strings.\n\n        Parameters\n        ----------\n        table : `~astropy.table.Table`\n            Input table data\n\n        Returns\n        -------\n        lines : list\n            List of strings corresponding to ASCII table\n\n        '
        self.data.fill_values.append((core.masked, 'null'))
        self.header.cols = list(table.columns.values())
        self.header.check_column_names(self.names, self.strict_names, self.guessing)
        core._apply_include_exclude_names(table, self.names, self.include_names, self.exclude_names)
        self._check_multidim_table(table)
        new_cols = list(table.columns.values())
        self.header.cols = new_cols
        self.data.cols = new_cols
        lines = []
        raw_comments = table.meta.get('comments', [])
        for comment in raw_comments:
            lines.extend(wrap(str(comment), 80, initial_indent='\\ ', subsequent_indent='\\ '))
        if len(lines) > len(raw_comments):
            warn(f'Wrapping comment lines > 78 characters produced {len(lines) - len(raw_comments)} extra line(s)', AstropyUserWarning)
        if 'keywords' in table.meta:
            keydict = table.meta['keywords']
            for keyword in keydict:
                try:
                    val = keydict[keyword]['value']
                    lines.append(f'\\{keyword.strip()}={val!r}')
                except TypeError:
                    warn(f"Table metadata keyword {keyword} has been skipped.  IPAC metadata must be in the form {{{{'keywords':{{{{'keyword': {{{{'value': value}}}} }}}}", AstropyUserWarning)
        ignored_keys = [key for key in table.meta if key not in ('keywords', 'comments')]
        if any(ignored_keys):
            warn(f"Table metadata keyword(s) {ignored_keys} were not written.  IPAC metadata must be in the form {{{{'keywords':{{{{'keyword': {{{{'value': value}}}} }}}}", AstropyUserWarning)
        self.data._set_fill_values(self.data.cols)
        for (i, col) in enumerate(table.columns.values()):
            col.headwidth = max((len(vals[i]) for vals in self.header.str_vals()))
        data_str_vals = list(zip(*self.data.str_vals()))
        for (i, col) in enumerate(table.columns.values()):
            if data_str_vals:
                col.width = max((len(vals[i]) for vals in data_str_vals))
            else:
                col.width = 0
        widths = [max(col.width, col.headwidth) for col in table.columns.values()]
        self.header.write(lines, widths)
        self.data.write(lines, widths, data_str_vals)
        return lines