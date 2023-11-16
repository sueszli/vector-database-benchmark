import fnmatch
import os
import re
import sys
import numpy as np
from astropy import log
from astropy.utils.console import Getch, color_print, conf, terminal_size
from astropy.utils.data_info import dtype_info_name
__all__ = []

def default_format_func(format_, val):
    if False:
        while True:
            i = 10
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    else:
        return str(val)

def _use_str_for_masked_values(format_func):
    if False:
        for i in range(10):
            print('nop')
    'Wrap format function to trap masked values.\n\n    String format functions and most user functions will not be able to deal\n    with masked values, so we wrap them to ensure they are passed to str().\n    '
    return lambda format_, val: str(val) if val is np.ma.masked else format_func(format_, val)

def _possible_string_format_functions(format_):
    if False:
        while True:
            i = 10
    'Iterate through possible string-derived format functions.\n\n    A string can either be a format specifier for the format built-in,\n    a new-style format string, or an old-style format string.\n    '
    yield (lambda format_, val: format(val, format_))
    yield (lambda format_, val: format_.format(val))
    yield (lambda format_, val: format_ % val)
    yield (lambda format_, val: format_.format(**{k: val[k] for k in val.dtype.names}))

def get_auto_format_func(col=None, possible_string_format_functions=_possible_string_format_functions):
    if False:
        print('Hello World!')
    '\n    Return a wrapped ``auto_format_func`` function which is used in\n    formatting table columns.  This is primarily an internal function but\n    gets used directly in other parts of astropy, e.g. `astropy.io.ascii`.\n\n    Parameters\n    ----------\n    col_name : object, optional\n        Hashable object to identify column like id or name. Default is None.\n\n    possible_string_format_functions : func, optional\n        Function that yields possible string formatting functions\n        (defaults to internal function to do this).\n\n    Returns\n    -------\n    Wrapped ``auto_format_func`` function\n    '

    def _auto_format_func(format_, val):
        if False:
            while True:
                i = 10
        'Format ``val`` according to ``format_`` for a plain format specifier,\n        old- or new-style format strings, or using a user supplied function.\n        More importantly, determine and cache (in _format_funcs) a function\n        that will do this subsequently.  In this way this complicated logic is\n        only done for the first value.\n\n        Returns the formatted value.\n        '
        if format_ is None:
            return default_format_func(format_, val)
        if format_ in col.info._format_funcs:
            return col.info._format_funcs[format_](format_, val)
        if callable(format_):
            format_func = lambda format_, val: format_(val)
            try:
                out = format_func(format_, val)
                if not isinstance(out, str):
                    raise ValueError(f'Format function for value {val} returned {type(val)} instead of string type')
            except Exception as err:
                if val is np.ma.masked:
                    return str(val)
                raise ValueError(f'Format function for value {val} failed.') from err
            try:
                format_func(format_, np.ma.masked)
            except Exception:
                format_func = _use_str_for_masked_values(format_func)
        else:
            if val is np.ma.masked:
                return str(val)
            for format_func in possible_string_format_functions(format_):
                try:
                    out = format_func(format_, val)
                    if out == format_:
                        raise ValueError('the format passed in did nothing.')
                except Exception:
                    continue
                else:
                    break
            else:
                raise ValueError(f'unable to parse format string {format_} for its column.')
            format_func = _use_str_for_masked_values(format_func)
        col.info._format_funcs[format_] = format_func
        return out
    return _auto_format_func

def _get_pprint_include_names(table):
    if False:
        return 10
    'Get the set of names to show in pprint from the table pprint_include_names\n    and pprint_exclude_names attributes.\n\n    These may be fnmatch unix-style globs.\n    '

    def get_matches(name_globs, default):
        if False:
            for i in range(10):
                print('nop')
        match_names = set()
        if name_globs:
            for name in table.colnames:
                for name_glob in name_globs:
                    if fnmatch.fnmatch(name, name_glob):
                        match_names.add(name)
                        break
        else:
            match_names.update(default)
        return match_names
    include_names = get_matches(table.pprint_include_names(), table.colnames)
    exclude_names = get_matches(table.pprint_exclude_names(), [])
    return include_names - exclude_names

class TableFormatter:

    @staticmethod
    def _get_pprint_size(max_lines=None, max_width=None):
        if False:
            i = 10
            return i + 15
        'Get the output size (number of lines and character width) for Column and\n        Table pformat/pprint methods.\n\n        If no value of ``max_lines`` is supplied then the height of the\n        screen terminal is used to set ``max_lines``.  If the terminal\n        height cannot be determined then the default will be determined\n        using the ``astropy.table.conf.max_lines`` configuration item. If a\n        negative value of ``max_lines`` is supplied then there is no line\n        limit applied.\n\n        The same applies for max_width except the configuration item is\n        ``astropy.table.conf.max_width``.\n\n        Parameters\n        ----------\n        max_lines : int or None\n            Maximum lines of output (header + data rows)\n\n        max_width : int or None\n            Maximum width (characters) output\n\n        Returns\n        -------\n        max_lines, max_width : int\n\n        '
        lines = None
        width = None
        if max_lines is None:
            max_lines = conf.max_lines
        if max_width is None:
            max_width = conf.max_width
        if max_lines is None or max_width is None:
            (lines, width) = terminal_size()
        if max_lines is None:
            max_lines = lines
        elif max_lines < 0:
            max_lines = sys.maxsize
        if max_lines < 8:
            max_lines = 8
        if max_width is None:
            max_width = width
        elif max_width < 0:
            max_width = sys.maxsize
        if max_width < 10:
            max_width = 10
        return (max_lines, max_width)

    def _pformat_col(self, col, max_lines=None, show_name=True, show_unit=None, show_dtype=False, show_length=None, html=False, align=None):
        if False:
            for i in range(10):
                print('nop')
        "Return a list of formatted string representation of column values.\n\n        Parameters\n        ----------\n        max_lines : int\n            Maximum lines of output (header + data rows)\n\n        show_name : bool\n            Include column name. Default is True.\n\n        show_unit : bool\n            Include a header row for unit.  Default is to show a row\n            for units only if one or more columns has a defined value\n            for the unit.\n\n        show_dtype : bool\n            Include column dtype. Default is False.\n\n        show_length : bool\n            Include column length at end.  Default is to show this only\n            if the column is not shown completely.\n\n        html : bool\n            Output column as HTML\n\n        align : str\n            Left/right alignment of columns. Default is '>' (right) for all\n            columns. Other allowed values are '<', '^', and '0=' for left,\n            centered, and 0-padded, respectively.\n\n        Returns\n        -------\n        lines : list\n            List of lines with formatted column values\n\n        outs : dict\n            Dict which is used to pass back additional values\n            defined within the iterator.\n\n        "
        if show_unit is None:
            show_unit = col.info.unit is not None
        outs = {}
        col_strs_iter = self._pformat_col_iter(col, max_lines, show_name=show_name, show_unit=show_unit, show_dtype=show_dtype, show_length=show_length, outs=outs)
        col_strs = [val.replace('\t', '\\t').replace('\n', '\\n') for val in col_strs_iter]
        if len(col_strs) > 0:
            col_width = max((len(x) for x in col_strs))
        if html:
            from astropy.utils.xml.writer import xml_escape
            n_header = outs['n_header']
            for (i, col_str) in enumerate(col_strs):
                if i == n_header - 1:
                    continue
                td = 'th' if i < n_header else 'td'
                val = f'<{td}>{xml_escape(col_str.strip())}</{td}>'
                row = '<tr>' + val + '</tr>'
                if i < n_header:
                    row = '<thead>' + row + '</thead>'
                col_strs[i] = row
            if n_header > 0:
                col_strs.pop(n_header - 1)
            col_strs.insert(0, '<table>')
            col_strs.append('</table>')
        else:
            col_width = max((len(x) for x in col_strs)) if col_strs else 1
            for i in outs['i_centers']:
                col_strs[i] = col_strs[i].center(col_width)
            if outs['i_dashes'] is not None:
                col_strs[outs['i_dashes']] = '-' * col_width
            re_fill_align = re.compile('(?P<fill>.?)(?P<align>[<^>=])')
            match = None
            if align:
                match = re_fill_align.match(align)
                if not match:
                    raise ValueError("column align must be one of '<', '^', '>', or '='")
            elif isinstance(col.info.format, str):
                match = re_fill_align.match(col.info.format)
            if match:
                fill_char = match.group('fill')
                align_char = match.group('align')
                if align_char == '=':
                    if fill_char != '0':
                        raise ValueError("fill character must be '0' for '=' align")
                    fill_char = ''
            else:
                fill_char = ''
                align_char = '>'
            justify_methods = {'<': 'ljust', '^': 'center', '>': 'rjust', '=': 'zfill'}
            justify_method = justify_methods[align_char]
            justify_args = (col_width, fill_char) if fill_char else (col_width,)
            for (i, col_str) in enumerate(col_strs):
                col_strs[i] = getattr(col_str, justify_method)(*justify_args)
        if outs['show_length']:
            col_strs.append(f'Length = {len(col)} rows')
        return (col_strs, outs)

    def _name_and_structure(self, name, dtype, sep=' '):
        if False:
            print('Hello World!')
        'Format a column name, including a possible structure.\n\n        Normally, just returns the name, but if it has a structured dtype,\n        will add the parts in between square brackets.  E.g.,\n        "name [f0, f1]" or "name [f0[sf0, sf1], f1]".\n        '
        if dtype is None or dtype.names is None:
            return name
        structure = ', '.join([self._name_and_structure(name, dt, sep='') for (name, (dt, _)) in dtype.fields.items()])
        return f'{name}{sep}[{structure}]'

    def _pformat_col_iter(self, col, max_lines, show_name, show_unit, outs, show_dtype=False, show_length=None):
        if False:
            return 10
        'Iterator which yields formatted string representation of column values.\n\n        Parameters\n        ----------\n        max_lines : int\n            Maximum lines of output (header + data rows)\n\n        show_name : bool\n            Include column name. Default is True.\n\n        show_unit : bool\n            Include a header row for unit.  Default is to show a row\n            for units only if one or more columns has a defined value\n            for the unit.\n\n        outs : dict\n            Must be a dict which is used to pass back additional values\n            defined within the iterator.\n\n        show_dtype : bool\n            Include column dtype. Default is False.\n\n        show_length : bool\n            Include column length at end.  Default is to show this only\n            if the column is not shown completely.\n        '
        (max_lines, _) = self._get_pprint_size(max_lines, -1)
        dtype = getattr(col, 'dtype', None)
        multidims = getattr(col, 'shape', [0])[1:]
        if multidims:
            multidim0 = tuple((0 for n in multidims))
            multidim1 = tuple((n - 1 for n in multidims))
            multidims_all_ones = np.prod(multidims) == 1
            multidims_has_zero = 0 in multidims
        i_dashes = None
        i_centers = []
        n_header = 0
        if show_name:
            i_centers.append(n_header)
            col_name = str(col.info.name)
            n_header += 1
            yield self._name_and_structure(col_name, dtype)
        if show_unit:
            i_centers.append(n_header)
            n_header += 1
            yield str(col.info.unit or '')
        if show_dtype:
            i_centers.append(n_header)
            n_header += 1
            if dtype is not None:
                col_dtype = dtype_info_name((dtype, multidims))
            else:
                col_dtype = col.__class__.__qualname__ or 'object'
            yield col_dtype
        if show_unit or show_name or show_dtype:
            i_dashes = n_header
            n_header += 1
            yield '---'
        max_lines -= n_header
        n_print2 = max_lines // 2
        n_rows = len(col)
        col_format = col.info.format or getattr(col.info, 'default_format', None)
        pssf = getattr(col.info, 'possible_string_format_functions', None) or _possible_string_format_functions
        auto_format_func = get_auto_format_func(col, pssf)
        format_func = col.info._format_funcs.get(col_format, auto_format_func)
        if len(col) > max_lines:
            if show_length is None:
                show_length = True
            i0 = n_print2 - (1 if show_length else 0)
            i1 = n_rows - n_print2 - max_lines % 2
            indices = np.concatenate([np.arange(0, i0 + 1), np.arange(i1 + 1, len(col))])
        else:
            i0 = -1
            indices = np.arange(len(col))

        def format_col_str(idx):
            if False:
                while True:
                    i = 10
            if multidims:
                if multidims_all_ones:
                    return format_func(col_format, col[(idx,) + multidim0])
                elif multidims_has_zero:
                    return ''
                else:
                    left = format_func(col_format, col[(idx,) + multidim0])
                    right = format_func(col_format, col[(idx,) + multidim1])
                    return f'{left} .. {right}'
            else:
                return format_func(col_format, col[idx])
        for idx in indices:
            if idx == i0:
                yield '...'
            else:
                try:
                    yield format_col_str(idx)
                except ValueError:
                    raise ValueError('Unable to parse format string "{}" for entry "{}" in column "{}"'.format(col_format, col[idx], col.info.name))
        outs['show_length'] = show_length
        outs['n_header'] = n_header
        outs['i_centers'] = i_centers
        outs['i_dashes'] = i_dashes

    def _pformat_table(self, table, max_lines=None, max_width=None, show_name=True, show_unit=None, show_dtype=False, html=False, tableid=None, tableclass=None, align=None):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of lines for the formatted string representation of\n        the table.\n\n        Parameters\n        ----------\n        max_lines : int or None\n            Maximum number of rows to output\n\n        max_width : int or None\n            Maximum character width of output\n\n        show_name : bool\n            Include a header row for column names. Default is True.\n\n        show_unit : bool\n            Include a header row for unit.  Default is to show a row\n            for units only if one or more columns has a defined value\n            for the unit.\n\n        show_dtype : bool\n            Include a header row for column dtypes. Default is to False.\n\n        html : bool\n            Format the output as an HTML table. Default is False.\n\n        tableid : str or None\n            An ID tag for the table; only used if html is set.  Default is\n            "table{id}", where id is the unique integer id of the table object,\n            id(table)\n\n        tableclass : str or list of str or None\n            CSS classes for the table; only used if html is set.  Default is\n            none\n\n        align : str or list or tuple\n            Left/right alignment of columns. Default is \'>\' (right) for all\n            columns. Other allowed values are \'<\', \'^\', and \'0=\' for left,\n            centered, and 0-padded, respectively. A list of strings can be\n            provided for alignment of tables with multiple columns.\n\n        Returns\n        -------\n        rows : list\n            Formatted table as a list of strings\n\n        outs : dict\n            Dict which is used to pass back additional values\n            defined within the iterator.\n\n        '
        (max_lines, max_width) = self._get_pprint_size(max_lines, max_width)
        if show_unit is None:
            show_unit = any((col.info.unit for col in table.columns.values()))
        n_cols = len(table.columns)
        if align is None or isinstance(align, str):
            align = [align] * n_cols
        elif isinstance(align, (list, tuple)):
            if len(align) != n_cols:
                raise ValueError(f'got {len(align)} alignment values instead of the number of columns ({n_cols})')
        else:
            raise TypeError(f'align keyword must be str or list or tuple (got {type(align)})')
        pprint_include_names = _get_pprint_include_names(table)
        cols = []
        outs = None
        for (align_, col) in zip(align, table.columns.values()):
            if col.info.name not in pprint_include_names:
                continue
            (lines, outs) = self._pformat_col(col, max_lines, show_name=show_name, show_unit=show_unit, show_dtype=show_dtype, align=align_)
            if outs['show_length']:
                lines = lines[:-1]
            cols.append(lines)
        if not cols:
            return (['<No columns>'], {'show_length': False})
        n_header = outs['n_header']
        n_rows = len(cols[0])

        def outwidth(cols):
            if False:
                return 10
            return sum((len(c[0]) for c in cols)) + len(cols) - 1
        dots_col = ['...'] * n_rows
        middle = len(cols) // 2
        while outwidth(cols) > max_width:
            if len(cols) == 1:
                break
            if len(cols) == 2:
                cols[1] = dots_col
                break
            if cols[middle] is dots_col:
                cols.pop(middle)
                middle = len(cols) // 2
            cols[middle] = dots_col
        rows = []
        if html:
            from astropy.utils.xml.writer import xml_escape
            if tableid is None:
                tableid = f'table{id(table)}'
            if tableclass is not None:
                if isinstance(tableclass, list):
                    tableclass = ' '.join(tableclass)
                rows.append(f'<table id="{tableid}" class="{tableclass}">')
            else:
                rows.append(f'<table id="{tableid}">')
            for i in range(n_rows):
                if i == n_header - 1:
                    continue
                td = 'th' if i < n_header else 'td'
                vals = (f'<{td}>{xml_escape(col[i].strip())}</{td}>' for col in cols)
                row = '<tr>' + ''.join(vals) + '</tr>'
                if i < n_header:
                    row = '<thead>' + row + '</thead>'
                rows.append(row)
            rows.append('</table>')
        else:
            for i in range(n_rows):
                row = ' '.join((col[i] for col in cols))
                rows.append(row)
        return (rows, outs)

    def _more_tabcol(self, tabcol, max_lines=None, max_width=None, show_name=True, show_unit=None, show_dtype=False):
        if False:
            i = 10
            return i + 15
        'Interactive "more" of a table or column.\n\n        Parameters\n        ----------\n        max_lines : int or None\n            Maximum number of rows to output\n\n        max_width : int or None\n            Maximum character width of output\n\n        show_name : bool\n            Include a header row for column names. Default is True.\n\n        show_unit : bool\n            Include a header row for unit.  Default is to show a row\n            for units only if one or more columns has a defined value\n            for the unit.\n\n        show_dtype : bool\n            Include a header row for column dtypes. Default is False.\n        '
        allowed_keys = 'f br<>qhpn'
        n_header = 0
        if show_name:
            n_header += 1
        if show_unit:
            n_header += 1
        if show_dtype:
            n_header += 1
        if show_name or show_unit or show_dtype:
            n_header += 1
        kwargs = dict(max_lines=-1, show_name=show_name, show_unit=show_unit, show_dtype=show_dtype)
        if hasattr(tabcol, 'columns'):
            kwargs['max_width'] = max_width
        (max_lines1, max_width) = self._get_pprint_size(max_lines, max_width)
        if max_lines is None:
            max_lines1 += 2
        delta_lines = max_lines1 - n_header
        inkey = Getch()
        i0 = 0
        showlines = True
        while True:
            i1 = i0 + delta_lines
            if showlines:
                try:
                    os.system('cls' if os.name == 'nt' else 'clear')
                except Exception:
                    pass
                lines = tabcol[i0:i1].pformat(**kwargs)
                colors = ('red' if i < n_header else 'default' for i in range(len(lines)))
                for (color, line) in zip(colors, lines):
                    color_print(line, color)
            showlines = True
            print()
            print('-- f, <space>, b, r, p, n, <, >, q h (help) --', end=' ')
            while True:
                try:
                    key = inkey().lower()
                except Exception:
                    print('\n')
                    log.error('Console does not support getting a character as required by more().  Use pprint() instead.')
                    return
                if key in allowed_keys:
                    break
            print(key)
            if key.lower() == 'q':
                break
            if key == ' ' or key == 'f':
                i0 += delta_lines
            elif key == 'b':
                i0 = i0 - delta_lines
            elif key == 'r':
                pass
            elif key == '<':
                i0 = 0
            elif key == '>':
                i0 = len(tabcol)
            elif key == 'p':
                i0 -= 1
            elif key == 'n':
                i0 += 1
            elif key == 'h':
                showlines = False
                print('\n    Browsing keys:\n       f, <space> : forward one page\n       b : back one page\n       r : refresh same page\n       n : next row\n       p : previous row\n       < : go to beginning\n       > : go to end\n       q : quit browsing\n       h : print this help', end=' ')
            if i0 < 0:
                i0 = 0
            if i0 >= len(tabcol) - delta_lines:
                i0 = len(tabcol) - delta_lines
            print('\n')