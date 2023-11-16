"""
An extensible ASCII table reader and writer.

Classes to read DAOphot table format

:Copyright: Smithsonian Astrophysical Observatory (2011)
:Author: Tom Aldcroft (aldcroft@head.cfa.harvard.edu)
"""
import itertools as itt
import re
from collections import OrderedDict, defaultdict
import numpy as np
from . import core, fixedwidth
from .misc import first_false_index, first_true_index, groupmore

class DaophotHeader(core.BaseHeader):
    """
    Read the header from a file produced by the IRAF DAOphot routine.
    """
    comment = '\\s*#K'
    re_format = re.compile('%-?(\\d+)\\.?\\d?[sdfg]')
    re_header_keyword = re.compile('[#]K\\s+ (?P<name> \\w+)\\s* = (?P<stuff> .+) $', re.VERBOSE)
    aperture_values = ()

    def __init__(self):
        if False:
            return 10
        core.BaseHeader.__init__(self)

    def parse_col_defs(self, grouped_lines_dict):
        if False:
            i = 10
            return i + 15
        'Parse a series of column definition lines.\n\n        Examples\n        --------\n        When parsing, there may be several such blocks in a single file\n        (where continuation characters have already been stripped).\n        #N ID    XCENTER   YCENTER   MAG         MERR          MSKY           NITER\n        #U ##    pixels    pixels    magnitudes  magnitudes    counts         ##\n        #F %-9d  %-10.3f   %-10.3f   %-12.3f     %-14.3f       %-15.7g        %-6d\n        '
        line_ids = ('#N', '#U', '#F')
        coldef_dict = defaultdict(list)
        stripper = lambda s: s[2:].strip(' \\')
        for defblock in zip(*map(grouped_lines_dict.get, line_ids)):
            for (key, line) in zip(line_ids, map(stripper, defblock)):
                coldef_dict[key].append(line.split())
        if self.data.is_multiline:
            (last_names, last_units, last_formats) = list(zip(*map(coldef_dict.get, line_ids)))[-1]
            N_multiline = len(self.data.first_block)
            for i in np.arange(1, N_multiline + 1).astype('U2'):
                extended_names = list(map(''.join, zip(last_names, itt.repeat(i))))
                if i == '1':
                    coldef_dict['#N'][-1] = extended_names
                else:
                    coldef_dict['#N'].append(extended_names)
                    coldef_dict['#U'].append(last_units)
                    coldef_dict['#F'].append(last_formats)
        get_col_width = lambda s: int(self.re_format.search(s).groups()[0])
        col_widths = [[get_col_width(f) for f in formats] for formats in coldef_dict['#F']]
        row_widths = np.fromiter(map(sum, col_widths), int)
        row_short = Daophot.table_width - row_widths
        for (w, r) in zip(col_widths, row_short):
            w[-1] += r
        self.col_widths = col_widths
        coldef_dict = {k: sum(v, []) for (k, v) in coldef_dict.items()}
        return coldef_dict

    def update_meta(self, lines, meta):
        if False:
            return 10
        "\n        Extract table-level keywords for DAOphot table.  These are indicated by\n        a leading '#K ' prefix.\n        "
        table_meta = meta['table']
        Nlines = len(self.lines)
        if Nlines > 0:
            get_line_id = lambda s: s.split(None, 1)[0]
            (gid, groups) = zip(*groupmore(get_line_id, self.lines, range(Nlines)))
            (grouped_lines, gix) = zip(*groups)
            grouped_lines_dict = dict(zip(gid, grouped_lines))
            if '#K' in grouped_lines_dict:
                keywords = OrderedDict(map(self.extract_keyword_line, grouped_lines_dict['#K']))
                table_meta['keywords'] = keywords
            coldef_dict = self.parse_col_defs(grouped_lines_dict)
            line_ids = ('#N', '#U', '#F')
            for (name, unit, fmt) in zip(*map(coldef_dict.get, line_ids)):
                meta['cols'][name] = {'unit': unit, 'format': fmt}
            self.meta = meta
            self.names = coldef_dict['#N']

    def extract_keyword_line(self, line):
        if False:
            return 10
        '\n        Extract info from a header keyword line (#K).\n        '
        m = self.re_header_keyword.match(line)
        if m:
            vals = m.group('stuff').strip().rsplit(None, 2)
            keyword_dict = {'units': vals[-2], 'format': vals[-1], 'value': vals[0] if len(vals) > 2 else ''}
            return (m.group('name'), keyword_dict)

    def get_cols(self, lines):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the header Column objects from the table ``lines`` for a DAOphot\n        header.  The DAOphot header is specialized so that we just copy the entire BaseHeader\n        get_cols routine and modify as needed.\n\n        Parameters\n        ----------\n        lines : list\n            List of table lines\n\n        Returns\n        -------\n        col : list\n            List of table Columns\n        '
        if not self.names:
            raise core.InconsistentTableError('No column names found in DAOphot header')
        self._set_cols_from_names()
        coldefs = self.meta['cols']
        for col in self.cols:
            (unit, fmt) = map(coldefs[col.name].get, ('unit', 'format'))
            if unit != '##':
                col.unit = unit
            if fmt != '##':
                col.format = fmt
        col_width = sum(self.col_widths, [])
        ends = np.cumsum(col_width)
        starts = ends - col_width
        for (i, col) in enumerate(self.cols):
            (col.start, col.end) = (starts[i], ends[i])
            col.span = col.end - col.start
            if hasattr(col, 'format'):
                if any((x in col.format for x in 'fg')):
                    col.type = core.FloatType
                elif 'd' in col.format:
                    col.type = core.IntType
                elif 's' in col.format:
                    col.type = core.StrType
        self.data.fill_values.append(('INDEF', '0'))

class DaophotData(core.BaseData):
    splitter_class = fixedwidth.FixedWidthSplitter
    start_line = 0
    comment = '\\s*#'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        core.BaseData.__init__(self)
        self.is_multiline = False

    def get_data_lines(self, lines):
        if False:
            print('Hello World!')
        if self.is_multiline:
            aplist = next(zip(*map(str.split, self.first_block)))
            self.header.aperture_values = tuple(map(float, aplist))
        core.BaseData.get_data_lines(self, lines)

class DaophotInputter(core.ContinuationLinesInputter):
    continuation_char = '\\'
    multiline_char = '*'
    replace_char = ' '
    re_multiline = re.compile('(#?)[^\\\\*#]*(\\*?)(\\\\*) ?$')

    def search_multiline(self, lines, depth=150):
        if False:
            for i in range(10):
                print('nop')
        '\n        Search lines for special continuation character to determine number of\n        continued rows in a datablock.  For efficiency, depth gives the upper\n        limit of lines to search.\n        '
        (comment, special, cont) = zip(*(self.re_multiline.search(line).groups() for line in lines[:depth]))
        data_start = first_false_index(comment)
        if data_start is None:
            return (None, None, lines[:depth])
        header_lines = lines[:data_start]
        first_special = first_true_index(special[data_start:depth])
        if first_special is None:
            return (None, None, header_lines)
        last_special = first_false_index(special[data_start + first_special:depth])
        markers = np.cumsum([data_start, first_special, last_special])
        multiline_block = lines[markers[1]:markers[-1]]
        return (markers, multiline_block, header_lines)

    def process_lines(self, lines):
        if False:
            return 10
        (markers, block, header) = self.search_multiline(lines)
        self.data.is_multiline = markers is not None
        self.data.markers = markers
        self.data.first_block = block
        self.data.header.lines = header
        if markers is not None:
            lines = lines[markers[0]:]
        continuation_char = self.continuation_char
        multiline_char = self.multiline_char
        replace_char = self.replace_char
        parts = []
        outlines = []
        for (i, line) in enumerate(lines):
            mo = self.re_multiline.search(line)
            if mo:
                (comment, special, cont) = mo.groups()
                if comment or cont:
                    line = line.replace(continuation_char, replace_char)
                if special:
                    line = line.replace(multiline_char, replace_char)
                if cont and (not comment):
                    parts.append(line)
                if not cont:
                    parts.append(line)
                    outlines.append(''.join(parts))
                    parts = []
            else:
                raise core.InconsistentTableError(f'multiline re could not match line {i}: {line}')
        return outlines

class Daophot(core.BaseReader):
    """
    DAOphot format table.

    Example::

      #K MERGERAD   = INDEF                   scaleunit  %-23.7g
      #K IRAF = NOAO/IRAFV2.10EXPORT version %-23s
      #K USER = davis name %-23s
      #K HOST = tucana computer %-23s
      #
      #N ID    XCENTER   YCENTER   MAG         MERR          MSKY           NITER    \\
      #U ##    pixels    pixels    magnitudes  magnitudes    counts         ##       \\
      #F %-9d  %-10.3f   %-10.3f   %-12.3f     %-14.3f       %-15.7g        %-6d
      #
      #N         SHARPNESS   CHI         PIER  PERROR                                \\
      #U         ##          ##          ##    perrors                               \\
      #F         %-23.3f     %-12.3f     %-6d  %-13s
      #
      14       138.538     INDEF   15.461      0.003         34.85955       4        \\
                  -0.032      0.802       0     No_error

    The keywords defined in the #K records are available via the output table
    ``meta`` attribute::

      >>> import os
      >>> from astropy.io import ascii
      >>> filename = os.path.join(ascii.__path__[0], 'tests/data/daophot.dat')
      >>> data = ascii.read(filename)
      >>> for name, keyword in data.meta['keywords'].items():
      ...     print(name, keyword['value'], keyword['units'], keyword['format'])
      ...
      MERGERAD INDEF scaleunit %-23.7g
      IRAF NOAO/IRAFV2.10EXPORT version %-23s
      USER  name %-23s
      ...

    The unit and formats are available in the output table columns::

      >>> for colname in data.colnames:
      ...     col = data[colname]
      ...     print(colname, col.unit, col.format)
      ...
      ID None %-9d
      XCENTER pixels %-10.3f
      YCENTER pixels %-10.3f
      ...

    Any column values of INDEF are interpreted as a missing value and will be
    masked out in the resultant table.

    In case of multi-aperture daophot files containing repeated entries for the last
    row of fields, extra unique column names will be created by suffixing
    corresponding field names with numbers starting from 2 to N (where N is the
    total number of apertures).
    For example,
    first aperture radius will be RAPERT and corresponding magnitude will be MAG,
    second aperture radius will be RAPERT2 and corresponding magnitude will be MAG2,
    third aperture radius will be RAPERT3 and corresponding magnitude will be MAG3,
    and so on.

    """
    _format_name = 'daophot'
    _io_registry_format_aliases = ['daophot']
    _io_registry_can_write = False
    _description = 'IRAF DAOphot format table'
    header_class = DaophotHeader
    data_class = DaophotData
    inputter_class = DaophotInputter
    table_width = 80

    def __init__(self):
        if False:
            return 10
        core.BaseReader.__init__(self)
        self.inputter.data = self.data

    def write(self, table=None):
        if False:
            print('Hello World!')
        raise NotImplementedError