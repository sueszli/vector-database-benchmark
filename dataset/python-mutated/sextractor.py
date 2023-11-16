"""sextractor.py:
  Classes to read SExtractor table format.

Built on daophot.py:
:Copyright: Smithsonian Astrophysical Observatory (2011)
:Author: Tom Aldcroft (aldcroft@head.cfa.harvard.edu)
"""
import re
from . import core

class SExtractorHeader(core.BaseHeader):
    """Read the header from a file produced by SExtractor."""
    comment = '^\\s*#\\s*\\S\\D.*'

    def get_cols(self, lines):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the header Column objects from the table ``lines`` for a SExtractor\n        header.  The SExtractor header is specialized so that we just copy the entire BaseHeader\n        get_cols routine and modify as needed.\n\n        Parameters\n        ----------\n        lines : list\n            List of table lines\n\n        '
        columns = {}
        re_name_def = re.compile('^\\s* \\# \\s*             # possible whitespace around #\n                (?P<colnumber> [0-9]+)\\s+   # number of the column in table\n                (?P<colname> [-\\w]+)        # name of the column\n                # column description, match any character until...\n                (?:\\s+(?P<coldescr> \\w .+)\n                # ...until [non-space][space][unit] or [not-right-bracket][end]\n                (?:(?<!(\\]))$|(?=(?:(?<=\\S)\\s+\\[.+\\]))))?\n                (?:\\s*\\[(?P<colunit>.+)\\])?.* # match units in brackets\n                ', re.VERBOSE)
        dataline = None
        for line in lines:
            if not line.startswith('#'):
                dataline = line
                break
            match = re_name_def.search(line)
            if match:
                colnumber = int(match.group('colnumber'))
                colname = match.group('colname')
                coldescr = match.group('coldescr')
                colunit = match.group('colunit')
                columns[colnumber] = (colname, coldescr, colunit)
        colnumbers = sorted(columns)
        if dataline is not None:
            n_data_cols = len(dataline.split())
        else:
            n_data_cols = colnumbers[-1]
        columns[n_data_cols + 1] = (None, None, None)
        colnumbers.append(n_data_cols + 1)
        if len(columns) > 1:
            previous_column = 0
            for n in colnumbers:
                if n != previous_column + 1:
                    for c in range(previous_column + 1, n):
                        column_name = columns[previous_column][0] + f'_{c - previous_column}'
                        column_descr = columns[previous_column][1]
                        column_unit = columns[previous_column][2]
                        columns[c] = (column_name, column_descr, column_unit)
                previous_column = n
        colnumbers = sorted(columns)[:-1]
        self.names = []
        for n in colnumbers:
            self.names.append(columns[n][0])
        if not self.names:
            raise core.InconsistentTableError('No column names found in SExtractor header')
        self.cols = []
        for n in colnumbers:
            col = core.Column(name=columns[n][0])
            col.description = columns[n][1]
            col.unit = columns[n][2]
            self.cols.append(col)

class SExtractorData(core.BaseData):
    start_line = 0
    delimiter = ' '
    comment = '\\s*#'

class SExtractor(core.BaseReader):
    """SExtractor format table.

    SExtractor is a package for faint-galaxy photometry (Bertin & Arnouts
    1996, A&A Supp. 317, 393.)

    See: https://sextractor.readthedocs.io/en/latest/

    Example::

      # 1 NUMBER
      # 2 ALPHA_J2000
      # 3 DELTA_J2000
      # 4 FLUX_RADIUS
      # 7 MAG_AUTO [mag]
      # 8 X2_IMAGE Variance along x [pixel**2]
      # 9 X_MAMA Barycenter position along MAMA x axis [m**(-6)]
      # 10 MU_MAX Peak surface brightness above background [mag * arcsec**(-2)]
      1 32.23222 10.1211 0.8 1.2 1.4 18.1 1000.0 0.00304 -3.498
      2 38.12321 -88.1321 2.2 2.4 3.1 17.0 1500.0 0.00908 1.401

    Note the skipped numbers since flux_radius has 3 columns.  The three
    FLUX_RADIUS columns will be named FLUX_RADIUS, FLUX_RADIUS_1, FLUX_RADIUS_2
    Also note that a post-ID description (e.g. "Variance along x") is optional
    and that units may be specified at the end of a line in brackets.

    """
    _format_name = 'sextractor'
    _io_registry_can_write = False
    _description = 'SExtractor format table'
    header_class = SExtractorHeader
    data_class = SExtractorData
    inputter_class = core.ContinuationLinesInputter

    def read(self, table):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read input data (file-like object, filename, list of strings, or\n        single string) into a Table and return the result.\n        '
        out = super().read(table)
        if 'comments' in out.meta:
            del out.meta['comments']
        return out

    def write(self, table):
        if False:
            return 10
        raise NotImplementedError