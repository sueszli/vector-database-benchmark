"""Classes to read AAS MRT table format.

Ref: https://journals.aas.org/mrt-standards

:Copyright: Smithsonian Astrophysical Observatory (2021)
:Author: Tom Aldcroft (aldcroft@head.cfa.harvard.edu),          Suyog Garg (suyog7130@gmail.com)
"""
import re
import warnings
from io import StringIO
from math import ceil, floor
from string import Template
from textwrap import wrap
import numpy as np
from astropy import units as u
from astropy.table import Column, MaskedColumn, Table
from . import cds, core, fixedwidth
MAX_SIZE_README_LINE = 80
MAX_COL_INTLIMIT = 100000
__doctest_skip__ = ['*']
BYTE_BY_BYTE_TEMPLATE = ['Byte-by-byte Description of file: $file', '--------------------------------------------------------------------------------', ' Bytes Format Units  Label     Explanations', '--------------------------------------------------------------------------------', '$bytebybyte', '--------------------------------------------------------------------------------']
MRT_TEMPLATE = ['Title:', 'Authors:', 'Table:', '================================================================================', '$bytebybyte', 'Notes:', '--------------------------------------------------------------------------------']

class MrtSplitter(fixedwidth.FixedWidthSplitter):
    """
    Contains the join function to left align the MRT columns
    when writing to a file.
    """

    def join(self, vals, widths):
        if False:
            return 10
        vals = [val + ' ' * (width - len(val)) for (val, width) in zip(vals, widths)]
        return self.delimiter.join(vals)

class MrtHeader(cds.CdsHeader):
    _subfmt = 'MRT'

    def _split_float_format(self, value):
        if False:
            while True:
                i = 10
        '\n        Splits a Float string into different parts to find number\n        of digits after decimal and check if the value is in Scientific\n        notation.\n\n        Parameters\n        ----------\n        value : str\n            String containing the float value to split.\n\n        Returns\n        -------\n        fmt: (int, int, int, bool, bool)\n            List of values describing the Float string.\n            (size, dec, ent, sign, exp)\n            size, length of the given string.\n            ent, number of digits before decimal point.\n            dec, number of digits after decimal point.\n            sign, whether or not given value signed.\n            exp, is value in Scientific notation?\n        '
        regfloat = re.compile('(?P<sign> [+-]*)\n                (?P<ent> [^eE.]+)\n                (?P<deciPt> [.]*)\n                (?P<decimals> [0-9]*)\n                (?P<exp> [eE]*-*)[0-9]*', re.VERBOSE)
        mo = regfloat.match(value)
        if mo is None:
            raise Exception(f'{value} is not a float number')
        return (len(value), len(mo.group('ent')), len(mo.group('decimals')), mo.group('sign') != '', mo.group('exp') != '')

    def _set_column_val_limits(self, col):
        if False:
            while True:
                i = 10
        '\n        Sets the ``col.min`` and ``col.max`` column attributes,\n        taking into account columns with Null values.\n        '
        col.max = max(col)
        col.min = min(col)
        if col.max is np.ma.core.MaskedConstant:
            col.max = None
        if col.min is np.ma.core.MaskedConstant:
            col.min = None

    def column_float_formatter(self, col):
        if False:
            while True:
                i = 10
        '\n        String formatter function for a column containing Float values.\n        Checks if the values in the given column are in Scientific notation,\n        by splitting the value string. It is assumed that the column either has\n        float values or Scientific notation.\n\n        A ``col.formatted_width`` attribute is added to the column. It is not added\n        if such an attribute is already present, say when the ``formats`` argument\n        is passed to the writer. A properly formatted format string is also added as\n        the ``col.format`` attribute.\n\n        Parameters\n        ----------\n        col : A ``Table.Column`` object.\n        '
        (maxsize, maxprec, maxent, maxdec) = (1, 0, 1, 0)
        sign = False
        fformat = 'F'
        for val in col.str_vals:
            if val is None or val == '':
                continue
            fmt = self._split_float_format(val)
            if fmt[4] is True:
                if fformat == 'F':
                    (maxsize, maxprec, maxdec) = (1, 0, 0)
                fformat = 'E'
            elif fformat == 'E':
                continue
            if maxsize < fmt[0]:
                maxsize = fmt[0]
            if maxent < fmt[1]:
                maxent = fmt[1]
            if maxdec < fmt[2]:
                maxdec = fmt[2]
            if fmt[3]:
                sign = True
            if maxprec < fmt[1] + fmt[2]:
                maxprec = fmt[1] + fmt[2]
        if fformat == 'E':
            if getattr(col, 'formatted_width', None) is None:
                col.formatted_width = maxsize
                if sign:
                    col.formatted_width += 1
            col.fortran_format = fformat + str(col.formatted_width) + '.' + str(maxprec)
            col.format = str(col.formatted_width) + '.' + str(maxdec) + 'e'
        else:
            lead = ''
            if getattr(col, 'formatted_width', None) is None:
                col.formatted_width = maxent + maxdec + 1
                if sign:
                    col.formatted_width += 1
            elif col.format.startswith('0'):
                lead = '0'
            col.fortran_format = fformat + str(col.formatted_width) + '.' + str(maxdec)
            col.format = lead + col.fortran_format[1:] + 'f'

    def write_byte_by_byte(self):
        if False:
            print('Hello World!')
        '\n        Writes the Byte-By-Byte description of the table.\n\n        Columns that are `astropy.coordinates.SkyCoord` or `astropy.time.TimeSeries`\n        objects or columns with values that are such objects are recognized as such,\n        and some predefined labels and description is used for them.\n        See the Vizier MRT Standard documentation in the link below for more details\n        on these. An example Byte-By-Byte table is shown here.\n\n        See: https://vizier.unistra.fr/doc/catstd-3.1.htx\n\n        Example::\n\n        --------------------------------------------------------------------------------\n        Byte-by-byte Description of file: table.dat\n        --------------------------------------------------------------------------------\n        Bytes Format Units  Label     Explanations\n        --------------------------------------------------------------------------------\n         1- 8  A8     ---    names   Description of names\n        10-14  E5.1   ---    e       [-3160000.0/0.01] Description of e\n        16-23  F8.5   ---    d       [22.25/27.25] Description of d\n        25-31  E7.1   ---    s       [-9e+34/2.0] Description of s\n        33-35  I3     ---    i       [-30/67] Description of i\n        37-39  F3.1   ---    sameF   [5.0/5.0] Description of sameF\n        41-42  I2     ---    sameI   [20] Description of sameI\n        44-45  I2     h      RAh     Right Ascension (hour)\n        47-48  I2     min    RAm     Right Ascension (minute)\n        50-67  F18.15 s      RAs     Right Ascension (second)\n           69  A1     ---    DE-     Sign of Declination\n        70-71  I2     deg    DEd     Declination (degree)\n        73-74  I2     arcmin DEm     Declination (arcmin)\n        76-91  F16.13 arcsec DEs     Declination (arcsec)\n\n        --------------------------------------------------------------------------------\n        '
        vals_list = list(zip(*self.data.str_vals()))
        for (i, col) in enumerate(self.cols):
            col.width = max((len(vals[i]) for vals in vals_list))
            if self.start_line is not None:
                col.width = max(col.width, len(col.info.name))
        widths = [col.width for col in self.cols]
        startb = 1
        byte_count_width = len(str(sum(widths) + len(self.cols) - 1))
        singlebfmt = '{:' + str(byte_count_width) + 'd}'
        fmtb = singlebfmt + '-' + singlebfmt
        singlebfmt += ' '
        fmtb += ' '
        (max_label_width, max_descrip_size) = (7, 16)
        bbb = Table(names=['Bytes', 'Format', 'Units', 'Label', 'Explanations'], dtype=[str] * 5)
        for (i, col) in enumerate(self.cols):
            col.has_null = isinstance(col, MaskedColumn)
            if col.format is not None:
                col.formatted_width = max((len(sval) for sval in col.str_vals))
            if np.issubdtype(col.dtype, np.integer):
                self._set_column_val_limits(col)
                if getattr(col, 'formatted_width', None) is None:
                    col.formatted_width = max(len(str(col.max)), len(str(col.min)))
                col.fortran_format = 'I' + str(col.formatted_width)
                if col.format is None:
                    col.format = '>' + col.fortran_format[1:]
            elif np.issubdtype(col.dtype, np.dtype(float).type):
                self._set_column_val_limits(col)
                self.column_float_formatter(col)
            else:
                dtype = col.dtype.str
                if col.has_null:
                    mcol = col
                    mcol.fill_value = ''
                    coltmp = Column(mcol.filled(), dtype=str)
                    dtype = coltmp.dtype.str
                if getattr(col, 'formatted_width', None) is None:
                    col.formatted_width = int(re.search('(\\d+)$', dtype).group(1))
                col.fortran_format = 'A' + str(col.formatted_width)
                col.format = str(col.formatted_width) + 's'
            endb = col.formatted_width + startb - 1
            if col.name is None:
                col.name = 'Unknown'
            if col.description is not None:
                description = col.description
            else:
                description = 'Description of ' + col.name
            nullflag = ''
            if col.has_null:
                nullflag = '?'
            if col.unit is not None:
                col_unit = col.unit.to_string('cds')
            elif col.name.lower().find('magnitude') > -1:
                col_unit = 'mag'
            else:
                col_unit = '---'
            lim_vals = ''
            if col.min and col.max and (not any((x in col.name for x in ['RA', 'DE', 'LON', 'LAT', 'PLN', 'PLT']))):
                if col.fortran_format[0] == 'I':
                    if abs(col.min) < MAX_COL_INTLIMIT and abs(col.max) < MAX_COL_INTLIMIT:
                        if col.min == col.max:
                            lim_vals = f'[{col.min}]'
                        else:
                            lim_vals = f'[{col.min}/{col.max}]'
                elif col.fortran_format[0] in ('E', 'F'):
                    lim_vals = f'[{floor(col.min * 100) / 100.0}/{ceil(col.max * 100) / 100.0}]'
            if lim_vals != '' or nullflag != '':
                description = f'{lim_vals}{nullflag} {description}'
            if len(col.name) > max_label_width:
                max_label_width = len(col.name)
            if len(description) > max_descrip_size:
                max_descrip_size = len(description)
            if col.name == 'DEd':
                bbb.add_row([singlebfmt.format(startb), 'A1', '---', 'DE-', 'Sign of Declination'])
                col.fortran_format = 'I2'
                startb += 1
            bbb.add_row([singlebfmt.format(startb) if startb == endb else fmtb.format(startb, endb), '' if col.fortran_format is None else col.fortran_format, col_unit, '' if col.name is None else col.name, description])
            startb = endb + 2
        bbblines = StringIO()
        bbb.write(bbblines, format='ascii.fixed_width_no_header', delimiter=' ', bookend=False, delimiter_pad=None, formats={'Format': '<6s', 'Units': '<6s', 'Label': '<' + str(max_label_width) + 's', 'Explanations': '' + str(max_descrip_size) + 's'})
        bbblines = bbblines.getvalue().splitlines()
        nsplit = byte_count_width * 2 + 1 + 12 + max_label_width + 4
        buff = ''
        for newline in bbblines:
            if len(newline) > MAX_SIZE_README_LINE:
                buff += '\n'.join(wrap(newline, subsequent_indent=' ' * nsplit, width=MAX_SIZE_README_LINE))
                buff += '\n'
            else:
                buff += newline + '\n'
        self.linewidth = endb
        buff = buff[:-1]
        return buff

    def write(self, lines):
        if False:
            return 10
        '\n        Writes the Header of the MRT table, aka ReadMe, which\n        also contains the Byte-By-Byte description of the table.\n        '
        from astropy.coordinates import SkyCoord
        coord_systems = {'galactic': ('GLAT', 'GLON', 'b', 'l'), 'ecliptic': ('ELAT', 'ELON', 'lat', 'lon'), 'heliographic': ('HLAT', 'HLON', 'lat', 'lon'), 'helioprojective': ('HPLT', 'HPLN', 'Ty', 'Tx')}
        eqtnames = ['RAh', 'RAm', 'RAs', 'DEd', 'DEm', 'DEs']
        to_pop = []
        for (i, col) in enumerate(self.cols):
            if not isinstance(col, SkyCoord) and isinstance(col[0], SkyCoord):
                try:
                    col = SkyCoord(col)
                except (ValueError, TypeError):
                    if not isinstance(col, Column):
                        col = Column(col)
                    col = Column([str(val) for val in col])
                    self.cols[i] = col
                    continue
            if isinstance(col, SkyCoord):
                if 'ra' in col.representation_component_names.keys() and len(set(eqtnames) - set(self.colnames)) == 6:
                    (ra_c, dec_c) = (col.ra.hms, col.dec.dms)
                    coords = [ra_c.h.round().astype('i1'), ra_c.m.round().astype('i1'), ra_c.s, dec_c.d.round().astype('i1'), dec_c.m.round().astype('i1'), dec_c.s]
                    coord_units = [u.h, u.min, u.second, u.deg, u.arcmin, u.arcsec]
                    coord_descrip = ['Right Ascension (hour)', 'Right Ascension (minute)', 'Right Ascension (second)', 'Declination (degree)', 'Declination (arcmin)', 'Declination (arcsec)']
                    for (coord, name, coord_unit, descrip) in zip(coords, eqtnames, coord_units, coord_descrip):
                        if name in ['DEm', 'DEs']:
                            coord_col = Column(list(np.abs(coord)), name=name, unit=coord_unit, description=descrip)
                        else:
                            coord_col = Column(list(coord), name=name, unit=coord_unit, description=descrip)
                        if name == 'RAs':
                            coord_col.format = '013.10f'
                        elif name == 'DEs':
                            coord_col.format = '012.9f'
                        elif name == 'RAh':
                            coord_col.format = '2d'
                        elif name == 'DEd':
                            coord_col.format = '+03d'
                        elif name.startswith(('RA', 'DE')):
                            coord_col.format = '02d'
                        self.cols.append(coord_col)
                    to_pop.append(i)
                else:
                    frminfo = ''
                    for (frame, latlon) in coord_systems.items():
                        if frame in col.name and len(set(latlon[:2]) - set(self.colnames)) == 2:
                            if frame != col.name:
                                frminfo = f' ({col.name})'
                            lon_col = Column(getattr(col, latlon[3]), name=latlon[1], description=f'{frame.capitalize()} Longitude{frminfo}', unit=col.representation_component_units[latlon[3]], format='.12f')
                            lat_col = Column(getattr(col, latlon[2]), name=latlon[0], description=f'{frame.capitalize()} Latitude{frminfo}', unit=col.representation_component_units[latlon[2]], format='+.12f')
                            self.cols.append(lon_col)
                            self.cols.append(lat_col)
                            to_pop.append(i)
                if i not in to_pop:
                    warnings.warn(f"Coordinate system of type '{col.name}' already stored in table as CDS/MRT-syle columns or of unrecognized type. So column {i} is being skipped with designation of a string valued column `{self.colnames[i]}`.", UserWarning)
                    self.cols.append(Column(col.to_string(), name=self.colnames[i]))
                    to_pop.append(i)
            elif not isinstance(col, Column):
                col = Column(col)
                if np.issubdtype(col.dtype, np.dtype(object).type):
                    col = Column([str(val) for val in col])
                self.cols[i] = col
        for i in to_pop[::-1]:
            self.cols.pop(i)
        if any((x in self.colnames for x in ['RAh', 'DEd', 'ELON', 'GLAT'])):
            for (i, col) in enumerate(self.cols):
                if isinstance(col, SkyCoord):
                    self.cols[i] = Column(col.to_string(), name=self.colnames[i])
                    message = f'Table already has coordinate system in CDS/MRT-syle columns. So column {i} should have been replaced already with a string valued column `{self.colnames[i]}`.'
                    raise core.InconsistentTableError(message)
        bbb_template = Template('\n'.join(BYTE_BY_BYTE_TEMPLATE))
        byte_by_byte = bbb_template.substitute({'file': 'table.dat', 'bytebybyte': self.write_byte_by_byte()})
        rm_template = Template('\n'.join(MRT_TEMPLATE))
        readme_filled = rm_template.substitute({'bytebybyte': byte_by_byte})
        lines.append(readme_filled)

class MrtData(cds.CdsData):
    """MRT table data reader."""
    _subfmt = 'MRT'
    splitter_class = MrtSplitter

    def write(self, lines):
        if False:
            for i in range(10):
                print('nop')
        self.splitter.delimiter = ' '
        fixedwidth.FixedWidthData.write(self, lines)

class Mrt(core.BaseReader):
    """AAS MRT (Machine-Readable Table) format table.

    **Reading**
    ::

      >>> from astropy.io import ascii
      >>> table = ascii.read('data.mrt', format='mrt')

    **Writing**

    Use ``ascii.write(table, 'data.mrt', format='mrt')`` to  write tables to
    Machine Readable Table (MRT) format.

    Note that the metadata of the table, apart from units, column names and
    description, will not be written. These have to be filled in by hand later.

    See also: :ref:`cds_mrt_format`.

    Caveats:

    * The Units and Explanations are available in the column ``unit`` and
      ``description`` attributes, respectively.
    * The other metadata defined by this format is not available in the output table.
    """
    _format_name = 'mrt'
    _io_registry_format_aliases = ['mrt']
    _io_registry_can_write = True
    _description = 'MRT format table'
    data_class = MrtData
    header_class = MrtHeader

    def write(self, table=None):
        if False:
            while True:
                i = 10
        if len(table) == 0:
            raise NotImplementedError
        self.data.header = self.header
        self.header.position_line = None
        self.header.start_line = None
        table = table.copy()
        return super().write(table)