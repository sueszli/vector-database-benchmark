import re
import warnings
from collections import OrderedDict, defaultdict
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.table import Column, MaskedColumn
from astropy.table.column import col_copy
from astropy.time import Time, TimeDelta
from astropy.time.core import BARYCENTRIC_SCALES
from astropy.time.formats import FITS_DEPRECATED_SCALES
from astropy.utils.exceptions import AstropyUserWarning
from . import Card, Header
TCTYP_RE_TYPE = re.compile('(?P<type>[A-Z]+)[-]+')
TCTYP_RE_ALGO = re.compile('(?P<algo>[A-Z]+)\\s*')
FITS_TIME_UNIT = ['s', 'd', 'a', 'cy', 'min', 'h', 'yr', 'ta', 'Ba']
OBSGEO_XYZ = ('OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z')
OBSGEO_LBH = ('OBSGEO-L', 'OBSGEO-B', 'OBSGEO-H')
TIME_KEYWORDS = ('DATE', 'DATE-AVG', 'DATE-BEG', 'DATE-END', 'DATE-OBS', 'DATEREF', 'JDREF', 'MJD-AVG', 'MJD-BEG', 'MJD-END', 'MJD-OBS', 'MJDREF', 'TIMEOFFS', 'TIMESYS', 'TIMEUNIT', 'TREFDIR', 'TREFPOS') + OBSGEO_LBH + OBSGEO_XYZ
COLUMN_TIME_KEYWORDS = ('TCTYP', 'TCUNI', 'TRPOS')
COLUMN_TIME_KEYWORD_REGEXP = f"({'|'.join(COLUMN_TIME_KEYWORDS)})[0-9]+"

def is_time_column_keyword(keyword):
    if False:
        return 10
    '\n    Check if the FITS header keyword is a time column-specific keyword.\n\n    Parameters\n    ----------\n    keyword : str\n        FITS keyword.\n    '
    return re.match(COLUMN_TIME_KEYWORD_REGEXP, keyword) is not None
GLOBAL_TIME_INFO = {'TIMESYS': ('UTC', 'Default time scale'), 'JDREF': (0.0, 'Time columns are jd = jd1 + jd2'), 'TREFPOS': ('TOPOCENTER', 'Time reference position')}

def _verify_global_info(global_info):
    if False:
        while True:
            i = 10
    '\n    Given the global time reference frame information, verify that\n    each global time coordinate attribute will be given a valid value.\n\n    Parameters\n    ----------\n    global_info : dict\n        Global time reference frame information.\n    '
    global_info['scale'] = FITS_DEPRECATED_SCALES.get(global_info['TIMESYS'], global_info['TIMESYS'].lower())
    if global_info['scale'] not in Time.SCALES:
        if global_info['scale'] == 'gps':
            warnings.warn('Global time scale (TIMESYS) has a FITS recognized time scale value "GPS". In Astropy, "GPS" is a time from epoch format which runs synchronously with TAI; GPS is approximately 19 s ahead of TAI. Hence, this format will be used.', AstropyUserWarning)
            global_info['scale'] = 'tai'
            global_info['format'] = 'gps'
        if global_info['scale'] == 'local':
            warnings.warn('Global time scale (TIMESYS) has a FITS recognized time scale value "LOCAL". However, the standard states that "LOCAL" should be tied to one of the existing scales because it is intrinsically unreliable and/or ill-defined. Astropy will thus use the default global time scale "UTC" instead of "LOCAL".', AstropyUserWarning)
            global_info['scale'] = 'utc'
            global_info['format'] = None
        else:
            raise AssertionError('Global time scale (TIMESYS) should have a FITS recognized time scale value (got {!r}). The FITS standard states that the use of local time scales should be restricted to alternate coordinates.'.format(global_info['TIMESYS']))
    else:
        global_info['format'] = None
    obs_geo = [global_info[attr] for attr in OBSGEO_XYZ if attr in global_info]
    if len(obs_geo) == 3:
        global_info['location'] = EarthLocation.from_geocentric(*obs_geo, unit=u.m)
    else:
        if obs_geo:
            warnings.warn(f'The geocentric observatory location {obs_geo} is not completely specified (X, Y, Z) and will be ignored.', AstropyUserWarning)
        obs_geo = [global_info[attr] for attr in OBSGEO_LBH if attr in global_info]
        if len(obs_geo) == 3:
            global_info['location'] = EarthLocation.from_geodetic(*obs_geo)
        else:
            if obs_geo:
                warnings.warn(f'The geodetic observatory location {obs_geo} is not completely specified (lon, lat, alt) and will be ignored.', AstropyUserWarning)
            global_info['location'] = None
    for (key, format_) in (('MJDREF', 'mjd'), ('JDREF', 'jd'), ('DATEREF', 'fits')):
        if key in global_info:
            global_info['ref_time'] = {'val': global_info[key], 'format': format_}
            break
    else:
        global_info['ref_time'] = {'val': 0, 'format': 'mjd'}

def _verify_column_info(column_info, global_info):
    if False:
        while True:
            i = 10
    '\n    Given the column-specific time reference frame information, verify that\n    each column-specific time coordinate attribute has a valid value.\n    Return True if the coordinate column is time, or else return False.\n\n    Parameters\n    ----------\n    global_info : dict\n        Global time reference frame information.\n    column_info : dict\n        Column-specific time reference frame override information.\n    '
    scale = column_info.get('TCTYP', None)
    unit = column_info.get('TCUNI', None)
    location = column_info.get('TRPOS', None)
    if scale is not None:
        if TCTYP_RE_TYPE.match(scale[:5]) and TCTYP_RE_ALGO.match(scale[5:]):
            return False
        elif scale.lower() in Time.SCALES:
            column_info['scale'] = scale.lower()
            column_info['format'] = None
        elif scale in FITS_DEPRECATED_SCALES.keys():
            column_info['scale'] = FITS_DEPRECATED_SCALES[scale]
            column_info['format'] = None
        elif scale == 'TIME':
            column_info['scale'] = global_info['scale']
            column_info['format'] = global_info['format']
        elif scale == 'GPS':
            warnings.warn('Table column "{}" has a FITS recognized time scale value "GPS". In Astropy, "GPS" is a time from epoch format which runs synchronously with TAI; GPS runs ahead of TAI approximately by 19 s. Hence, this format will be used.'.format(column_info), AstropyUserWarning)
            column_info['scale'] = 'tai'
            column_info['format'] = 'gps'
        elif scale == 'LOCAL':
            warnings.warn('Table column "{}" has a FITS recognized time scale value "LOCAL". However, the standard states that "LOCAL" should be tied to one of the existing scales because it is intrinsically unreliable and/or ill-defined. Astropy will thus use the global time scale (TIMESYS) as the default.'.format(column_info), AstropyUserWarning)
            column_info['scale'] = global_info['scale']
            column_info['format'] = global_info['format']
        else:
            return False
    elif unit is not None and unit in FITS_TIME_UNIT or location is not None:
        column_info['scale'] = global_info['scale']
        column_info['format'] = global_info['format']
    else:
        return False
    if location is not None:
        if location == 'TOPOCENTER':
            column_info['location'] = global_info['location']
            if column_info['location'] is None:
                warnings.warn('Time column reference position "TRPOSn" value is "TOPOCENTER". However, the observatory position is not properly specified. The FITS standard does not support this and hence reference position will be ignored.', AstropyUserWarning)
        else:
            column_info['location'] = None
    elif global_info['TREFPOS'] == 'TOPOCENTER':
        if global_info['location'] is not None:
            warnings.warn('Time column reference position "TRPOSn" is not specified. The default value for it is "TOPOCENTER", and the observatory position has been specified. However, for supporting column-specific location, reference position will be ignored for this column.', AstropyUserWarning)
        column_info['location'] = None
    else:
        column_info['location'] = None
    column_info['ref_time'] = global_info['ref_time']
    return True

def _get_info_if_time_column(col, global_info):
    if False:
        while True:
            i = 10
    "\n    Check if a column without corresponding time column keywords in the\n    FITS header represents time or not. If yes, return the time column\n    information needed for its conversion to Time.\n    This is only applicable to the special-case where a column has the\n    name 'TIME' and a time unit.\n    "
    if col.info.name.upper() == 'TIME' and col.info.unit in FITS_TIME_UNIT:
        column_info = {'scale': global_info['scale'], 'format': global_info['format'], 'ref_time': global_info['ref_time'], 'location': None}
        if global_info['TREFPOS'] == 'TOPOCENTER':
            column_info['location'] = global_info['location']
            if column_info['location'] is None:
                warnings.warn(f'Time column "{col.info.name}" reference position will be ignored due to unspecified observatory position.', AstropyUserWarning)
        return column_info
    return None

def _convert_global_time(table, global_info):
    if False:
        while True:
            i = 10
    '\n    Convert the table metadata for time informational keywords\n    to astropy Time.\n\n    Parameters\n    ----------\n    table : `~astropy.table.Table`\n        The table whose time metadata is to be converted.\n    global_info : dict\n        Global time reference frame information.\n    '
    for key in global_info:
        if key not in table.meta:
            try:
                table.meta[key] = _convert_time_key(global_info, key)
            except ValueError:
                pass

def _convert_time_key(global_info, key):
    if False:
        while True:
            i = 10
    '\n    Convert a time metadata key to a Time object.\n\n    Parameters\n    ----------\n    global_info : dict\n        Global time reference frame information.\n    key : str\n        Time key.\n\n    Returns\n    -------\n    astropy.time.Time\n\n    Raises\n    ------\n    ValueError\n        If key is not a valid global time keyword.\n    '
    value = global_info[key]
    if key.startswith('DATE'):
        scale = 'utc' if key == 'DATE' else global_info['scale']
        precision = len(value.split('.')[-1]) if '.' in value else 0
        return Time(value, format='fits', scale=scale, precision=precision)
    elif key.startswith('MJD-'):
        return Time(value, format='mjd', scale=global_info['scale'])
    else:
        raise ValueError('Key is not a valid global time keyword')

def _convert_time_column(col, column_info):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert time columns to astropy Time columns.\n\n    Parameters\n    ----------\n    col : `~astropy.table.Column`\n        The time coordinate column to be converted to Time.\n    column_info : dict\n        Column-specific time reference frame override information.\n    '
    try:
        if col.info.dtype.kind in ['S', 'U']:
            precision = max(int(col.info.dtype.str[2:]) - 20, 0)
            return Time(col, format='fits', scale=column_info['scale'], precision=precision, location=column_info['location'])
        if column_info['format'] == 'gps':
            return Time(col, format='gps', location=column_info['location'])
        if column_info['ref_time']['val'] == 0 and column_info['ref_time']['format'] in ['jd', 'mjd']:
            if col.shape[-1] == 2 and col.ndim > 1:
                return Time(col[..., 0], col[..., 1], scale=column_info['scale'], format=column_info['ref_time']['format'], location=column_info['location'])
            else:
                return Time(col, scale=column_info['scale'], format=column_info['ref_time']['format'], location=column_info['location'])
        ref_time = Time(column_info['ref_time']['val'], scale=column_info['scale'], format=column_info['ref_time']['format'], location=column_info['location'])
        if col.shape[-1] == 2 and col.ndim > 1:
            delta_time = TimeDelta(col[..., 0], col[..., 1])
        else:
            delta_time = TimeDelta(col)
        return ref_time + delta_time
    except Exception as err:
        warnings.warn(f'The exception "{err}" was encountered while trying to convert the time column "{col.info.name}" to Astropy Time.', AstropyUserWarning)
        return col

def fits_to_time(hdr, table):
    if False:
        for i in range(10):
            print('nop')
    '\n    Read FITS binary table time columns as `~astropy.time.Time`.\n\n    This method reads the metadata associated with time coordinates, as\n    stored in a FITS binary table header, converts time columns into\n    `~astropy.time.Time` columns and reads global reference times as\n    `~astropy.time.Time` instances.\n\n    Parameters\n    ----------\n    hdr : `~astropy.io.fits.header.Header`\n        FITS Header\n    table : `~astropy.table.Table`\n        The table whose time columns are to be read as Time\n\n    Returns\n    -------\n    hdr : `~astropy.io.fits.header.Header`\n        Modified FITS Header (time metadata removed)\n    '
    global_info = {'TIMESYS': 'UTC', 'TREFPOS': 'TOPOCENTER'}
    time_columns = defaultdict(OrderedDict)
    hcopy = hdr.copy(strip=True)
    for (key, value, comment) in hdr.cards:
        if key in TIME_KEYWORDS:
            global_info[key] = value
            hcopy.remove(key)
        elif is_time_column_keyword(key):
            (base, idx) = re.match('([A-Z]+)([0-9]+)', key).groups()
            time_columns[int(idx)][base] = value
            hcopy.remove(key)
        elif value in OBSGEO_XYZ and re.match('TTYPE[0-9]+', key):
            global_info[value] = table[value]
    _verify_global_info(global_info)
    _convert_global_time(table, global_info)
    if time_columns:
        for (idx, column_info) in time_columns.items():
            if _verify_column_info(column_info, global_info):
                colname = table.colnames[idx - 1]
                table[colname] = _convert_time_column(table[colname], column_info)
    for (idx, colname) in enumerate(table.colnames):
        if idx + 1 not in time_columns:
            column_info = _get_info_if_time_column(table[colname], global_info)
            if column_info:
                table[colname] = _convert_time_column(table[colname], column_info)
    return hcopy

def time_to_fits(table):
    if False:
        while True:
            i = 10
    '\n    Replace Time columns in a Table with non-mixin columns containing\n    each element as a vector of two doubles (jd1, jd2) and return a FITS\n    header with appropriate time coordinate keywords.\n    jd = jd1 + jd2 represents time in the Julian Date format with\n    high-precision.\n\n    Parameters\n    ----------\n    table : `~astropy.table.Table`\n        The table whose Time columns are to be replaced.\n\n    Returns\n    -------\n    table : `~astropy.table.Table`\n        The table with replaced Time columns\n    hdr : `~astropy.io.fits.header.Header`\n        Header containing global time reference frame FITS keywords\n    '
    new_cols = []
    for col in table.itercols():
        if isinstance(col, Column):
            new_col = col.copy(copy_data=False)
        else:
            new_col = col_copy(col, copy_indices=False) if col.info.indices else col
        new_cols.append(new_col)
    newtable = table.__class__(new_cols, copy=False)
    newtable.meta = table.meta
    hdr = Header([Card(keyword=key, value=val[0], comment=val[1]) for (key, val) in GLOBAL_TIME_INFO.items()])
    newtable.meta['__coordinate_columns__'] = defaultdict(OrderedDict)
    coord_meta = newtable.meta['__coordinate_columns__']
    time_cols = table.columns.isinstance(Time)
    location = None
    for col in time_cols:
        col_cls = MaskedColumn if col.masked else Column
        if col.info.serialize_method['fits'] == 'formatted_value':
            newtable.replace_column(col.info.name, col_cls(col.value))
            continue
        jd12 = np.empty_like(col.jd1, shape=col.jd1.shape + (2,))
        jd12[..., 0] = col.jd1
        jd12[..., 1] = col.jd2
        newtable.replace_column(col.info.name, col_cls(jd12, unit='d'))
        coord_meta[col.info.name]['coord_type'] = col.scale.upper()
        coord_meta[col.info.name]['coord_unit'] = 'd'
        if col.location is None:
            coord_meta[col.info.name]['time_ref_pos'] = None
            if location is not None:
                warnings.warn('Time Column "{}" has no specified location, but global Time Position is present, which will be the default for this column in FITS specification.'.format(col.info.name), AstropyUserWarning)
        else:
            coord_meta[col.info.name]['time_ref_pos'] = 'TOPOCENTER'
            if col.scale in BARYCENTRIC_SCALES:
                warnings.warn('Earth Location "TOPOCENTER" for Time Column "{}" is incompatible with scale "{}".'.format(col.info.name, col.scale.upper()), AstropyUserWarning)
            if location is None:
                location = col.location
                if location.size > 1:
                    for dim in ('x', 'y', 'z'):
                        newtable.add_column(Column(getattr(location, dim).to_value(u.m)), name=f'OBSGEO-{dim.upper()}')
                else:
                    hdr.extend([Card(keyword=f'OBSGEO-{dim.upper()}', value=getattr(location, dim).to_value(u.m)) for dim in ('x', 'y', 'z')])
            elif np.any(location != col.location):
                raise ValueError(f'Multiple Time Columns with different geocentric observatory locations ({location}, {col.location}) encountered.This is not supported by the FITS standard.')
    return (newtable, hdr)