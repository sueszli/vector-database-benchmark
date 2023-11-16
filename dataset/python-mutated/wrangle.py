import re
import tarfile
from datetime import datetime
import numpy as np
import pandas as pd
from woodwork.logical_types import Datetime, Ordinal
from featuretools.entityset.timedelta import Timedelta

def _check_timedelta(td):
    if False:
        return 10
    '\n    Convert strings to Timedelta objects\n    Allows for both shortform and longform units, as well as any form of capitalization\n    \'2 Minutes\'\n    \'2 minutes\'\n    \'2 m\'\n    \'1 Minute\'\n    \'1 minute\'\n    \'1 m\'\n    \'1 units\'\n    \'1 Units\'\n    \'1 u\'\n    Shortform is fine if space is dropped\n    \'2m\'\n    \'1u"\n    If a pd.Timedelta object is passed, units will be converted to seconds due to the underlying representation\n        of pd.Timedelta.\n    If a pd.DateOffset object is passed, it will be converted to a Featuretools Timedelta if it has one\n        temporal parameter. Otherwise, it will remain a pd.DateOffset.\n    '
    if td is None:
        return td
    if isinstance(td, Timedelta):
        return td
    elif not isinstance(td, (int, float, str, pd.DateOffset, pd.Timedelta)):
        raise ValueError('Unable to parse timedelta: {}'.format(td))
    if isinstance(td, pd.Timedelta):
        unit = 's'
        value = td.total_seconds()
        times = {unit: value}
        return Timedelta(times, delta_obj=td)
    elif isinstance(td, pd.DateOffset):
        if td.__class__.__name__ != 'DateOffset':
            if hasattr(td, '__dict__'):
                value = td.__dict__['n']
            else:
                value = td.n
            unit = td.__class__.__name__
            times = dict([(unit, value)])
        else:
            times = dict()
            for (td_unit, td_value) in td.kwds.items():
                times[td_unit] = td_value
        return Timedelta(times, delta_obj=td)
    else:
        pattern = '([0-9]+) *([a-zA-Z]+)$'
        match = re.match(pattern, td)
        (value, unit) = match.groups()
        try:
            value = int(value)
        except Exception:
            try:
                value = float(value)
            except Exception:
                raise ValueError('Unable to parse value {} from '.format(value) + 'timedelta string: {}'.format(td))
        times = {unit: value}
        return Timedelta(times)

def _check_time_against_column(time, time_column):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check to make sure that time is compatible with time_column,\n    where time could be a timestamp, or a Timedelta, number, or None,\n    and time_column is a Woodwork initialized column. Compatibility means that\n    arithmetic can be performed between time and elements of time_column\n\n    If time is None, then we don't care if arithmetic can be performed\n    (presumably it won't ever be performed)\n    "
    if time is None:
        return True
    elif isinstance(time, (int, float)):
        return time_column.ww.schema.is_numeric
    elif isinstance(time, (pd.Timestamp, datetime, pd.DateOffset)):
        return time_column.ww.schema.is_datetime
    elif isinstance(time, Timedelta):
        if time_column.ww.schema.is_datetime:
            return True
        elif time.unit not in Timedelta._time_units:
            if isinstance(time_column.ww.logical_type, Ordinal) or 'numeric' in time_column.ww.semantic_tags or 'time_index' in time_column.ww.semantic_tags:
                return True
    return False

def _check_time_type(time):
    if False:
        i = 10
        return i + 15
    '\n    Checks if `time` is an instance of common int, float, or datetime types.\n    Returns "numeric" or Datetime based on results\n    '
    time_type = None
    if isinstance(time, (datetime, np.datetime64)):
        time_type = Datetime
    elif isinstance(time, (int, float)) or np.issubdtype(time, np.integer) or np.issubdtype(time, np.floating):
        time_type = 'numeric'
    return time_type

def _is_s3(string):
    if False:
        i = 10
        return i + 15
    '\n    Checks if the given string is a s3 path.\n    Returns a boolean.\n    '
    return string.startswith('s3://')

def _is_url(string):
    if False:
        while True:
            i = 10
    '\n    Checks if the given string is an url path.\n    Returns a boolean.\n    '
    return string.startswith('http')

def _is_local_tar(string):
    if False:
        print('Hello World!')
    '\n    Checks if the given string is a local tarfile path.\n    Returns a boolean.\n    '
    return string.endswith('.tar') and tarfile.is_tarfile(string)