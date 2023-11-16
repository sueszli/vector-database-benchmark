"""
ISO8601 date related utility functions.
"""
from __future__ import absolute_import
import re
import datetime
from dateutil import tz as tzi
from st2common.util import date as date_utils
import six
__all__ = ['format', 'validate', 'parse']
ISO8601_FORMAT = '%Y-%m-%dT%H:%M:%S'
ISO8601_FORMAT_MICROSECOND = '%Y-%m-%dT%H:%M:%S.%f'
ISO8601_UTC_REGEX = '^\\d{4}\\-\\d{2}\\-\\d{2}(\\s|T)\\d{2}:\\d{2}:\\d{2}(\\.\\d{3,6})?(Z|\\+00|\\+0000|\\+00:00)$'

def format(dt, usec=True, offset=True):
    if False:
        while True:
            i = 10
    '\n    Format a provided datetime object and return ISO8601 string.\n\n    :type dt: ``datetime.datetime``\n    '
    if isinstance(dt, six.string_types):
        dt = parse(dt)
    elif isinstance(dt, int):
        dt = datetime.datetime.fromtimestamp(dt, tzi.tzutc())
    fmt = ISO8601_FORMAT_MICROSECOND if usec else ISO8601_FORMAT
    if offset:
        ost = dt.strftime('%z')
        ost = ost[:3] + ':' + ost[3:] if ost else '+00:00'
    else:
        tz = dt.tzinfo.tzname(dt) if dt.tzinfo else 'UTC'
        ost = 'Z' if tz == 'UTC' else tz
    return dt.strftime(fmt) + ost

def validate(value, raise_exception=True):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(value, datetime.datetime) or (type(value) in [str, six.text_type] and re.match(ISO8601_UTC_REGEX, value)):
        return True
    if raise_exception:
        raise ValueError('Datetime value does not match expected format.')
    return False

def parse(value, preserve_original_tz=False, validate_value=True):
    if False:
        while True:
            i = 10
    '\n    Parse date in the ISO8601 format and return a time-zone aware datetime object.\n\n    :param value: Date in ISO8601 format.\n    :type value: ``str``\n\n    :param preserve_original_tz: True to preserve the original timezone - by default result is\n                                 converted into UTC.\n    :type preserve_original_tz: ``boolean``\n\n    :param validate_value: True to validate that the date is in the ISO8601 format.\n    :type validate_value: ``boolean``\n\n    :rtype: ``datetime.datetime``\n    '
    if validate_value:
        validate(value, raise_exception=True)
    dt = date_utils.parse(value=value, preserve_original_tz=preserve_original_tz)
    return dt