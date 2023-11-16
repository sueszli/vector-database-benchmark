"""Formatters are mappings from object(s) to a string."""
import decimal
import math
import re
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from markupsafe import escape

def list_args(func: Callable) -> Callable:
    if False:
        i = 10
        return i + 15
    'Extend the function to allow taking a list as the first argument, and apply the function on each of the elements.\n\n    Args:\n        func: the function to extend\n\n    Returns:\n        The extended function\n    '

    def inner(arg: Any, *args: Any, **kwargs: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(arg, list):
            return [func(v, *args, **kwargs) for v in arg]
        return func(arg, *args, **kwargs)
    return inner

@list_args
def fmt_color(text: str, color: str) -> str:
    if False:
        i = 10
        return i + 15
    'Format a string in a certain color (`<span>`).\n\n    Args:\n      text: The text to format.\n      color: Any valid CSS color.\n\n    Returns:\n        A `<span>` that contains the colored text.\n    '
    return f'<span style="color:{color}">{text}</span>'

@list_args
def fmt_class(text: str, cls: str) -> str:
    if False:
        i = 10
        return i + 15
    'Format a string in a certain class (`<span>`).\n\n    Args:\n      text: The text to format.\n      cls: The name of the class.\n\n    Returns:\n        A `<span>` with a class added.\n    '
    return f'<span class="{cls}">{text}</span>'

@list_args
def fmt_bytesize(num: float, suffix: str='B') -> str:
    if False:
        i = 10
        return i + 15
    "Change a number of bytes in a human-readable format.\n\n    Args:\n      num: number to format\n      suffix: (Default value = 'B')\n\n    Returns:\n      The value formatted in human readable format (e.g. KiB).\n    "
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f'{num:3.1f} {unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f} Yi{suffix}'

@list_args
def fmt_percent(value: float, edge_cases: bool=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Format a ratio as a percentage.\n\n    Args:\n        edge_cases: Check for edge cases?\n        value: The ratio.\n\n    Returns:\n        The percentage with 1 point precision.\n    '
    if edge_cases and round(value, 3) == 0 and (value > 0):
        return '< 0.1%'
    if edge_cases and round(value, 3) == 1 and (value < 1):
        return '> 99.9%'
    return f'{value * 100:2.1f}%'

@list_args
def fmt_timespan(num_seconds: Any, detailed: bool=False, max_units: int=3) -> str:
    if False:
        for i in range(10):
            print('nop')
    time_units: List[Dict[str, Any]] = [{'divider': 1e-09, 'singular': 'nanosecond', 'plural': 'nanoseconds', 'abbreviations': ['ns']}, {'divider': 1e-06, 'singular': 'microsecond', 'plural': 'microseconds', 'abbreviations': ['us']}, {'divider': 0.001, 'singular': 'millisecond', 'plural': 'milliseconds', 'abbreviations': ['ms']}, {'divider': 1, 'singular': 'second', 'plural': 'seconds', 'abbreviations': ['s', 'sec', 'secs']}, {'divider': 60, 'singular': 'minute', 'plural': 'minutes', 'abbreviations': ['m', 'min', 'mins']}, {'divider': 60 * 60, 'singular': 'hour', 'plural': 'hours', 'abbreviations': ['h']}, {'divider': 60 * 60 * 24, 'singular': 'day', 'plural': 'days', 'abbreviations': ['d']}, {'divider': 60 * 60 * 24 * 7, 'singular': 'week', 'plural': 'weeks', 'abbreviations': ['w']}, {'divider': 60 * 60 * 24 * 7 * 52, 'singular': 'year', 'plural': 'years', 'abbreviations': ['y']}]

    def round_number(count: Any, keep_width: bool=False) -> str:
        if False:
            i = 10
            return i + 15
        text = f'{float(count):.2f}'
        if not keep_width:
            text = re.sub('0+$', '', text)
            text = re.sub('\\.$', '', text)
        return text

    def coerce_seconds(value: Union[timedelta, int, float]) -> float:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, timedelta):
            return value.total_seconds()
        return float(value)

    def concatenate(items: List[str]) -> str:
        if False:
            print('Hello World!')
        items = list(items)
        if len(items) > 1:
            return ', '.join(items[:-1]) + ' and ' + items[-1]
        elif items:
            return items[0]
        else:
            return ''

    def pluralize(count: Any, singular: str, plural: Optional[str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        if not plural:
            plural = singular + 's'
        return f'{count} {(singular if math.floor(float(count)) == 1 else plural)}'
    num_seconds = coerce_seconds(num_seconds)
    if num_seconds < 60 and (not detailed):
        return pluralize(round_number(num_seconds), 'second')
    else:
        result = []
        num_seconds = decimal.Decimal(str(num_seconds))
        relevant_units = list(reversed(time_units[0 if detailed else 3:]))
        for unit in relevant_units:
            divider = decimal.Decimal(str(unit['divider']))
            count = num_seconds / divider
            num_seconds %= divider
            if unit != relevant_units[-1]:
                count = int(count)
            else:
                count = round_number(count)
            if count not in (0, '0'):
                result.append(pluralize(count, unit['singular'], unit['plural']))
        if len(result) == 1:
            return result[0]
        else:
            if not detailed:
                result = result[:max_units]
            return concatenate(result)

def fmt_timespan_timedelta(delta: Any, detailed: bool=False, max_units: int=3, precision: int=10) -> str:
    if False:
        return 10
    if isinstance(delta, pd.Timedelta):
        num_seconds = delta.total_seconds()
        if delta.microseconds > 0:
            num_seconds += delta.microseconds * 1e-06
        if delta.nanoseconds > 0:
            num_seconds += delta.nanoseconds * 1e-09
        return fmt_timespan(num_seconds, detailed, max_units)
    else:
        return fmt_numeric(delta, precision)

@list_args
def fmt_numeric(value: float, precision: int=10) -> str:
    if False:
        print('Hello World!')
    'Format any numeric value.\n\n    Args:\n        value: The numeric value to format.\n        precision: The numeric precision\n\n    Returns:\n        The numeric value with the given precision.\n    '
    fmtted = f'{{:.{precision}g}}'.format(value)
    for v in ['e+', 'e-']:
        if v in fmtted:
            sign = '-' if v in 'e-' else ''
            fmtted = fmtted.replace(v, ' × 10<sup>') + '</sup>'
            fmtted = fmtted.replace('<sup>0', '<sup>')
            fmtted = fmtted.replace('<sup>', f'<sup>{sign}')
    return fmtted

@list_args
def fmt_number(value: int) -> str:
    if False:
        i = 10
        return i + 15
    'Format any numeric value.\n\n    Args:\n        value: The numeric value to format.\n\n    Returns:\n        The numeric value with the given precision.\n    '
    return f'{value:n}'

@list_args
def fmt_array(value: np.ndarray, threshold: Any=np.nan) -> str:
    if False:
        while True:
            i = 10
    'Format numpy arrays.\n\n    Args:\n        value: Array to format.\n        threshold: Threshold at which to show ellipsis\n\n    Returns:\n        The string representation of the numpy array.\n    '
    with np.printoptions(threshold=3, edgeitems=threshold):
        return_value = str(value)
    return return_value

@list_args
def fmt(value: Any) -> str:
    if False:
        print('Hello World!')
    'Format any value.\n\n    Args:\n        value: The value to format.\n\n    Returns:\n        The numeric formatting if the value is float or int, the string formatting otherwise.\n    '
    if type(value) in [float, int]:
        return fmt_numeric(value)
    else:
        return str(escape(value))

@list_args
def fmt_monotonic(value: int) -> str:
    if False:
        return 10
    if value == 2:
        return 'Strictly increasing'
    elif value == 1:
        return 'Increasing'
    elif value == 0:
        return 'Not monotonic'
    elif value == -1:
        return 'Decreasing'
    elif value == -2:
        return 'Strictly decreasing'
    else:
        raise ValueError('Value should be integer ranging from -2 to 2.')

def help(title: str, url: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    'Creat help badge\n\n    Args:\n        title: help text\n        url: url to open in new tab (optional)\n\n    Returns:\n        HTML formatted help badge\n    '
    if url is not None:
        return f'<a title="{title}" href="{url}"><span class="badge pull-right" style="color:#fff;background-color:#337ab7;" title="{title}">?</span></a>'
    else:
        return f'<span class="badge pull-right" style="color:#fff;background-color:#337ab7;" title="{title}">?</span>'

@list_args
def fmt_badge(value: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return re.sub('\\((\\d+)\\)', '<span class="badge">\\1</span>', value)