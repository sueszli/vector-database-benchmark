"""Functions for reporting filesizes. Borrowed from https://github.com/PyFilesystem/pyfilesystem2

The functions declared in this module should cover the different
use cases needed to generate a string representation of a file size
using several different units. Since there are many standards regarding
file size units, three different functions have been implemented.

See Also:
    * `Wikipedia: Binary prefix <https://en.wikipedia.org/wiki/Binary_prefix>`_

"""
__all__ = ['decimal']
from typing import Iterable, List, Optional, Tuple

def _to_str(size: int, suffixes: Iterable[str], base: int, *, precision: Optional[int]=1, separator: Optional[str]=' ') -> str:
    if False:
        return 10
    if size == 1:
        return '1 byte'
    elif size < base:
        return '{:,} bytes'.format(size)
    for (i, suffix) in enumerate(suffixes, 2):
        unit = base ** i
        if size < unit:
            break
    return '{:,.{precision}f}{separator}{}'.format(base * size / unit, suffix, precision=precision, separator=separator)

def pick_unit_and_suffix(size: int, suffixes: List[str], base: int) -> Tuple[int, str]:
    if False:
        i = 10
        return i + 15
    'Pick a suffix and base for the given size.'
    for (i, suffix) in enumerate(suffixes):
        unit = base ** i
        if size < unit * base:
            break
    return (unit, suffix)

def decimal(size: int, *, precision: Optional[int]=1, separator: Optional[str]=' ') -> str:
    if False:
        for i in range(10):
            print('nop')
    'Convert a filesize in to a string (powers of 1000, SI prefixes).\n\n    In this convention, ``1000 B = 1 kB``.\n\n    This is typically the format used to advertise the storage\n    capacity of USB flash drives and the like (*256 MB* meaning\n    actually a storage capacity of more than *256 000 000 B*),\n    or used by **Mac OS X** since v10.6 to report file sizes.\n\n    Arguments:\n        int (size): A file size.\n        int (precision): The number of decimal places to include (default = 1).\n        str (separator): The string to separate the value from the units (default = " ").\n\n    Returns:\n        `str`: A string containing a abbreviated file size and units.\n\n    Example:\n        >>> filesize.decimal(30000)\n        \'30.0 kB\'\n        >>> filesize.decimal(30000, precision=2, separator="")\n        \'30.00kB\'\n\n    '
    return _to_str(size, ('kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'), 1000, precision=precision, separator=separator)