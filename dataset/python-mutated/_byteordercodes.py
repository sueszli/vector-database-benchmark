""" Byteorder utilities for system - numpy byteorder encoding

Converts a variety of string codes for little endian, big endian,
native byte order and swapped byte order to explicit NumPy endian
codes - one of '<' (little endian) or '>' (big endian)

"""
import sys
__all__ = ['aliases', 'native_code', 'swapped_code', 'sys_is_le', 'to_numpy_code']
sys_is_le = sys.byteorder == 'little'
native_code = sys_is_le and '<' or '>'
swapped_code = sys_is_le and '>' or '<'
aliases = {'little': ('little', '<', 'l', 'le'), 'big': ('big', '>', 'b', 'be'), 'native': ('native', '='), 'swapped': ('swapped', 'S')}

def to_numpy_code(code):
    if False:
        return 10
    "\n    Convert various order codings to NumPy format.\n\n    Parameters\n    ----------\n    code : str\n        The code to convert. It is converted to lower case before parsing.\n        Legal values are:\n        'little', 'big', 'l', 'b', 'le', 'be', '<', '>', 'native', '=',\n        'swapped', 's'.\n\n    Returns\n    -------\n    out_code : {'<', '>'}\n        Here '<' is the numpy dtype code for little endian,\n        and '>' is the code for big endian.\n\n    Examples\n    --------\n    >>> import sys\n    >>> from scipy.io.matlab._byteordercodes import to_numpy_code\n    >>> sys_is_le = (sys.byteorder == 'little')\n    >>> sys_is_le\n    True\n    >>> to_numpy_code('big')\n    '>'\n    >>> to_numpy_code('little')\n    '<'\n    >>> nc = to_numpy_code('native')\n    >>> nc == '<' if sys_is_le else nc == '>'\n    True\n    >>> sc = to_numpy_code('swapped')\n    >>> sc == '>' if sys_is_le else sc == '<'\n    True\n\n    "
    code = code.lower()
    if code is None:
        return native_code
    if code in aliases['little']:
        return '<'
    elif code in aliases['big']:
        return '>'
    elif code in aliases['native']:
        return native_code
    elif code in aliases['swapped']:
        return swapped_code
    else:
        raise ValueError('We cannot handle byte order %s' % code)