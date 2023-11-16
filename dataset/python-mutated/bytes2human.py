"""Helper class to convert bytes to human-readable form

It supports l10n through gettext, decimal and binary units.

>>> n = 1572864
>>> [binary(n), decimal(n)]
['1.5 MiB', '1.6 MB']
"""
import locale
_BYTES_STRINGS_I18N = (N_('%(value)s B'), N_('%(value)s kB'), N_('%(value)s KiB'), N_('%(value)s MB'), N_('%(value)s MiB'), N_('%(value)s GB'), N_('%(value)s GiB'), N_('%(value)s TB'), N_('%(value)s TiB'), N_('%(value)s PB'), N_('%(value)s PiB'))

def decimal(number, scale=1, l10n=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert bytes to short human-readable string, decimal mode\n\n    >>> [decimal(n) for n in [1000, 1024, 15500]]\n    ['1 kB', '1 kB', '15.5 kB']\n    "
    return short_string(int(number), 1000, scale=scale, l10n=l10n)

def binary(number, scale=1, l10n=True):
    if False:
        i = 10
        return i + 15
    "\n    Convert bytes to short human-readable string, binary mode\n    >>> [binary(n) for n in [1000, 1024, 15500]]\n    ['1000 B', '1 KiB', '15.1 KiB']\n    "
    return short_string(int(number), 1024, scale=scale, l10n=l10n)

def short_string(number, multiple, scale=1, l10n=True):
    if False:
        return 10
    "\n    Returns short human-readable string for `number` bytes\n    >>> [short_string(n, 1024, 2) for n in [1000, 1100, 15500]]\n    ['1000 B', '1.07 KiB', '15.14 KiB']\n    >>> [short_string(n, 1000, 1) for n in [10000, 11000, 1550000]]\n    ['10 kB', '11 kB', '1.6 MB']\n    "
    (num, unit) = calc_unit(number, multiple)
    n = int(num)
    nr = round(num, scale)
    if n == nr or unit == 'B':
        fmt = '%d'
        num = n
    else:
        fmt = '%%0.%df' % scale
        num = nr
    if l10n:
        fmtnum = locale.format_string(fmt, num)
        fmt = '%(value)s ' + unit
        return _(fmt) % {'value': fmtnum}
    else:
        return fmt % num + ' ' + unit

def calc_unit(number, multiple=1000):
    if False:
        while True:
            i = 10
    "\n    Calculate rounded number of multiple * bytes, finding best unit\n\n    >>> calc_unit(12456, 1024)\n    (12.1640625, 'KiB')\n    >>> calc_unit(-12456, 1000)\n    (-12.456, 'kB')\n    >>> calc_unit(0, 1001)\n    Traceback (most recent call last):\n        ...\n    ValueError: multiple parameter has to be 1000 or 1024\n    "
    if number < 0:
        sign = -1
        number = -number
    else:
        sign = 1
    n = float(number)
    if multiple == 1000:
        (k, b) = ('k', 'B')
    elif multiple == 1024:
        (k, b) = ('K', 'iB')
    else:
        raise ValueError('multiple parameter has to be 1000 or 1024')
    suffixes = ['B'] + [i + b for i in k + 'MGTP']
    for suffix in suffixes:
        if n < multiple or suffix == suffixes[-1]:
            return (sign * n, suffix)
        else:
            n /= multiple