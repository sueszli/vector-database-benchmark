"""
Display numbers as strings using engineering notation.
"""
scale_factor = {}
scale_factor['E'] = 1e+18
scale_factor['P'] = 1000000000000000.0
scale_factor['T'] = 1000000000000.0
scale_factor['G'] = 1000000000.0
scale_factor['M'] = 1000000.0
scale_factor['k'] = 1000.0
scale_factor['m'] = 0.001
scale_factor['u'] = 1e-06
scale_factor['n'] = 1e-09
scale_factor['p'] = 1e-12
scale_factor['f'] = 1e-15
scale_factor['a'] = 1e-18

def num_to_str(n, precision=6):
    if False:
        i = 10
        return i + 15
    'Convert a number to a string in engineering notation.  E.g., 5e-9 -> 5n'
    m = abs(n)
    format_spec = '%.' + repr(int(precision)) + 'g'
    if m >= 1000000000.0:
        return '%sG' % float(format_spec % (n * 1e-09))
    elif m >= 1000000.0:
        return '%sM' % float(format_spec % (n * 1e-06))
    elif m >= 1000.0:
        return '%sk' % float(format_spec % (n * 0.001))
    elif m >= 1:
        return '%s' % float(format_spec % n)
    elif m >= 0.001:
        return '%sm' % float(format_spec % (n * 1000.0))
    elif m >= 1e-06:
        return '%su' % float(format_spec % (n * 1000000.0))
    elif m >= 1e-09:
        return '%sn' % float(format_spec % (n * 1000000000.0))
    elif m >= 1e-12:
        return '%sp' % float(format_spec % (n * 1000000000000.0))
    elif m >= 1e-15:
        return '%sf' % float(format_spec % (n * 1000000000000000.0))
    else:
        return '%s' % float(format_spec % n)

def str_to_num(value):
    if False:
        for i in range(10):
            print('nop')
    "Convert a string in engineering notation to a number.  E.g., '15m' -> 15e-3"
    try:
        if not isinstance(value, str):
            raise TypeError('Value must be a string')
        scale = 1.0
        suffix = value[-1]
        if suffix in scale_factor:
            return float(value[0:-1]) * scale_factor[suffix]
        return float(value)
    except (TypeError, KeyError, ValueError):
        raise ValueError('Invalid engineering notation value: %r' % (value,))