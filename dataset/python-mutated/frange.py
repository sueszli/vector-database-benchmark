from .robottypes import is_integer, is_string

def frange(*args):
    if False:
        for i in range(10):
            print('nop')
    'Like ``range()`` but accepts float arguments.'
    if all((is_integer(arg) for arg in args)):
        return list(range(*args))
    (start, stop, step) = _get_start_stop_step(args)
    digits = max(_digits(start), _digits(stop), _digits(step))
    factor = pow(10, digits)
    return [x / factor for x in range(round(start * factor), round(stop * factor), round(step * factor))]

def _get_start_stop_step(args):
    if False:
        for i in range(10):
            print('nop')
    if len(args) == 1:
        return (0, args[0], 1)
    if len(args) == 2:
        return (args[0], args[1], 1)
    if len(args) == 3:
        return args
    raise TypeError('frange expected 1-3 arguments, got %d.' % len(args))

def _digits(number):
    if False:
        return 10
    if not is_string(number):
        number = repr(number)
    if 'e' in number:
        return _digits_with_exponent(number)
    if '.' in number:
        return _digits_with_fractional(number)
    return 0

def _digits_with_exponent(number):
    if False:
        for i in range(10):
            print('nop')
    (mantissa, exponent) = number.split('e')
    mantissa_digits = _digits(mantissa)
    exponent_digits = int(exponent) * -1
    return max(mantissa_digits + exponent_digits, 0)

def _digits_with_fractional(number):
    if False:
        for i in range(10):
            print('nop')
    fractional = number.split('.')[1]
    if fractional == '0':
        return 0
    return len(fractional)