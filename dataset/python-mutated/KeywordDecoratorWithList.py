from datetime import date
from decimal import Decimal
from robot.api.deco import keyword

@keyword(types=[int, Decimal, bool, date, list])
def basics(integer, decimal, boolean, date_, list_=None):
    if False:
        i = 10
        return i + 15
    _validate_type(integer, 42)
    _validate_type(decimal, Decimal('3.14'))
    _validate_type(boolean, True)
    _validate_type(date_, date(2018, 8, 30))
    _validate_type(list_, ['foo'])

@keyword(types=[int, None, float])
def none_means_no_type(foo, bar, zap):
    if False:
        while True:
            i = 10
    _validate_type(foo, 1)
    _validate_type(bar, '2')
    _validate_type(zap, 3.0)

@keyword(types=['', int, False])
def falsy_types_mean_no_type(foo, bar, zap):
    if False:
        i = 10
        return i + 15
    _validate_type(foo, '1')
    _validate_type(bar, 2)
    _validate_type(zap, '3')

@keyword(types=[int, type(None), float])
def nonetype(foo, bar, zap):
    if False:
        i = 10
        return i + 15
    _validate_type(foo, 1)
    _validate_type(bar, None)
    _validate_type(zap, 3.0)

@keyword(types=[int, 'None', float])
def none_as_string_is_none(foo, bar, zap):
    if False:
        return 10
    _validate_type(foo, 1)
    _validate_type(bar, None)
    _validate_type(zap, 3.0)

@keyword(types=[(int, None), (None,)])
def none_in_tuple_is_alias_for_nonetype(arg1, arg2, exp1=None, exp2=None):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(arg1, eval(exp1) if exp1 else None)
    _validate_type(arg2, eval(exp2) if exp2 else None)

@keyword(types=[int, float])
def less_types_than_arguments_is_ok(foo, bar, zap):
    if False:
        i = 10
        return i + 15
    _validate_type(foo, 1)
    _validate_type(bar, 2.0)
    _validate_type(zap, '3')

@keyword(types=[int, int])
def too_many_types(argument):
    if False:
        i = 10
        return i + 15
    raise RuntimeError('Should not be executed!')

@keyword(types=[int, int, int])
def varargs_and_kwargs(arg, *varargs, **kwargs):
    if False:
        i = 10
        return i + 15
    _validate_type(arg, 1)
    _validate_type(varargs, (2, 3, 4))
    _validate_type(kwargs, {'kw': 5})

@keyword(types=[None, int, float])
def kwonly(*, foo, bar=None, zap):
    if False:
        for i in range(10):
            print('nop')
    _validate_type(foo, '1')
    _validate_type(bar, 2)
    _validate_type(zap, 3.0)

@keyword(types=[None, None, int, float, Decimal])
def kwonly_with_varargs_and_kwargs(*varargs, foo, bar=None, zap, **kwargs):
    if False:
        print('Hello World!')
    _validate_type(varargs, ('0',))
    _validate_type(foo, '1')
    _validate_type(bar, 2)
    _validate_type(zap, 3.0)
    _validate_type(kwargs, {'quux': Decimal(4)})

def _validate_type(argument, expected):
    if False:
        return 10
    if argument != expected or type(argument) != type(expected):
        raise AssertionError('%r (%s) != %r (%s)' % (argument, type(argument).__name__, expected, type(expected).__name__))