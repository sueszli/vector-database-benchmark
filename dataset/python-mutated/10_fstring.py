"""This program is self-checking!"""
var1 = 'x'
var2 = 'y'
abc = 'def'
assert f"interpolate {var1} strings {var2!r} {var2!s} 'py36" == "interpolate x strings 'y' y 'py36"
assert 'def0' == f'{abc}0'
assert 'defdef' == f'{abc}{abc!s}'

def fn(x):
    if False:
        return 10
    yield f'x:{(yield (lambda i: x * i))}'
(k, v) = ('1', ['2'])
x = f'{k}={v!r}'
y = f"functools.{x}({', '.join(v)})"
assert x == "1=['2']"
assert y == "functools.1=['2'](2)"
chunk = ['a', 'b', 'c']
chunk2 = 'd'
chunk = f'{len(chunk):X}' + chunk2
assert chunk == '3d'
chunk = b'abc'
chunk2 = 'd'
chunk = f'{len(chunk):X}\r\n'.encode('ascii') + chunk + b'\r\n'
assert chunk == b'3\r\nabc\r\n'
import os
filename = '.'
source = 'foo'
source = f"__file__ = r'''{os.path.abspath(filename)}'''\n" + source + '\ndel __file__'
f = 'one'
name = 'two'
assert f"{f}{'{{name}}'} {f}{'{name}'}" == 'one{{name}} one{name}'
log_rounds = 5
assert '05$' == f'{log_rounds:02d}$'

def testit(a, b, ll):
    if False:
        i = 10
        return i + 15
    return ll

def _repr_fn(fields):
    if False:
        i = 10
        return i + 15
    return testit('__repr__', ('self',), ['return xx + f"(' + ', '.join([f'{f}={{self.{f}!r}}' for f in fields]) + ')"'])
fields = ['a', 'b', 'c']
assert _repr_fn(fields) == ['return xx + f"(a={self.a!r}, b={self.b!r}, c={self.c!r})"']
x = 5
assert f"{(lambda y: x * y)('8')!r}" == "'88888'"
assert f"{(lambda y: x * y)('8')!r:10}" == "'88888'   "
assert f"{(lambda y: x * y)('8'):10}" == '88888     '
try:
    eval("f'{lambda x:x}'")
except SyntaxError:
    pass
else:
    assert False, "f'{lambda x:x}' should be a syntax error"
(x, y, width) = ('foo', 2, 10)
assert f'x={x * y:{width}}' == 'x=foofoo    '

def f():
    if False:
        print('Hello World!')
    f'Not a docstring'

def g():
    if False:
        i = 10
        return i + 15
    f'Not a docstring'
assert f.__doc__ is None
assert g.__doc__ is None
import decimal
(width, precision, value) = (10, 4, decimal.Decimal('12.34567'))
assert f'result: {value:{width}.{precision}}' == 'result:      12.35'
assert f'result: {value:{width:0}.{precision:1}}' == 'result:      12.35'
assert f'{2}\t' == '2\t'
assert f"{f'{0}' * 3}" == '000'
assert f'expr={ {x: y for (x, y) in [(1, 2)]}}' == 'expr={1: 2}'

class Line:

    def __init__(self, x, y):
        if False:
            while True:
                i = 10
        self.x = x
        self.y = y

    def __str__(self):
        if False:
            return 10
        return f'{self.x} -> {self.y}'
line = Line(1, 2)
assert str(line) == '1 -> 2'