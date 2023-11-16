""" Test that should cover sporadic usages of built-ins that we implemented.

"""
from __future__ import print_function
import os
from math import copysign

def someFunctionWritingLocals():
    if False:
        while True:
            i = 10
    x = 1
    r = locals()
    y = 2
    r['z'] = 3
    del x
    try:
        z
    except Exception as e:
        print('Accessing z writing to locals gives Exception', e)
    return (r, y)

def someFunctionWritingLocalsContainingExec():
    if False:
        print('Hello World!')
    _x = 1
    r = locals()
    y = 2
    r['z'] = 3
    try:
        z
    except Exception as e:
        print('Accessing z writing to locals in exec function gives Exception', e)
    return (r, y)
    exec('')
print('Testing locals():')
print('writing to locals():', someFunctionWritingLocals())
print('writing to locals() with exec() usage:', someFunctionWritingLocalsContainingExec())

def displayDict(d):
    if False:
        return 10
    if '__loader__' in d:
        d = dict(d)
        if str is bytes:
            del d['__loader__']
        else:
            d['__loader__'] = '<__loader__ removed>'
    if '__file__' in d:
        d = dict(d)
        d['__file__'] = '<__file__ removed>'
    if '__compiled__' in d:
        d = dict(d)
        del d['__compiled__']
    import pprint
    return pprint.pformat(d)
print('Vars on module level', displayDict(vars()))
module_locals = locals()
module_locals['__file__'] = os.path.basename(module_locals['__file__'])
del module_locals
print('Use of locals on the module level', displayDict(locals()))

def someFunctionUsingGlobals():
    if False:
        for i in range(10):
            print('nop')
    g = globals()
    g['hallo'] = 'du'
    global hallo
    print('hallo', hallo)
print('Testing globals():')
someFunctionUsingGlobals()
print('Testing dir():')
print('Module dir', end=' ')

def someFunctionUsingDir():
    if False:
        for i in range(10):
            print('nop')
    q = someFunctionUsingGlobals()
    print('Function dir', dir())
    return q
someFunctionUsingDir()
print('Making a new type, with type() and 3 args:', end=' ')
NewClass = type('Name', (object,), {})
print(NewClass, NewClass())
print('None has type', type(None))
print('Constant ranges', range(2), range(1, 6), range(3, 0, -1), range(3, 8, 2), range(5, -5, -3))
print('Border cases', range(0), range(-1), range(-1, 1))
print('Corner case large negative value', range(-2 ** 100))
print('Corner case with large start/end values in small range', range(2 ** 100, 2 ** 100 + 2))
try:
    print('Range with 0 step gives:', end=' ')
    print(range(3, 8, 0))
except ValueError as e:
    print(repr(e))
try:
    print('Range with float:', end=' ')
    print(range(1.0))
except TypeError as e:
    print('Gives exception:', repr(e))
try:
    print('Empty range call', end=' ')
    print(range())
except TypeError as e:
    print('Gives exception:', e)
print('List from iterable', list('abc'), list())
try:
    print('List from sequence', end=' ')
    print(list(sequence=(0, 1, 2)))
except TypeError as e:
    print('Gives exception:', e)
print('Tuple from iterable', tuple('cda'), tuple())
try:
    print('Tuple from sequence', end=' ')
    print(tuple(sequence=(0, 1, 2)))
except TypeError as e:
    print('Gives exception:', e)
print('Dictionary from iterable and keywords', displayDict(dict(('ab', (1, 2)), f=1, g=1)))
print('More constant dictionaries', {'two': 2, 'one': 1}, {}, dict())
g = {'two': 2, 'one': 1}
print('Variable dictionary', dict(g))
print('Found during optimization', dict(dict({'le': 2, 'la': 1}), fu=3), dict(named=dict({'le': 2, 'la': 1})))
print('Floats from constants', float('3.0'), float())
try:
    print('Float keyword arg', end=' ')
except TypeError as e:
    print(float(x=9.0))
print('Found during optimization', float(float('3.2')), float(float(11.0)))
print('Complex from constants', complex('3.0j'), complex(real=9.0), complex(imag=9.0), complex(1, 2), complex())
print('Found during optimization', complex(float('3.2')), complex(real=float(11.0)), complex(imag=float(11.0)))
print('Strs from constants', str('3.3'), str(object=9.1), str())
print('Found during optimization', str(float('3.3')), str(object=float(12.0)))
print('Bools from constants', bool('3.3'), bool(0), bool())
print('Found during optimization', bool(float('3.3')), bool(range(0)))
print('Ints from constants', int('3'), int('f', 16), int('0101', base=2), int(0), int())
try:
    print('Int keyword arg1', end=' ')
    print(int(x='9'))
    print(int(x='e', base=16))
except TypeError as e:
    print('Gives exception:', e)
print('Found ints during optimization', int(int('3')), int(int(0.0)))
try:
    print('Longs from constants', long('3'), long(x='9'), long('f', 16), long(x='e', base=16), long('0101', base=2), long(0), long())
    print('Found longs during optimization', long(long('3')), long(x=long(0.0)))
except NameError:
    print('Python3 has no long type.')
try:
    print('Int with only base', int(base=2), end=' ')
except Exception as e:
    print('Caused', repr(e))
else:
    print('Worked')
try:
    print('Int with large base', int(2, 37), end=' ')
except Exception as e:
    print('Caused', repr(e))
else:
    print('Worked')
try:
    print('Long with only base', int(base=2), end=' ')
except Exception as e:
    print('Caused', repr(e))
else:
    print('Worked')
print('Oct from constants', oct(467), oct(0))
print('Found during optimization', oct(int('3')))
print('Hex from constants', hex(467), hex(0))
print('Found during optimization', hex(int('3')))
print('Bin from constants', bin(467), bin(0))
print('Found during optimization', bin(int('3')))
try:
    int(1, 2, 3)
except Exception as e:
    print('Too many args gave', repr(e))
try:
    int(y=1)
except Exception as e:
    print('Wrong arg', repr(e))
f = 3
print('Unoptimized call of int', int('0' * f, base=16))
try:
    d = {'x': '12', 'base': 8}
    print('Dict star argument call of int', end=' ')
    print(int(**d))
except TypeError as e:
    print('Gives exception:', e)
base = 16
try:
    value = unicode('20')
except NameError:
    print('Python3: Has unicode by default.')
    value = '20'
print('Unoptimized calls of int with unicode args', int(value, base), int(value))
base = 37
try:
    print('Int with large base', int(2, base), end=' ')
except Exception as e:
    print('Caused', repr(e))
else:
    print('Worked')
try:
    print(chr())
except Exception as e:
    print('Disallowed without args', repr(e))
try:
    print(ord())
except Exception as e:
    print('Disallowed without args', repr(e))
try:
    print(ord(s=1))
except Exception as e:
    print('Disallowed keyword args', repr(e))
try:
    print(ord(1, 2))
except Exception as e:
    print('Too many plain args', repr(e))
try:
    print(ord(1, s=2))
except Exception as e:
    print('Too many args, some keywords', repr(e))
try:
    print(sum())
except Exception as e:
    print('Disallowed without args', repr(e))
x = range(17)
print('Sum of range(17) is', sum(x))
print('Sum of range(17) starting with 5 is', sum(x, 5))
try:
    print(str('1', offer=2))
except Exception as e:
    print('Too many args, some keywords', repr(e))
a = 2
print('Can optimize the star list argness away', int(*(a,)), end=' ')
print('Can optimize the empty star list arg away', int(*tuple()), end=' ')
print('Can optimize the empty star dict arg away', int(**dict()))
print('Dict building with keyword arguments', dict(), dict(a=f))
print('Dictionary entirely from constant args', displayDict(dict(q='Guido', w='van', e='Rossum', r='invented', t='Python', y='')))
a = 5
print('Instance check recognises', isinstance(a, int))
try:
    print('Instance check with too many arguments', isinstance(a, float, int))
except Exception as e:
    print('Too many args', repr(e))
try:
    print('Instance check with too many arguments', isinstance(a))
except Exception as e:
    print('Too few args', repr(e))

def usingIterToCheckIterable(a):
    if False:
        while True:
            i = 10
    try:
        iter(a)
    except TypeError:
        print('not iterable')
    else:
        print('ok')
usingIterToCheckIterable(1)
print('Nested constant, dict inside a list, referencing a built-in compile time constant', end=' ')
print([dict(type=int)])
print('nan and -nan sign checks:')
print('nan:', float('nan'), copysign(1.0, float('nan')))
print('-nan:', float('-nan'), copysign(1.0, float('-nan')))
print('Using != to detect nan floats:')
a = float('nan')
if a != a:
    print('is nan')
else:
    print("isn't nan")
print('inf and -inf sign checks:')
print('inf:', float('inf'), copysign(1.0, float('inf')))
print('-inf:', float('-inf'), copysign(1.0, float('-inf')))

class CustomStr(str):
    pass

class CustomBytes(bytes):
    pass

class CustomByteArray(bytearray):
    pass
values = [b'100', b'', bytearray(b'100'), CustomStr('100'), CustomBytes(b'100'), CustomByteArray(b'100')]
for x in values:
    try:
        print('int', repr(x), int(x), int(x, 2))
    except (TypeError, ValueError) as e:
        print('caught', repr(e))
    try:
        print('long', repr(x), long(x), long(x, 2))
    except (TypeError, ValueError) as e:
        print('caught', repr(e))
    except NameError:
        print('Python3 has no long')
z = range(5)
try:
    next(z)
except TypeError as e:
    print('caught', repr(e))
try:
    open()
except TypeError as e:
    print('Open without arguments gives', repr(e))
print('Type of id values:', type(id(id)))

class OtherBytesSubclass(bytes):
    pass

class BytesOverload:

    def __bytes__(self):
        if False:
            print('Hello World!')
        return OtherBytesSubclass()
b = BytesOverload()
v = bytes(b)
if type(v) is bytes:
    print('Bytes overload ineffective (expected for Python2)')
elif isinstance(v, bytes):
    print('Bytes overload successful.')
else:
    print('Oops, must not happen.')

class OtherFloatSubclass(float):
    pass

class FloatOverload:

    def __float__(self):
        if False:
            for i in range(10):
                print('nop')
        return OtherFloatSubclass()
b = FloatOverload()
v = float(b)
if type(v) is float:
    print('Float overload ineffective (must not happen)')
elif isinstance(v, float):
    print('Float overload successful.')
else:
    print('Oops, must not happen.')

class OtherStrSubclass(str):
    pass

class StrOverload:

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return OtherStrSubclass()
b = StrOverload()
v = str(b)
if type(v) is str:
    print('Str overload ineffective (must not happen)')
elif isinstance(v, str):
    print('Str overload successful.')
else:
    print('Oops, must not happen.')
if str is bytes:

    class OtherUnicodeSubclass(unicode):
        pass

    class UnicodeOverload:

        def __unicode__(self):
            if False:
                return 10
            return OtherUnicodeSubclass()
    b = UnicodeOverload()
    v = unicode(b)
    if type(v) is unicode:
        print('Unicode overload ineffective (must not happen)')
    elif isinstance(v, unicode):
        print('Unicode overload successful.')
    else:
        print('Oops, must not happen.')

class OtherIntSubclass(int):
    pass

class IntOverload:

    def __int__(self):
        if False:
            return 10
        return OtherIntSubclass()
b = IntOverload()
v = int(b)
if type(v) is int:
    print('Int overload ineffective (must not happen)')
elif isinstance(v, int):
    print('Int overload successful.')
else:
    print('Oops, must not happen.')
if str is bytes:

    class OtherLongSubclass(long):
        pass

    class LongOverload:

        def __long__(self):
            if False:
                while True:
                    i = 10
            return OtherLongSubclass()
    b = LongOverload()
    v = long(b)
    if type(v) is long:
        print('Long overload ineffective (must not happen)')
    elif isinstance(v, long):
        print('Long overload successful.')
    else:
        print('Oops, must not happen.')

class OtherComplexSubclass(complex):
    pass

class ComplexOverload:

    def __complex__(self):
        if False:
            for i in range(10):
                print('nop')
        return OtherComplexSubclass()
b = ComplexOverload()
v = complex(b)
if type(v) is complex:
    print('Complex overload ineffective (must happen)')
elif isinstance(v, complex):
    print('Oops, must not happen.')
else:
    print('Oops, must not happen.')
print('Tests for abs():')
print(abs(-1000000 ** 10))
print(abs(len([1, 2, 3])))
print(abs(-100))
print(abs(float('nan')))
print('abs() with list:')
try:
    print(abs([1, 2]))
except Exception as e:
    print('caught ', repr(e))

def S1():
    if False:
        return 10
    print('Yielding 40')
    yield 40
    print('Yielding 60')
    yield 60
    print('Yielding 30')
    yield 30

def S2():
    if False:
        print('Hello World!')
    print('Yielding 60')
    yield 60
    print('Yielding 40')
    yield 40
    print('Yielding 30')
    yield 30
print(all((x > 42 for x in S1())))
print(all((x > 42 for x in S2())))
print('Disallowed all() without args:')
try:
    print(all())
except Exception as e:
    print('caught ', repr(e))
print('all() with float not iterable:')
try:
    print(all(1.0))
except Exception as e:
    print('caught ', repr(e))
print('all() with float type not iterable:')
try:
    print(any(float))
except Exception as e:
    print('caught ', repr(e))
print('all with compile time lists:')
print(all([None, None, None]))
print(all([None, 4, None]))
print(all([]))
print(all([0] * 20000))
print(all([0] * 255))
print('all with compile time ranges:')
print(all(range(1, 10000)))
print(all(range(10000)))
print(all(range(2, 999, 4)))
print('all with compile time strings and bytes:')
print(all('Nuitka rocks!'))
print(all('string'))
print(all('unicode'))
print(all(b'bytes'))
print(all(b'bytes\x00'))
print(any((x > 42 for x in S1())))
print(any((x > 42 for x in S2())))
print('Disallowed any() without args:')
try:
    print(any())
except Exception as e:
    print('caught ', repr(e))
print('any() with float not iterable:')
try:
    print(any(1.0))
except Exception as e:
    print('caught ', repr(e))
print('any() with float type not iterable:')
try:
    print(any(float))
except Exception as e:
    print('caught ', repr(e))
print('any() with sets:')
print(any(set([0, 1, 2, 3, 3])))
print(any({1: 'One', 2: 'Two'}))
print('any with compile time lists:')
print(any([None, None, None]))
print(any([None, 4, None]))
print(any([]))
print(any([0] * 20000))
print(any([0] * 255))
print('any with compile time ranges:')
print(any(range(1, 10000)))
print(any(range(10000)))
print(any(range(2, 999, 4)))
print('any with compile time strings and bytes:')
print(any('Nuitka rocks!'))
print(any('string'))
print(any('unicode'))
print(any(b'bytes'))
print(any(b'bytes\x00'))
print('Tests for zip():')
print(zip('abc', 'cdd'))
print(zip([1, 2, 3], [2, 3, 4]))
print(zip([1, 2, 3], 'String'))
try:
    zip(1, 'String')
except TypeError as e:
    print('Occurred', repr(e))
print(zip())
x = [(u, v) for (u, v) in zip(range(8), reversed(range(8)))]
print(x)
for v in zip([1, 2, 3], 'String'):
    print(v)
func = '{foo}'.format
print(func(foo='Foo'))