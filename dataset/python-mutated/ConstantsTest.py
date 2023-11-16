""" Playing around with constants only. """
from __future__ import print_function
try:
    long
except NameError:
    long = int

def displayDict(d):
    if False:
        i = 10
        return i + 15
    result = '{'
    first = True
    for (key, value) in sorted(d.items()):
        if not first:
            result += ','
        result += '%s: %s' % (repr(key), repr(value))
        first = False
    result += '}'
    return result
print('A bunch of constants and their representation:')
for value in (0, 3, -4, 17, 'hey', (0,), 0.0, -0.0):
    print(value, ':', repr(value))
print('Comparing constants, optimizable:')
print(1 == 0)
print('Representation of long constants:')
a = long(0)
print(repr(long(0)), repr(a) == '0L')
print('Identity of empty dictionary constants:')
print({} is {})
a = ({}, [])
a[0][1] = 2
a[1].append(3)
print('Mutable list and dict inside an immutable tuple:')
print(a)
print('Empty list and dict are hopefully unchanged:')
print(({}, []))

def argChanger(a):
    if False:
        return 10
    a[0][1] = 2
    a[1].append(3)
    return a
print('Mutable list and dict inside an immutable tuple as arguments:')
print(argChanger(({}, [])))
print('Empty list and dict are hopefully still unchanged:')
print(({}, []))
print('Set constants:')
print(set(['foo']))

def mutableConstantChanger():
    if False:
        while True:
            i = 10
    a = ([1, 2], [3])
    print('Start out with value:')
    print(a)
    a[1].append(5)
    print('Changed to value:')
    print(a)
    d = {'l': [], 'm': []}
    print('Start out with value:')
    print(d)
    d['l'].append(7)
    print('Changed to value:')
    print(d)
    spec = dict(qual=[], storage=set(), type=[], function=set(), q=1)
    spec['type'].insert(0, 2)
    spec['storage'].add(3)
    print('Dictionary created from dict built-in.')
    print(sorted(spec))
mutableConstantChanger()
print('Redo constant changes, to catch corruptions:')
mutableConstantChanger()

def defaultKeepsIdentity(arg='str_value'):
    if False:
        print('Hello World!')
    print('Default constant values are still shared if immutable:', arg is 'str_value')
defaultKeepsIdentity()

def dd(**d):
    if False:
        print('Hello World!')
    return d

def f():
    if False:
        return 10

    def one():
        if False:
            i = 10
            return i + 15
        print('one')

    def two():
        if False:
            for i in range(10):
                print('nop')
        print('two')
    a = dd(qual=one(), storage=two(), type=[], function=[])
    print('f mutable', displayDict(a))
    a = dd(qual=1, storage=2, type=3, function=4)
    print('f immutable', displayDict(a))
    x = {'p': 7}
    a = dd(qual=[], storage=[], type=[], function=[], **x)
    print('f ext mutable', displayDict(a))
    x = {'p': 8}
    a = dd(qual=1, storage=2, type=3, function=4, **x)
    print('f ext immutable', displayDict(a))
f()
x = {}
x['function'] = []
x['type'] = []
x['storage'] = []
x['qual'] = []
print('Manual built dictionary:', x)
x = {}
x['function'] = 1
x['type'] = 2
x['storage'] = 3
x['qual'] = 4
print('Manual built dictionary:', x)
d = {'qual': [], 'storage': [], 'type2': [], 'function': []}
print('Mutable values dictionary constant:', displayDict(d))
d = {'qual': 1, 'storage': 2, 'type2': 3, 'function': 4}
print('Immutable values dictionary constant:', displayDict(d))
min_signed_int = int(-(2 ** (8 * 8 - 1) - 1) - 1)
print('Small int:', min_signed_int, type(min_signed_int))
min_signed_int = int(-(2 ** (8 * 4 - 1) - 1) - 1)
print('Small int', min_signed_int, type(min_signed_int))
min_signed_long = long(-(2 ** (8 * 8 - 1) - 1) - 1)
print('Small long', min_signed_long, type(min_signed_long))
min_signed_long = long(-(2 ** (8 * 4 - 1) - 1) - 1)
print('Small long', min_signed_long, type(min_signed_long))
try:
    type_prepare = type.__prepare__
except AttributeError:
    print('Python2 has no type.__prepare__')
else:
    print('Type prepare', type_prepare)