from __future__ import print_function

def separator():
    if False:
        while True:
            i = 10
    print('*' * 80)

def dictOrderCheck():
    if False:
        i = 10
        return i + 15

    def key1():
        if False:
            while True:
                i = 10
        print('key1 called')
        return 1

    def key2():
        if False:
            for i in range(10):
                print('nop')
        print('key2 called')
        return 2

    def value1():
        if False:
            return 10
        print('value1 called')
        return 11

    def value2():
        if False:
            i = 10
            return i + 15
        print('value2 called')
        return 22
    print('Checking order of calls in dictionary creation from callables:')
    print({key1(): value1(), key2(): value2()})
    try:
        (1 / 0)[1j / 0] = 1.0 / 0
    except ZeroDivisionError as e:
        print('Expected exception caught:', repr(e))
    try:
        (1 / 0)[1.0 / 0] = 1
    except ZeroDivisionError as e:
        print('Expected exception caught:', repr(e))
    try:
        (1 / 0)[1] = 1.0 / 0
    except ZeroDivisionError as e:
        print('Expected exception caught:', repr(e))

def listOrderCheck():
    if False:
        while True:
            i = 10

    def value1():
        if False:
            while True:
                i = 10
        print('value1 called')
        return 11

    def value2():
        if False:
            i = 10
            return i + 15
        print('value2 called')
        return 22
    print([value1(), value2()])

def sliceOrderCheck():
    if False:
        for i in range(10):
            print('nop')
    print('Slices:')
    d = list(range(10))

    def lvalue():
        if False:
            print('Hello World!')
        print('lvalue', end=' ')
        return d

    def rvalue():
        if False:
            while True:
                i = 10
        print('rvalue', end=' ')
        return range(2)

    def rvalue4():
        if False:
            return 10
        print('rvalue', end=' ')
        return range(4)

    def low():
        if False:
            for i in range(10):
                print('nop')
        print('low', end=' ')
        return 0

    def high():
        if False:
            print('Hello World!')
        print('high', end=' ')
        return 4

    def step():
        if False:
            for i in range(10):
                print('nop')
        print('step', end=' ')
        return 2
    print('Complex slice lookup:', end=' ')
    print(lvalue()[low():high():step()])
    print('Complex slice assignment:', end=' ')
    lvalue()[low():high():step()] = rvalue()
    print(d)
    print('Complex slice del:', end=' ')
    del lvalue()[low():high():step()]
    print(d)
    print('Complex inplace slice operation', end=' ')
    print(d)
    d = list(range(10))
    print('Simple slice lookup', end=' ')
    print(lvalue()[low():high()])
    print('Simple slice assignment', end=' ')
    lvalue()[3 + low():3 + high()] = rvalue()
    print(d)
    print('Simple slice del', end=' ')
    del lvalue()[3 + low():3 + high()]
    print(d)
    print('Simple inplace slice operation', end=' ')
    lvalue()[low():high()] += rvalue4()
    print(d)

def subscriptOrderCheck():
    if False:
        while True:
            i = 10
    print('Subscripts:')
    d = {}

    def lvalue():
        if False:
            while True:
                i = 10
        print('lvalue', end=' ')
        return d

    def rvalue():
        if False:
            print('Hello World!')
        print('rvalue', end=' ')
        return 2

    def subscript():
        if False:
            for i in range(10):
                print('nop')
        print('subscript', end=' ')
        return 1
    print('Assigning subscript:')
    lvalue()[subscript()] = rvalue()
    print(d)
    print('Lookup subscript:')
    print(lvalue()[subscript()])
    print('Deleting subscript:')
    del lvalue()[subscript()]
    print(d)

def attributeOrderCheck():
    if False:
        for i in range(10):
            print('nop')

    def lvalue():
        if False:
            i = 10
            return i + 15
        print('lvalue', end=' ')
        return lvalue

    def rvalue():
        if False:
            return 10
        print('rvalue', end=' ')
        return 2
    print('Attribute assignment order:')
    lvalue().xxx = rvalue()
    print('Assigned was indeed:', lvalue.xxx)
    print('Checking attribute assignment to unassigned value from unassigned:')
    try:
        undefined_global_zzz.xxx = undefined_global_yyy
    except Exception as e:
        print('Expected exception caught:', repr(e))
    else:
        assert False
    try:
        (1 / 0).x = 1.0 / 0
    except ZeroDivisionError as e:
        print('Expected exception caught:', repr(e))

def compareOrderCheck():
    if False:
        return 10

    def lvalue():
        if False:
            i = 10
            return i + 15
        print('lvalue', end=' ')
        return 1

    def rvalue():
        if False:
            while True:
                i = 10
        print('rvalue', end=' ')
        return 2
    print('Comparisons:')
    print('==', lvalue() == rvalue())
    print('<=', lvalue() <= rvalue())
    print('>=', lvalue() >= rvalue())
    print('!=', lvalue() != rvalue())
    print('>', lvalue() > rvalue())
    print('<', lvalue() < rvalue())
    print('Comparison used in bool context:')
    print('==', 'yes' if lvalue() == rvalue() else 'no')
    print('<=', 'yes' if lvalue() <= rvalue() else 'no')
    print('>=', 'yes' if lvalue() >= rvalue() else 'no')
    print('!=', 'yes' if lvalue() != rvalue() else 'no')
    print('>', 'yes' if lvalue() > rvalue() else 'no')
    print('<', 'yes' if lvalue() < rvalue() else 'no')

def operatorOrderCheck():
    if False:
        return 10

    def left():
        if False:
            while True:
                i = 10
        print('left operand', end=' ')
        return 1

    def middle():
        if False:
            return 10
        print('middle operand', end=' ')
        return 3

    def right():
        if False:
            while True:
                i = 10
        print('right operand', end=' ')
        return 2
    print('Operations:')
    print('+', left() + middle() + right())
    print('-', left() - middle() - right())
    print('*', left() * middle() * right())
    print('/', left() / middle() / right())
    print('%', left() % middle() % right())
    print('**', left() ** middle() ** right())

def generatorOrderCheck():
    if False:
        for i in range(10):
            print('nop')
    print('Generators:')

    def default1():
        if False:
            return 10
        print('default1', end=' ')
        return 1

    def default2():
        if False:
            while True:
                i = 10
        print('default2', end=' ')
        return 2

    def default3():
        if False:
            return 10
        print('default3', end=' ')
        return 3

    def value(x):
        if False:
            print('Hello World!')
        print('value', x, end=' ')
        return x

    def generator(a=default1(), b=default2(), c=default3()):
        if False:
            i = 10
            return i + 15
        print('generator entry')
        yield value(a)
        yield value(b)
        yield value(c)
        print('generator exit')
    result = list(generator())
    print('Result', result)

def classOrderCheck():
    if False:
        print('Hello World!')
    print('Checking order of class constructions:')

    class B1:
        pass

    class B2:
        pass

    def base1():
        if False:
            while True:
                i = 10
        print('base1', end=' ')
        return B1

    def base2():
        if False:
            return 10
        print('base2', end=' ')
        return B2

    def deco1(cls):
        if False:
            for i in range(10):
                print('nop')
        print('deco1', end=' ')
        return cls

    def deco2(cls):
        if False:
            return 10
        print('deco2')
        return B2

    @deco2
    @deco1
    class X(base1(), base2()):
        print('class body', end=' ')
    print

def inOrderCheck():
    if False:
        return 10
    print('Checking order of in operator:')

    def container():
        if False:
            return 10
        print('container', end=' ')
        return [3]

    def searched():
        if False:
            for i in range(10):
                print('nop')
        print('searched', end=' ')
        return 3
    print('in:', searched() in container())
    print('not in:', searched() not in container())

def unpackOrderCheck():
    if False:
        return 10

    class TraceRelease:

        def __init__(self, value):
            if False:
                i = 10
                return i + 15
            self.value = value

        def __del__(self):
            if False:
                return 10
            print('Deleting iteration value', self.value)
            pass
    print('Unpacking values:')

    class Iterable:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.consumed = 2

        def __iter__(self):
            if False:
                i = 10
                return i + 15
            return Iterable()

        def __del__(self):
            if False:
                return 10
            print('Deleted iterable with', self.consumed)

        def next(self):
            if False:
                print('Hello World!')
            print('Next with state', self.consumed)
            if self.consumed:
                self.consumed -= 1
            else:
                raise StopIteration
            return TraceRelease(self.consumed)
        __next__ = next
    iterable = Iterable()

    class RejectAttributeOwnership:

        def __setattr__(self, key, value):
            if False:
                print('Hello World!')
            print('Setting', key, value.value)
    try:
        (RejectAttributeOwnership().x, RejectAttributeOwnership().y) = (a, b) = iterable
    except Exception as e:
        print('Caught', repr(e))
    return (a, b)

def superOrderCheck():
    if False:
        i = 10
        return i + 15
    print('Built-in super:')
    try:
        super(zzz, xxx)
    except Exception as e:
        print('Expected exception caught super 2', repr(e))

def isinstanceOrderCheck():
    if False:
        return 10
    print('Built-in isinstance:')
    try:
        isinstance(zzz, xxx)
    except Exception as e:
        print('Expected exception caught isinstance 2', repr(e))

def rangeOrderCheck():
    if False:
        print('Hello World!')
    print('Built-in range:')
    try:
        range(zzz, yyy, xxx)
    except Exception as e:
        print('Expected exception caught range 3', repr(e))
    try:
        range(zzz, xxx)
    except Exception as e:
        print('Expected exception caught range 2', repr(e))

def importOrderCheck():
    if False:
        print('Hello World!')
    print('Built-in __import__:')

    def name():
        if False:
            i = 10
            return i + 15
        print('name', end=' ')

    def globals():
        if False:
            print('Hello World!')
        print('globals', end=' ')

    def locals():
        if False:
            i = 10
            return i + 15
        print('locals', end=' ')

    def fromlist():
        if False:
            for i in range(10):
                print('nop')
        print('fromlist', end=' ')

    def level():
        if False:
            return 10
        print('level')
    try:
        __import__(name(), globals(), locals(), fromlist(), level())
    except Exception as e:
        print('Expected exception caught __import__ 5', repr(e))

def hasattrOrderCheck():
    if False:
        for i in range(10):
            print('nop')
    print('Built-in hasattr:')
    try:
        hasattr(zzz, yyy)
    except Exception as e:
        print('Expected exception caught hasattr', repr(e))

def getattrOrderCheck():
    if False:
        for i in range(10):
            print('nop')
    print('Built-in getattr:')
    try:
        getattr(zzz, yyy)
    except Exception as e:
        print('Expected exception caught getattr 2', repr(e))
    try:
        getattr(zzz, yyy, xxx)
    except Exception as e:
        print('Expected exception caught getattr 3', repr(e))

    def default():
        if False:
            return 10
        print('default used')
    print('Default side effects:', end=' ')
    print(getattr(1, 'real', default()))

def typeOrderCheck():
    if False:
        while True:
            i = 10
    print('Built-in type:')
    try:
        type(zzz, yyy, xxx)
    except Exception as e:
        print('Expected exception caught type 3', repr(e))

def iterOrderCheck():
    if False:
        while True:
            i = 10
    print('Built-in iter:')
    try:
        iter(zzz, xxx)
    except Exception as e:
        print('Expected exception caught iter 2', repr(e))

def openOrderCheck():
    if False:
        for i in range(10):
            print('nop')
    print('Built-in open:')
    try:
        open(zzz, yyy, xxx)
    except Exception as e:
        print('Expected exception caught open 3', repr(e))

def unicodeOrderCheck():
    if False:
        for i in range(10):
            print('nop')
    print('Built-in unicode:')
    try:
        unicode(zzz, yyy, xxx)
    except Exception as e:
        print('Expected exception caught unicode', repr(e))

def longOrderCheck():
    if False:
        return 10
    print('Built-in long:')
    try:
        long(zzz, xxx)
    except Exception as e:
        print('Expected exception caught long 2', repr(e))

def intOrderCheck():
    if False:
        while True:
            i = 10
    print('Built-in int:')
    try:
        int(zzz, xxx)
    except Exception as e:
        print('Expected exception caught int', repr(e))

def nextOrderCheck():
    if False:
        return 10
    print('Built-in next:')
    try:
        next(zzz, xxx)
    except Exception as e:
        print('Expected exception caught next 2', repr(e))

def callOrderCheck():
    if False:
        i = 10
        return i + 15
    print('Checking nested call arguments:')

    class A:

        def __del__(self):
            if False:
                for i in range(10):
                    print('nop')
            print('Doing del inner object')

    def check(obj):
        if False:
            return 10
        print('Outer function')

    def p(obj):
        if False:
            return 10
        print('Inner function')
    check(p(A()))

def boolOrderCheck():
    if False:
        while True:
            i = 10
    print('Checking order of or/and arguments:')

    class A(int):

        def __init__(self, value):
            if False:
                while True:
                    i = 10
            self.value = value

        def __del__(self):
            if False:
                i = 10
                return i + 15
            print('Doing del of %s' % self)

        def __bool__(self):
            if False:
                print('Hello World!')
            print('Test of %s' % self)
            return self.value != 0
        __nonzero__ = __bool__

        def __str__(self):
            if False:
                while True:
                    i = 10
            return '<%s %r>' % (self.__class__.__name__, self.value)

    class B(A):
        pass

    class C(A):
        pass
    print('Two arguments, A or B:')
    for a in range(2):
        for b in range(2):
            print('Case %d or %d' % (a, b))
            r = A(a) or B(b)
            print(r)
            del r
    if True:
        print('Three arguments, A or B or C:')
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    print('Case %d or %d or %d' % (a, b, c))
                    r = A(a) or B(b) or C(c)
                    print(r)
                    del r
    print('Two arguments, A and B:')
    for a in range(2):
        for b in range(2):
            print('Case %d and %d' % (a, b))
            r = A(a) and B(b)
            print(r)
            del r
    if True:
        print('Three arguments, A and B and C:')
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    print('Case %d and %d and %d' % (a, b, c))
                    r = A(a) and B(b) and C(c)
                    print(r)
                    del r

def comparisonChainOrderCheck():
    if False:
        i = 10
        return i + 15
    print('Checking order of comparison chains:')

    class A(int):

        def __init__(self, value):
            if False:
                while True:
                    i = 10
            self.value = value

        def __del__(self):
            if False:
                i = 10
                return i + 15
            print('Doing del of %s' % self)

        def __le__(self, other):
            if False:
                i = 10
                return i + 15
            print('Test of %s <= %s' % (self, other))
            return self.value <= other.value

        def __str__(self):
            if False:
                i = 10
                return i + 15
            return '<%s %r>' % (self.__class__.__name__, self.value)

    class B(A):
        pass

    class C(A):
        pass
    if False:
        print('Three arguments, A <= B <= C:')
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    print('Case %d <= %d <= %d' % (a, b, c))
                    r = A(a) <= B(b) <= C(c)
                    print(r)
                    del r
dictOrderCheck()
separator()
listOrderCheck()
separator()
subscriptOrderCheck()
separator()
attributeOrderCheck()
separator()
operatorOrderCheck()
separator()
compareOrderCheck()
separator()
sliceOrderCheck()
separator()
generatorOrderCheck()
separator()
classOrderCheck()
separator()
inOrderCheck()
separator()
unpackOrderCheck()
separator()
superOrderCheck()
separator()
isinstanceOrderCheck()
separator()
rangeOrderCheck()
separator()
importOrderCheck()
separator()
hasattrOrderCheck()
separator()
getattrOrderCheck()
separator()
typeOrderCheck()
separator()
iterOrderCheck()
separator()
openOrderCheck()
separator()
unicodeOrderCheck()
separator()
nextOrderCheck()
separator()
longOrderCheck()
separator()
intOrderCheck()
separator()
callOrderCheck()
separator()
boolOrderCheck()
separator()
comparisonChainOrderCheck()