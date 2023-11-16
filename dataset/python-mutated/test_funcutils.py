from boltons.funcutils import copy_function, total_ordering, format_invocation, InstancePartial, CachedInstancePartial, noop

class Greeter(object):

    def __init__(self, greeting):
        if False:
            i = 10
            return i + 15
        self.greeting = greeting

    def greet(self, excitement='.'):
        if False:
            i = 10
            return i + 15
        return self.greeting.capitalize() + excitement
    partial_greet = InstancePartial(greet, excitement='!')
    cached_partial_greet = CachedInstancePartial(greet, excitement='...')

    def native_greet(self):
        if False:
            print('Hello World!')
        return self.greet(';')

class SubGreeter(Greeter):
    pass

def test_partials():
    if False:
        while True:
            i = 10
    g = SubGreeter('hello')
    assert g.greet() == 'Hello.'
    assert g.native_greet() == 'Hello;'
    assert g.partial_greet() == 'Hello!'
    assert g.cached_partial_greet() == 'Hello...'
    assert CachedInstancePartial(g.greet, excitement='s')() == 'Hellos'
    g.native_greet = 'native reassigned'
    assert g.native_greet == 'native reassigned'
    g.partial_greet = 'partial reassigned'
    assert g.partial_greet == 'partial reassigned'
    g.cached_partial_greet = 'cached_partial reassigned'
    assert g.cached_partial_greet == 'cached_partial reassigned'

def test_copy_function():
    if False:
        print('Hello World!')

    def callee():
        if False:
            return 10
        return 1
    callee_copy = copy_function(callee)
    assert callee is not callee_copy
    assert callee() == callee_copy()

def test_total_ordering():
    if False:
        print('Hello World!')

    @total_ordering
    class Number(object):

        def __init__(self, val):
            if False:
                return 10
            self.val = int(val)

        def __gt__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            return self.val > other

        def __eq__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            return self.val == other
    num = Number(3)
    assert num > 0
    assert num == 3
    assert num < 5
    assert num >= 2
    assert num != 1

def test_format_invocation():
    if False:
        for i in range(10):
            print('nop')
    assert format_invocation('d') == 'd()'
    assert format_invocation('f', ('a', 'b')) == "f('a', 'b')"
    assert format_invocation('g', (), {'x': 'y'}) == "g(x='y')"
    assert format_invocation('h', ('a', 'b'), {'x': 'y', 'z': 'zz'}) == "h('a', 'b', x='y', z='zz')"

def test_noop():
    if False:
        while True:
            i = 10
    assert noop() is None
    assert noop(1, 2) is None
    assert noop(a=1, b=2) is None