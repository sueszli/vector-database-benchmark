"""
Test reactions.
"""
import gc
import sys
import weakref
from flexx.util.testing import run_tests_if_main, skipif, skip, raises
from flexx.event.both_tester import run_in_both, this_is_js
from flexx.util.logging import capture_log
from flexx import event
loop = event.loop
logger = event.logger

class MyObject1(event.Component):

    @event.reaction('!a')
    def r1(self, *events):
        if False:
            return 10
        print('r1:' + ' '.join([ev.type for ev in events]))

    @event.reaction('!a', '!b')
    def r2(self, *events):
        if False:
            i = 10
            return i + 15
        print('r2:' + ' '.join([ev.type for ev in events]))

    @event.reaction('!c')
    def r3(self, *events):
        if False:
            for i in range(10):
                print('nop')
        pass

@run_in_both(MyObject1)
def test_reaction_order1():
    if False:
        return 10
    '\n    r1:a a\n    r2:a a\n    r1:a a\n    r2:a a\n    '
    m = MyObject1()
    with loop:
        m.emit('a', {})
        m.emit('a', {})
        m.emit('c', {})
        m.emit('c', {})
        m.emit('a', {})
        m.emit('a', {})

@run_in_both(MyObject1)
def test_reaction_order2():
    if False:
        return 10
    '\n    r1:a a\n    r2:a a b b a a\n    r1:a a\n    r1:a\n    r2:a\n    '
    m = MyObject1()
    with loop:
        m.emit('a', {})
        m.emit('a', {})
        m.emit('b', {})
        m.emit('b', {})
        m.emit('a', {})
        m.emit('a', {})
        m.emit('c', {})
        m.emit('a', {})

@run_in_both(MyObject1)
def test_reaction_order3():
    if False:
        i = 10
        return i + 15
    '\n    r2:b a a\n    r1:a a\n    '
    m = MyObject1()
    with loop:
        m.emit('b', {})
        m.emit('a', {})
        m.emit('a', {})

@run_in_both(MyObject1)
def test_reaction_order4():
    if False:
        while True:
            i = 10
    '\n    r2:b a a\n    r1:a a\n    '
    m = MyObject1()
    with loop:
        m.emit('b', {})
        m.emit('a', {})
        m.emit('a', {})

class MyObject_labeled(event.Component):

    @event.reaction('!a')
    def r1(self, *events):
        if False:
            i = 10
            return i + 15
        print('r1 ' + ' '.join([ev.type for ev in events]))

    @event.reaction('!a:b')
    def r2(self, *events):
        if False:
            print('Hello World!')
        print('r2 ' + ' '.join([ev.type for ev in events]))

    @event.reaction('!a:a')
    def r3(self, *events):
        if False:
            for i in range(10):
                print('nop')
        print('r3 ' + ' '.join([ev.type for ev in events]))

@run_in_both(MyObject_labeled)
def test_reaction_labels1():
    if False:
        for i in range(10):
            print('nop')
    '\n    r3 a a\n    r2 a a\n    r1 a a\n    '
    m = MyObject_labeled()
    with loop:
        m.emit('a', {})
        m.emit('a', {})

class MyObject_init(event.Component):
    foo = event.IntProp(settable=True)
    bar = event.IntProp(7, settable=True)
    spam = event.IntProp(settable=False)

    @event.reaction('foo', 'bar')
    def _report(self, *events):
        if False:
            while True:
                i = 10
        print('r ' + ', '.join(['%s:%i->%i' % (ev.type, ev.old_value, ev.new_value) for ev in events]))

@run_in_both(MyObject_init)
def test_reaction_init1():
    if False:
        return 10
    '\n    0 7\n    iter\n    r bar:7->7, foo:0->0\n    0 7\n    end\n    '
    m = MyObject_init()
    print(m.foo, m.bar)
    print('iter')
    loop.iter()
    print(m.foo, m.bar)
    print('end')

@skipif(sys.version_info < (3, 6), reason='need ordered kwargs')
@run_in_both(MyObject_init)
def test_reaction_init2():
    if False:
        print('Hello World!')
    '\n    4 4\n    iter\n    r foo:4->4, bar:4->4\n    4 4\n    end\n    '
    m = MyObject_init(foo=4, bar=4)
    print(m.foo, m.bar)
    print('iter')
    loop.iter()
    print(m.foo, m.bar)
    print('end')

@run_in_both(MyObject_init)
def test_reaction_init3():
    if False:
        print('Hello World!')
    '\n    0 7\n    iter\n    r bar:7->7, foo:0->0, foo:0->2, bar:7->2\n    2 2\n    end\n    '
    m = MyObject_init()
    m.set_foo(2)
    m.set_bar(2)
    print(m.foo, m.bar)
    print('iter')
    loop.iter()
    print(m.foo, m.bar)
    print('end')

@skipif(sys.version_info < (3, 6), reason='need ordered kwargs')
@run_in_both(MyObject_init)
def test_reaction_init4():
    if False:
        while True:
            i = 10
    '\n    4 4\n    iter\n    r foo:4->4, bar:4->4, foo:4->2, bar:4->2\n    2 2\n    end\n    '
    m = MyObject_init(foo=4, bar=4)
    m.set_foo(2)
    m.set_bar(2)
    print(m.foo, m.bar)
    print('iter')
    loop.iter()
    print(m.foo, m.bar)
    print('end')

@run_in_both(MyObject_init)
def test_reaction_init_fail1():
    if False:
        while True:
            i = 10
    '\n    ? AttributeError\n    end\n    '
    try:
        m = MyObject_init(blabla=1)
    except AttributeError as err:
        logger.exception(err)
    try:
        m = MyObject_init(spam=1)
    except TypeError as err:
        logger.exception(err)
    print('end')

class MyObjectSub(MyObject1):

    @event.reaction('!a', '!b')
    def r2(self, *events):
        if False:
            print('Hello World!')
        super().r2(*events)
        print('-- r2 sub')

@run_in_both(MyObjectSub)
def test_reaction_overloading1():
    if False:
        for i in range(10):
            print('nop')
    '\n    r1:a a\n    r2:a a\n    -- r2 sub\n    r2:b b\n    -- r2 sub\n    '
    m = MyObjectSub()
    with loop:
        m.emit('a', {})
        m.emit('a', {})
    with loop:
        m.emit('b', {})
        m.emit('b', {})

class MyObject2(event.Component):
    foo = event.IntProp(settable=True)
    bar = event.IntProp(7, settable=True)

@run_in_both(MyObject2)
def test_reaction_using_react_func1():
    if False:
        i = 10
        return i + 15
    '\n    r bar:7->7, foo:0->0, foo:0->2, bar:7->2\n    r bar:7->7, foo:0->0, foo:0->3, bar:7->3\n    '

    def foo(*events):
        if False:
            for i in range(10):
                print('nop')
        print('r ' + ', '.join(['%s:%i->%i' % (ev.type, ev.old_value, ev.new_value) for ev in events]))
    m = MyObject2()
    m.reaction(foo, 'foo', 'bar')
    m.set_foo(2)
    m.set_bar(2)
    loop.iter()
    m = MyObject2()
    m.reaction('foo', 'bar', foo)
    m.set_foo(3)
    m.set_bar(3)
    loop.iter()

@run_in_both(MyObject2)
def test_reaction_using_react_func2():
    if False:
        for i in range(10):
            print('nop')
    '\n    r foo:0->2, bar:7->2\n    r foo:0->3, bar:7->3\n    '

    def foo(*events):
        if False:
            print('Hello World!')
        print('r ' + ', '.join(['%s:%i->%i' % (ev.type, ev.old_value, ev.new_value) for ev in events]))
    m = MyObject2()
    loop.iter()
    m.reaction(foo, 'foo', 'bar')
    m.set_foo(2)
    m.set_bar(2)
    loop.iter()
    m = MyObject2()
    loop.iter()
    m.reaction('foo', 'bar', foo)
    m.set_foo(3)
    m.set_bar(3)
    loop.iter()

@run_in_both(MyObject2)
def test_reaction_using_react_func3():
    if False:
        i = 10
        return i + 15
    '\n    r foo:0->2, bar:7->2\n    '

    class Foo:

        def foo(self, *events):
            if False:
                while True:
                    i = 10
            print('r ' + ', '.join(['%s:%i->%i' % (ev.type, ev.old_value, ev.new_value) for ev in events]))
    f = Foo()
    m = MyObject2()
    loop.iter()
    m.reaction(f.foo, 'foo', 'bar')
    m.set_foo(2)
    m.set_bar(2)
    loop.iter()

@run_in_both(MyObject2, js=False)
def test_reaction_using_react_func4():
    if False:
        print('Hello World!')
    '\n    r bar:7->7, foo:0->0, foo:0->2, bar:7->2\n    '
    m = MyObject2()

    @m.reaction('foo', 'bar')
    def foo(*events):
        if False:
            i = 10
            return i + 15
        print('r ' + ', '.join(['%s:%i->%i' % (ev.type, ev.old_value, ev.new_value) for ev in events]))
    m.set_foo(2)
    m.set_bar(2)
    loop.iter()

def test_reaction_builtin_function():
    if False:
        return 10

    class Foo(event.Component):
        pass
    foo = Foo()
    foo.reaction('!bar', print)

def test_reaction_as_decorator_of_other_cls():
    if False:
        print('Hello World!')

    class C1(event.Component):
        foo = event.AnyProp(settable=True)
    c1 = C1()

    class C2(event.Component):

        @c1.reaction('foo')
        def on_foo(self, *events):
            if False:
                return 10
            print('x')
            self.xx = events[-1].new_value
    c2 = C2()
    loop.iter()
    c1.set_foo(3)
    loop.iter()
    assert c2.xx == 3

@run_in_both(MyObject1)
def test_reaction_calling():
    if False:
        return 10
    '\n    r1:\n    r2:\n    end\n    '
    m = MyObject1()
    m.r1()
    m.r2()
    loop.iter()
    print('end')

def test_reaction_exceptions1():
    if False:
        while True:
            i = 10
    m = event.Component()

    @m.reaction('!foo')
    def handle_foo(*events):
        if False:
            for i in range(10):
                print('nop')
        1 / 0
    m.emit('foo', {})
    sys.last_traceback = None
    assert sys.last_traceback is None
    loop.iter()
    loop.iter()
    if sys.version_info[0] >= 3:
        assert sys.last_traceback
    with raises(ZeroDivisionError):
        handle_foo()

def test_reaction_exceptions2():
    if False:
        for i in range(10):
            print('nop')

    class Foo(event.Component):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.bar = event.Component()
            self.bars = [self.bar]
    f = Foo()

    @f.reaction('bars*.spam')
    def handle_foo(*events):
        if False:
            while True:
                i = 10
        pass
    with raises(RuntimeError) as err:

        @f.reaction('bar*.spam')
        def handle_foo(*events):
            if False:
                return 10
            pass
    assert 'not a tuple' in str(err.value)

def test_reaction_decorator_fails():
    if False:
        for i in range(10):
            print('nop')

    class Foo:

        def foo(self, *events):
            if False:
                return 10
            pass
    f = Foo()

    def foo(*events):
        if False:
            while True:
                i = 10
        pass
    with raises(TypeError):
        event.reaction()
    with raises(TypeError):
        event.reaction('!foo')(3)
    with raises(TypeError):
        event.reaction('!foo')(foo)
    with raises(TypeError):
        event.reaction('!foo')(f.foo)

def test_reaction_descriptor_has_local_connection_strings():
    if False:
        return 10
    m = MyObject1()
    assert m.__class__.r1.local_connection_strings == ['!a']

@run_in_both(MyObject1)
def test_reaction_meta():
    if False:
        for i in range(10):
            print('nop')
    "\n    True\n    r1\n    [['!a', ['a:r1']]]\n    [['!a', ['a:r2']], ['!b', ['b:r2']]]\n    "
    m = MyObject1()
    print(hasattr(m.r1, 'dispose'))
    print(m.r1.get_name())
    print([list(x) for x in m.r1.get_connection_info()])
    print([list(x) for x in m.r2.get_connection_info()])

@run_in_both(MyObject1)
def test_reaction_not_settable():
    if False:
        return 10
    '\n    fail AttributeError\n    '
    m = MyObject1()
    try:
        m.r1 = 3
    except AttributeError:
        print('fail AttributeError')

def test_reaction_python_only():
    if False:
        i = 10
        return i + 15
    m = MyObject1()
    with raises(TypeError):
        event.reaction(3)
    with raises(TypeError):
        event.reaction(isinstance)
    assert isinstance(m.r1, event._reaction.Reaction)
    with raises(AttributeError):
        m.r1 = 3
    with raises(AttributeError):
        del m.r1
    assert 'reaction' in repr(m.__class__.r1).lower()
    assert 'reaction' in repr(m.r1).lower()
    assert 'r1' in repr(m.r1)
run_tests_if_main()