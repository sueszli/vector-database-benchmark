"""
Test event emitters.
"""
import sys
from flexx.util.testing import run_tests_if_main, skipif, skip, raises
from flexx.event.both_tester import run_in_both, this_is_js
from flexx import event
loop = event.loop

class MyObject(event.Component):

    @event.emitter
    def foo(self, v):
        if False:
            while True:
                i = 10
        if not isinstance(v, (int, float)):
            raise TypeError('Foo emitter expects a number.')
        return dict(value=float(v))

    @event.emitter
    def bar(self, v):
        if False:
            for i in range(10):
                print('nop')
        return dict(value=float(v) + 1)

    @event.emitter
    def wrong(self, v):
        if False:
            return 10
        return float(v)

    @event.reaction('foo')
    def on_foo(self, *events):
        if False:
            while True:
                i = 10
        print('foo', ', '.join([str(ev.value) for ev in events]))

    @event.reaction('bar')
    def on_bar(self, *events):
        if False:
            i = 10
            return i + 15
        print('bar', ', '.join([str(ev.value) for ev in events]))

class MyObject2(MyObject):

    @event.emitter
    def bar(self, v):
        if False:
            return 10
        return super().bar(v + 10)

class MyObject3(MyObject):

    @event.reaction('foo', mode='greedy')
    def on_foo(self, *events):
        if False:
            i = 10
            return i + 15
        print('foo', ', '.join([str(ev.value) for ev in events]))

    @event.reaction('bar', mode='greedy')
    def on_bar(self, *events):
        if False:
            for i in range(10):
                print('nop')
        print('bar', ', '.join([str(ev.value) for ev in events]))

@run_in_both(MyObject)
def test_emitter_ok():
    if False:
        return 10
    '\n    foo 3.2\n    foo 3.2, 3.3\n    bar 4.8, 4.9\n    bar 4.9\n    '
    m = MyObject()
    with loop:
        m.foo(3.2)
    with loop:
        m.foo(3.2)
        m.foo(3.3)
    with loop:
        m.bar(3.8)
        m.bar(3.9)
    with loop:
        m.bar(3.9)

@run_in_both(MyObject2)
def test_emitter_overloading():
    if False:
        return 10
    '\n    bar 14.2, 15.5\n    '
    m = MyObject2()
    with loop:
        m.bar(3.2)
        m.bar(4.5)

@run_in_both(MyObject)
def test_emitter_order():
    if False:
        for i in range(10):
            print('nop')
    '\n    foo 3.1, 3.2\n    bar 6.3, 6.4\n    foo 3.5, 3.6\n    bar 6.7, 6.8\n    bar 6.9, 6.9\n    '
    m = MyObject()
    with loop:
        m.foo(3.1)
        m.foo(3.2)
        m.bar(5.3)
        m.bar(5.4)
        m.foo(3.5)
        m.foo(3.6)
        m.bar(5.7)
        m.bar(5.8)
    with loop:
        m.bar(5.9)
        m.bar(5.9)

@run_in_both(MyObject3)
def test_emitter_order_greedy():
    if False:
        i = 10
        return i + 15
    '\n    foo 3.1, 3.2, 3.5, 3.6\n    bar 6.3, 6.4, 6.7, 6.8\n    bar 6.9, 6.9\n    '
    m = MyObject3()
    with loop:
        m.foo(3.1)
        m.foo(3.2)
        m.bar(5.3)
        m.bar(5.4)
        m.foo(3.5)
        m.foo(3.6)
        m.bar(5.7)
        m.bar(5.8)
    with loop:
        m.bar(5.9)
        m.bar(5.9)

@run_in_both(MyObject)
def test_emitter_fail():
    if False:
        while True:
            i = 10
    '\n    fail TypeError\n    fail TypeError\n    fail ValueError\n    '
    m = MyObject()
    try:
        m.wrong(1.1)
    except TypeError:
        print('fail TypeError')
    try:
        m.foo('bla')
    except TypeError:
        print('fail TypeError')
    try:
        m.emit('bla:x')
    except ValueError:
        print('fail ValueError')

@run_in_both(MyObject)
def test_emitter_not_settable():
    if False:
        while True:
            i = 10
    '\n    fail AttributeError\n    '
    m = MyObject()
    try:
        m.foo = 3
    except AttributeError:
        print('fail AttributeError')

def test_emitter_python_only():
    if False:
        while True:
            i = 10
    m = MyObject()
    with raises(TypeError):
        event.emitter(3)
    if '__pypy__' in sys.builtin_module_names:
        pass
    else:
        with raises(TypeError):
            event.emitter(isinstance)
    assert isinstance(m.foo, event._emitter.Emitter)
    with raises(AttributeError):
        m.foo = 3
    with raises(AttributeError):
        del m.foo
    assert 'emitter' in repr(m.__class__.foo).lower()
    assert 'emitter' in repr(m.foo).lower()
    assert 'foo' in repr(m.foo)
run_tests_if_main()