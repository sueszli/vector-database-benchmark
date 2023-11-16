"""
Test the more fancy stuff like:

* implicit reactions
* computed properties
* setting properties as callables to create implicit actions
* more

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
    foo = event.IntProp(settable=True)
    bar = event.IntProp(settable=True)

    @event.reaction('foo')
    def report1(self, *events):
        if False:
            for i in range(10):
                print('nop')
        print('foo', self.foo)

    @event.reaction('bar', mode='greedy')
    def report2(self, *events):
        if False:
            return 10
        print('bar', self.bar)

@run_in_both(MyObject1)
def test_reaction_greedy():
    if False:
        while True:
            i = 10
    '\n    normal greedy\n    bar 0\n    foo 0\n    -\n    foo 4\n    -\n    bar 4\n    -\n    foo 6\n    bar 6\n    foo 6\n    '
    m = MyObject1()
    print(m.report1.get_mode(), m.report2.get_mode())
    loop.iter()
    print('-')
    m.set_foo(3)
    m.set_foo(4)
    loop.iter()
    print('-')
    m.set_bar(3)
    m.set_bar(4)
    loop.iter()
    print('-')
    m.set_foo(4)
    m.set_bar(4)
    m.set_foo(5)
    m.set_bar(5)
    m.set_foo(6)
    m.set_bar(6)
    loop.iter()

class MyObject2(event.Component):
    foo = event.IntProp(settable=True)
    bar = event.IntProp(7, settable=True)

    @event.reaction
    def report(self, *events):
        if False:
            for i in range(10):
                print('nop')
        assert len(events) == 0
        print(self.foo, self.bar)

@run_in_both(MyObject2)
def test_reaction_auto1():
    if False:
        for i in range(10):
            print('nop')
    '\n    init\n    auto\n    0 7\n    4 7\n    4 4\n    end\n    '
    print('init')
    m = MyObject2()
    print(m.report.get_mode())
    loop.iter()
    m.set_foo(3)
    m.set_foo(4)
    loop.iter()
    m.set_bar(3)
    m.set_bar(24)
    m.set_bar(4)
    m.set_bar(4)
    loop.iter()
    m.set_foo(4)
    loop.iter()
    print('end')

class MyObject3(event.Component):
    foo = event.IntProp(settable=True)
    bar = event.IntProp(7, settable=True)

    @event.reaction('!spam', mode='auto')
    def report(self, *events):
        if False:
            return 10
        assert len(events) > 0
        print(self.foo, self.bar)

@run_in_both(MyObject3)
def test_reaction_auto2():
    if False:
        print('Hello World!')
    '\n    init\n    auto\n    0 7\n    4 7\n    4 4\n    4 4\n    end\n    '
    print('init')
    m = MyObject3()
    print(m.report.get_mode())
    loop.iter()
    m.set_foo(3)
    m.set_foo(4)
    loop.iter()
    m.set_bar(3)
    m.set_bar(24)
    m.set_bar(4)
    m.set_bar(4)
    loop.iter()
    m.emit('spam')
    loop.iter()
    m.set_foo(4)
    loop.iter()
    print('end')

class MyObject4(event.Component):
    bar = event.IntProp(7, settable=True)

@run_in_both(MyObject4)
def test_reaction_oneliner():
    if False:
        while True:
            i = 10
    '\n    7\n    2\n    xx\n    2\n    3\n    '
    m1 = MyObject4(bar=2)
    m2 = MyObject4(bar=lambda : m1.bar)
    loop.iter()
    print(m2.bar)
    loop.iter()
    print(m2.bar)
    print('xx')
    m1.set_bar(3)
    loop.iter()
    print(m2.bar)
    loop.iter()
    print(m2.bar)
run_tests_if_main()