"""
Test reactions more wrt dynamism.
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

class Node(event.Component):
    val = event.IntProp(settable=True)
    parent = event.ComponentProp(settable=True)
    children = event.TupleProp(settable=True)

    @event.reaction('parent.val')
    def handle_parent_val(self, *events):
        if False:
            return 10
        xx = []
        for ev in events:
            if self.parent:
                xx.append(self.parent.val)
            else:
                xx.append(None)
        print('parent.val ' + ', '.join([str(x) for x in xx]))

    @event.reaction('children*.val')
    def handle_children_val(self, *events):
        if False:
            i = 10
            return i + 15
        xx = []
        for ev in events:
            if isinstance(ev.new_value, (int, float)):
                xx.append(ev.new_value)
            else:
                xx.append(None)
        print('children.val ' + ', '.join([str(x) for x in xx]))

@run_in_both(Node)
def test_dynamism1():
    if False:
        return 10
    '\n    parent.val 17\n    parent.val 18\n    parent.val 29\n    done\n    '
    n = Node()
    n1 = Node()
    n2 = Node()
    loop.iter()
    with loop:
        n.set_parent(n1)
        n.set_val(42)
    with loop:
        n1.set_val(17)
        n2.set_val(27)
    with loop:
        n1.set_val(18)
        n2.set_val(28)
    with loop:
        n.set_parent(n2)
    with loop:
        n1.set_val(19)
        n2.set_val(29)
    with loop:
        n.set_parent(None)
    with loop:
        n1.set_val(11)
        n2.set_val(21)
    print('done')

@run_in_both(Node)
def test_dynamism2a():
    if False:
        return 10
    '\n    parent.val 17\n    parent.val 18\n    parent.val 29\n    [17, 18, 29]\n    '
    n = Node()
    n1 = Node()
    n2 = Node()
    res = []

    def func(*events):
        if False:
            i = 10
            return i + 15
        for ev in events:
            if n.parent:
                res.append(n.parent.val)
            else:
                res.append(None)
    n.reaction(func, 'parent.val')
    loop.iter()
    with loop:
        n.set_parent(n1)
        n.set_val(42)
    with loop:
        n1.set_val(17)
        n2.set_val(27)
    with loop:
        n1.set_val(18)
        n2.set_val(28)
    with loop:
        n.set_parent(n2)
    with loop:
        n1.set_val(19)
        n2.set_val(29)
    with loop:
        n.set_parent(None)
    with loop:
        n1.set_val(11)
        n2.set_val(21)
    print(res)

@run_in_both(Node)
def test_dynamism2b():
    if False:
        for i in range(10):
            print('nop')
    '\n    parent.val 17\n    parent.val 18\n    parent.val 29\n    [None, None, 17, 18, None, 29, None]\n    '
    n = Node()
    n1 = Node()
    n2 = Node()
    res = []

    def func(*events):
        if False:
            while True:
                i = 10
        for ev in events:
            if ev.type == 'val':
                res.append(n.parent.val)
            else:
                res.append(None)
    handler = n.reaction(func, 'parent', 'parent.val')
    loop.iter()
    with loop:
        n.set_parent(n1)
        n.set_val(42)
    with loop:
        n1.set_val(17)
        n2.set_val(27)
    with loop:
        n1.set_val(18)
        n2.set_val(28)
    with loop:
        n.set_parent(n2)
    with loop:
        n1.set_val(19)
        n2.set_val(29)
    with loop:
        n.set_parent(None)
    with loop:
        n1.set_val(11)
        n2.set_val(21)
    print(res)

@run_in_both(Node)
def test_dynamism3():
    if False:
        print('Hello World!')
    '\n    children.val 17, 27\n    children.val 18, 28\n    children.val 29\n    done\n    '
    n = Node()
    n1 = Node()
    n2 = Node()
    loop.iter()
    with loop:
        n.set_children((n1, n2))
        n.set_val(42)
    with loop:
        n1.set_val(17)
        n2.set_val(27)
    with loop:
        n1.set_val(18)
        n2.set_val(28)
    with loop:
        n.set_children((n2,))
    with loop:
        n1.set_val(19)
        n2.set_val(29)
    with loop:
        n.set_children(())
    with loop:
        n1.set_val(11)
        n2.set_val(21)
    print('done')

@run_in_both(Node)
def test_dynamism4a():
    if False:
        print('Hello World!')
    '\n    children.val 17, 27\n    children.val 18, 28\n    children.val 29\n    [17, 27, 18, 28, 29]\n    '
    n = Node()
    n1 = Node()
    n2 = Node()
    res = []

    def func(*events):
        if False:
            return 10
        for ev in events:
            if isinstance(ev.new_value, (float, int)):
                res.append(ev.new_value)
            else:
                res.append(None)
    handler = n.reaction(func, 'children*.val')
    loop.iter()
    with loop:
        n.set_children((n1, n2))
        n.set_val(42)
    with loop:
        n1.set_val(17)
        n2.set_val(27)
    with loop:
        n1.set_val(18)
        n2.set_val(28)
    with loop:
        n.set_children((n2,))
    with loop:
        n1.set_val(19)
        n2.set_val(29)
    with loop:
        n.set_children(())
    with loop:
        n1.set_val(11)
        n2.set_val(21)
    print(res)

@run_in_both(Node)
def test_dynamism4b():
    if False:
        while True:
            i = 10
    '\n    children.val 17, 27\n    children.val 18, 28\n    children.val 29\n    [None, None, 17, 27, 18, 28, None, 29, None]\n    '
    n = Node()
    n1 = Node()
    n2 = Node()
    res = []

    def func(*events):
        if False:
            for i in range(10):
                print('nop')
        for ev in events:
            if isinstance(ev.new_value, (float, int)):
                res.append(ev.new_value)
            else:
                res.append(None)
    handler = n.reaction(func, 'children', 'children*.val')
    loop.iter()
    with loop:
        n.set_children((n1, n2))
        n.set_val(42)
    with loop:
        n1.set_val(17)
        n2.set_val(27)
    with loop:
        n1.set_val(18)
        n2.set_val(28)
    with loop:
        n.set_children((n2,))
    with loop:
        n1.set_val(19)
        n2.set_val(29)
    with loop:
        n.set_children(())
    with loop:
        n1.set_val(11)
        n2.set_val(21)
    print(res)

@run_in_both(Node)
def test_dynamism5a():
    if False:
        print('Hello World!')
    '\n    [0, 17, 18, 19]\n    '
    n = Node()
    n1 = Node()
    n.foo = n1
    res = []

    def func(*events):
        if False:
            for i in range(10):
                print('nop')
        for ev in events:
            if isinstance(ev.new_value, (float, int)):
                res.append(ev.new_value)
            else:
                res.append(None)
    handler = n.reaction(func, 'foo.val')
    loop.iter()
    with loop:
        n.set_val(42)
    with loop:
        n1.set_val(17)
        n1.set_val(18)
    with loop:
        n.foo = None
    with loop:
        n1.set_val(19)
    print(res)

@run_in_both(Node)
def test_dynamism5b():
    if False:
        print('Hello World!')
    '\n    [17, 18, 19]\n    '
    n = Node()
    n1 = Node()
    n.foo = n1
    res = []

    def func(*events):
        if False:
            for i in range(10):
                print('nop')
        for ev in events:
            if isinstance(ev.new_value, (float, int)):
                res.append(ev.new_value)
            else:
                res.append(None)
    loop.iter()
    handler = n.reaction(func, 'foo.val')
    loop.iter()
    with loop:
        n.set_val(42)
    with loop:
        n1.set_val(17)
        n1.set_val(18)
    with loop:
        n.foo = None
    with loop:
        n1.set_val(19)
    print(res)

@run_in_both(Node)
def test_deep1():
    if False:
        i = 10
        return i + 15
    '\n    children.val 7\n    children.val 8\n    children.val 17\n    [7, 8, 17]\n    '
    n = Node()
    n1 = Node()
    n2 = Node()
    n.set_children((Node(), n1))
    loop.iter()
    n.children[0].set_children((Node(), n2))
    loop.iter()
    res = []

    def func(*events):
        if False:
            i = 10
            return i + 15
        for ev in events:
            if isinstance(ev.new_value, (float, int)):
                if ev.new_value:
                    res.append(ev.new_value)
            else:
                res.append(None)
    handler = n.reaction(func, 'children**.val')
    loop.iter()
    with loop:
        n1.set_val(7)
    with loop:
        n2.set_val(8)
    with loop:
        n.set_val(42)
    with loop:
        n1.set_children((Node(), Node()))
        n.children[0].set_children([])
    with loop:
        n1.set_val(17)
    with loop:
        n2.set_val(18)
    print(res)

@run_in_both(Node)
def test_deep2():
    if False:
        return 10
    "\n    children.val 11\n    children.val 12\n    ['id12', 'id11', 'id10', 'id11']\n    "
    n = Node()
    n1 = Node()
    n2 = Node()
    n.set_children((Node(), n1))
    loop.iter()
    n.children[0].set_children((Node(), n2))
    loop.iter()
    res = []

    def func(*events):
        if False:
            return 10
        for ev in events:
            if isinstance(ev.new_value, (float, int)):
                res.append(ev.new_value)
            elif ev.type == 'children':
                if ev.source.val:
                    res.append('id%i' % ev.source.val)
            else:
                res.append(None)
    handler = n.reaction(func, 'children**')
    loop.iter()
    with loop:
        n.set_val(10)
    with loop:
        n1.set_val(11)
    with loop:
        n2.set_val(12)
    with loop:
        n2.set_children((Node(), Node(), Node()))
        n1.set_children((Node(), Node()))
        n.set_children((Node(), n1, Node()))
    with loop:
        n2.set_children([])
        n1.set_children([])
    print(res)

class TestOb(event.Component):
    children = event.TupleProp(settable=True)
    foo = event.StringProp(settable=True)

class Tester(event.Component):
    children = event.TupleProp(settable=True)

    @event.reaction('children**.foo')
    def track_deep(self, *events):
        if False:
            print('Hello World!')
        for ev in events:
            if ev.new_value:
                print(ev.new_value)

    @event.action
    def set_foos(self, prefix):
        if False:
            print('Hello World!')
        for (i, child) in enumerate(self.children):
            child.set_foo(prefix + str(i))
            for (j, subchild) in enumerate(child.children):
                subchild.set_foo(prefix + str(i) + str(j))

    @event.action
    def make_children1(self):
        if False:
            for i in range(10):
                print('nop')
        t1 = TestOb()
        t2 = TestOb()
        t1.set_children((TestOb(),))
        t2.set_children((TestOb(),))
        self.set_children(t1, t2)

    @event.action
    def make_children2(self):
        if False:
            i = 10
            return i + 15
        for (i, child) in enumerate(self.children):
            child.set_children(child.children + (TestOb(),))

    @event.action
    def make_children3(self):
        if False:
            print('Hello World!')
        t = TestOb()
        my_children = self.children
        self.set_children(my_children + (t,))
        for (i, child) in enumerate(my_children):
            child.set_children(child.children + (t,))
        self.set_children(my_children)

@run_in_both(TestOb, Tester)
def test_issue_460_and_more():
    if False:
        return 10
    '\n    A0\n    A00\n    A1\n    A10\n    -\n    B0\n    B00\n    B01\n    B1\n    B10\n    B11\n    -\n    C0\n    C00\n    C01\n    C02\n    C1\n    C10\n    C11\n    C12\n    '
    tester = Tester()
    loop.iter()
    tester.make_children1()
    loop.iter()
    tester.set_foos('A')
    loop.iter()
    print('-')
    tester.make_children2()
    loop.iter()
    tester.set_foos('B')
    loop.iter()
    print('-')
    tester.make_children3()
    loop.iter()
    tester.set_foos('C')
    loop.iter()

class MyComponent(event.Component):
    a = event.AnyProp()
    aa = event.TupleProp()

def test_connectors1():
    if False:
        print('Hello World!')
    ' test connectors '
    x = MyComponent()

    def foo(*events):
        if False:
            i = 10
            return i + 15
        pass
    with capture_log('warning') as log:
        h = x.reaction(foo, 'a:+asdkjb&^*!')
    type = h.get_connection_info()[0][1][0]
    assert type.startswith('a:')
    assert not log
    with capture_log('warning') as log:
        h = x.reaction(foo, 'b')
    assert log
    x._Component__handlers.pop('b')
    with capture_log('warning') as log:
        h = x.reaction(foo, '!b')
    assert not log
    x._Component__handlers.pop('b')
    with capture_log('warning') as log:
        h = x.reaction(foo, '!b:meh')
    assert not log
    x._Component__handlers.pop('b')
    with capture_log('warning') as log:
        h = x.reaction(foo, 'b:meh!')
    assert log
    assert 'does not exist' in log[0]
    x._Component__handlers.pop('b')
    with capture_log('warning') as log:
        h = x.reaction(foo, 'b!:meh')
    assert log
    assert 'Exclamation mark' in log[0]

def test_connectors2():
    if False:
        return 10
    ' test connectors with sub '
    x = MyComponent()
    y = MyComponent()
    x.sub = [y]

    def foo(*events):
        if False:
            i = 10
            return i + 15
        pass
    with capture_log('warning') as log:
        h = x.reaction(foo, 'sub*.b')
    assert log
    y._Component__handlers.pop('b')
    with capture_log('warning') as log:
        h = x.reaction(foo, '!sub*.b')
    assert not log
    y._Component__handlers.pop('b')
    with capture_log('warning') as log:
        h = x.reaction(foo, '!sub*.b:meh')
    assert not log
    y._Component__handlers.pop('b')
    with capture_log('warning') as log:
        h = x.reaction(foo, 'sub*.!b:meh')
    assert log
    assert 'Exclamation mark' in log[0]
    y._Component__handlers.pop('b')
    with capture_log('warning') as log:
        h = x.reaction(foo, 'sub*.a')
    assert not log
    with capture_log('warning') as log:
        h = x.reaction(foo, 'sub.*.a')
    assert log
    with raises(ValueError):
        h = x.reaction(foo, 'sub.*a')
    with raises(RuntimeError):
        h = x.reaction(foo, 'sub.b')
    with raises(RuntimeError):
        h = y.reaction(foo, 'a*.b')
    with capture_log('warning') as log:
        h = x.reaction(foo, '!aa**')
    with capture_log('warning') as log:
        h = x.reaction(foo, '!aa*')
    assert not log
    with capture_log('warning') as log:
        h = y.reaction(foo, '!aa*')
    assert not log
    with capture_log('warning') as log:
        h = x.reaction(foo, '!aa**')
    assert not log
    with capture_log('warning') as log:
        h = x.reaction(foo, '!aa**:meh')
    assert not log

def test_dynamism_and_handler_reconnecting():
    if False:
        print('Hello World!')

    class Foo(event.Component):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
        bars = event.ListProp(settable=True)

        def disconnect(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            super().disconnect(*args)
            disconnects.append(self)

    class Bar(event.Component):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
        spam = event.AnyProp(0, settable=True)

        def disconnect(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            super().disconnect(*args)
            disconnects.append(self)
    f = Foo()
    triggers = []
    disconnects = []

    @f.reaction('!bars*.spam')
    def handle_foo(*events):
        if False:
            while True:
                i = 10
        triggers.append(len(events))
    assert len(triggers) == 0
    assert len(disconnects) == 0
    with event.loop:
        f.set_bars([Bar(), Bar()])
    assert len(triggers) == 0
    assert len(disconnects) == 0
    with event.loop:
        f.bars[0].set_spam(7)
        f.bars[1].set_spam(42)
    assert sum(triggers) == 2
    assert len(disconnects) == 0
    with event.loop:
        f.set_bars([Bar(), Bar(), Bar()])
    assert sum(triggers) == 2
    assert len(disconnects) == 2
    disconnects = []
    with event.loop:
        f.set_bars(f.bars + [Bar(), Bar()])
    assert len(disconnects) == 0
    disconnects = []
    with event.loop:
        f.set_bars(f.bars[:-1] + [Bar(), Bar()])
    assert len(disconnects) == 1
    disconnects = []
    with event.loop:
        f.set_bars(f.bars[1:] + [Bar(), Bar()])
    assert len(disconnects) == len(f.bars) - 1
    disconnects = []
    with event.loop:
        f.set_bars([Bar(), Bar()] + f.bars)
    assert len(disconnects) == 0
    disconnects = []
    with event.loop:
        f.set_bars([Bar(), Bar()] + f.bars[1:])
    assert len(disconnects) == 1
    disconnects = []
    with event.loop:
        f.set_bars([Bar(), Bar()] + f.bars[:-1])
    assert len(disconnects) == len(f.bars) - 1
run_tests_if_main()