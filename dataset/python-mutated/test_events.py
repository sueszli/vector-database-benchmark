import unittest
from manticore.utils.event import Eventful

class A(Eventful):
    _published_events = {'eventA'}

    def do_stuff(self):
        if False:
            for i in range(10):
                print('nop')
        self._publish('eventA', 1, 'a')

class B(Eventful):
    _published_events = {'eventB'}

    def __init__(self, child, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.child = child
        self.forward_events_from(child)

    def do_stuff(self):
        if False:
            print('Hello World!')
        self._publish('eventB', 2, 'b')

class C:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.received = []

    def callback(self, *args):
        if False:
            i = 10
            return i + 15
        self.received.append(args)

class ManticoreDriver(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_weak_references(self):
        if False:
            return 10
        a = A()
        self.assertSequenceEqual([len(s) for s in (a._signals, a._forwards)], (0, 0))
        b = B(a)
        self.assertSequenceEqual([len(s) for s in (a._signals, a._forwards)], (0, 1))
        self.assertSequenceEqual([len(s) for s in (b._signals, b._forwards)], (0, 0))
        c = C()
        b.subscribe('eventA', c.callback)
        self.assertSequenceEqual([len(s) for s in (a._signals, a._forwards)], (0, 1))
        self.assertSequenceEqual([len(s) for s in (b._signals, b._forwards)], (1, 0))
        b.subscribe('eventB', c.callback)
        self.assertSequenceEqual([len(s) for s in (a._signals, a._forwards)], (0, 1))
        self.assertSequenceEqual([len(s) for s in (b._signals, b._forwards)], (2, 0))
        del c
        self.assertSequenceEqual([len(s) for s in (a._signals, a._forwards)], (0, 1))
        self.assertSequenceEqual([len(s) for s in (b._signals, b._forwards)], (0, 0))
        del b
        self.assertSequenceEqual([len(s) for s in (a._signals, a._forwards)], (0, 0))

    def test_basic(self):
        if False:
            return 10
        a = A()
        b = B(a)
        c = C()
        b.subscribe('eventA', c.callback)
        b.subscribe('eventB', c.callback)
        a.do_stuff()
        self.assertSequenceEqual(c.received, [(1, 'a')])
        b.do_stuff()
        self.assertSequenceEqual(c.received, [(1, 'a'), (2, 'b')])