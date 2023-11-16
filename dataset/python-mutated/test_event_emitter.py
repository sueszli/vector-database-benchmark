import unittest
import copy
import functools
from vispy.util.event import Event, EventEmitter
from vispy.testing import run_tests_if_main, assert_raises, assert_equal

class BasicEvent(Event):
    pass

class TypedEvent(Event):

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        kwargs['type'] = 'typed_event'
        Event.__init__(self, **kwargs)

class TestEmitters(unittest.TestCase):

    def test_emitter(self):
        if False:
            return 10
        'Emitter constructed with no arguments'
        em = EventEmitter()
        try:
            em()
            assert False, 'Emitting event with no type should have failed.'
        except TypeError:
            pass
        ev = self.try_emitter(em, type='test_event')
        self.assert_result(event=ev, event_class=Event, source=None, type='test_event', sources=[None])

    def test_emitter_source(self):
        if False:
            return 10
        'Emitter constructed with source argument'
        em = EventEmitter(source=self)
        ev = self.try_emitter(em, type='test_event')
        self.assert_result(event=ev, event_class=Event, source=self, type='test_event', sources=[self])
        try:
            ev = em(type='test_event', source=None)
            assert False, 'Should not be able to specify source when emitting'
        except AttributeError:
            pass

    def test_emitter_type(self):
        if False:
            for i in range(10):
                print('nop')
        'Emitter constructed with type argument'
        em = EventEmitter(type='asdf')
        ev = self.try_emitter(em)
        self.assert_result(event=ev, event_class=Event, source=None, type='asdf', sources=[None])
        ev = self.try_emitter(em, type='qwer')
        self.assert_result(event=ev, event_class=Event, source=None, type='qwer', sources=[None])

    def test_emitter_type_event_class(self):
        if False:
            i = 10
            return i + 15
        'Emitter constructed with event_class argument'
        em = EventEmitter(event_class=BasicEvent)
        ev = self.try_emitter(em, type='test_event')
        self.assert_result(event=ev, event_class=BasicEvent, source=None, type='test_event', sources=[None])

        class X:

            def __init__(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                self.blocked = False

            def _push_source(self, s):
                if False:
                    return 10
                pass

            def _pop_source(self):
                if False:
                    i = 10
                    return i + 15
                pass
        try:
            em = EventEmitter(event_class=X)
            ev = self.try_emitter(em, type='test_event')
            self.assert_result()
            assert False, 'Should not be able to construct emitter with non-Event class'
        except Exception:
            pass

    def test_event_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Extra Event kwargs'
        em = EventEmitter(type='test_event')
        em.default_args['key1'] = 'test1'
        em.connect(self.record_event)
        self.result = None
        em(key2='test2')
        self.assert_result(key1='test1', key2='test2')

    def test_prebuilt_event(self):
        if False:
            for i in range(10):
                print('nop')
        'Emit pre-built event'
        em = EventEmitter(type='test_event')
        em.default_args['key1'] = 'test1'
        em.connect(self.record_event)
        self.result = None
        ev = Event(type='my_type')
        em(ev)
        self.assert_result(event=ev, type='my_type')
        assert not hasattr(self.result[0], 'key1')

    def test_emitter_subclass(self):
        if False:
            print('Hello World!')
        'The EventEmitter subclassing'

        class MyEmitter(EventEmitter):

            def _prepare_event(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                ev = super(MyEmitter, self)._prepare_event(*args, **kwargs)
                ev.test_tag = 1
                return ev
        em = MyEmitter(type='test_event')
        em.connect(self.record_event)
        self.result = None
        em()
        self.assert_result(test_tag=1)

    def test_typed_event(self):
        if False:
            i = 10
            return i + 15
        'Emit Event class with pre-specified type'
        em = EventEmitter(event_class=TypedEvent)
        ev = self.try_emitter(em)
        self.assert_result(event=ev, event_class=TypedEvent, source=None, type='typed_event', sources=[None])

    def test_disconnect(self):
        if False:
            for i in range(10):
                print('nop')
        'Emitter disconnection'
        em = EventEmitter(type='test_event')

        def cb1(ev):
            if False:
                print('Hello World!')
            self.result = 1

        def cb2(ev):
            if False:
                print('Hello World!')
            self.result = 2
        em.connect((self, 'record_event'))
        em.connect(cb1)
        em.connect(cb2)
        self.result = None
        em.disconnect(cb2)
        em.disconnect(cb2)
        ev = em()
        self.assert_result(event=ev)
        self.result = None
        em.disconnect((self, 'record_event'))
        ev = em()
        assert self.result == 1
        self.result = None
        em.connect(cb1)
        em.connect(cb2)
        em.connect((self, 'record_event'))
        em.disconnect()
        em()
        assert self.result is None

    def test_reconnect(self):
        if False:
            while True:
                i = 10
        'Ignore callback reconnect'
        em = EventEmitter(type='test_event')

        def cb(ev):
            if False:
                for i in range(10):
                    print('nop')
            self.result += 1
        em.connect(cb)
        em.connect(cb)
        self.result = 0
        em()
        assert self.result == 1

    def test_decorator_connection(self):
        if False:
            i = 10
            return i + 15
        'Connection by decorator'
        em = EventEmitter(type='test_event')

        @em.connect
        def cb(ev):
            if False:
                while True:
                    i = 10
            self.result = 1
        self.result = None
        em()
        assert self.result == 1

    def test_chained_emitters(self):
        if False:
            i = 10
            return i + 15
        'Chained emitters'
        em1 = EventEmitter(source=None, type='test_event1')
        em2 = EventEmitter(source=self, type='test_event2')
        em1.connect(em2)
        em1.connect(self.record_event)
        self.result = None
        ev = em1()
        self.assert_result(event=ev, event_class=Event, source=None, type='test_event1', sources=[None])
        em1.disconnect(self.record_event)
        em2.connect(self.record_event)
        self.result = None
        ev = em1()
        self.assert_result(event=ev, event_class=Event, source=self, type='test_event1', sources=[None, self])

    def test_emitter_error_handling(self):
        if False:
            while True:
                i = 10
        'Emitter error handling'
        em = EventEmitter(type='test_event')
        em.print_callback_errors = 'never'

        def cb(ev):
            if False:
                print('Hello World!')
            raise Exception('test')
        em.connect(self.record_event)
        em.connect(cb)
        self.result = None
        ev = em()
        self.assert_result(event=ev)
        self.result = None
        em.ignore_callback_errors = False
        try:
            em()
            assert False, 'Emission should have raised exception'
        except Exception as err:
            if str(err) != 'test':
                raise

    def test_emission_order(self):
        if False:
            for i in range(10):
                print('nop')
        'Event emission order'
        em = EventEmitter(type='test_event')

        def cb1(ev):
            if False:
                print('Hello World!')
            self.result = 1

        def cb2(ev):
            if False:
                i = 10
                return i + 15
            self.result = 2
        em.connect(cb1)
        em.connect(cb2)
        self.result = None
        em()
        assert self.result == 1, 'Events emitted in wrong order'
        em.disconnect()
        em.connect(cb2)
        em.connect(cb1)
        self.result = None
        em()
        assert self.result == 2, 'Events emitted in wrong order'

    def test_multiple_callbacks(self):
        if False:
            i = 10
            return i + 15
        'Multiple emitter callbacks'
        em = EventEmitter(type='test_event')
        em.connect(functools.partial(self.record_event, key=1))
        em.connect(functools.partial(self.record_event, key=2))
        em.connect(functools.partial(self.record_event, key=3))
        ev = em()
        self.assert_result(key=1, event=ev, sources=[None])
        self.assert_result(key=2, event=ev, sources=[None])
        self.assert_result(key=3, event=ev, sources=[None])

    def test_symbolic_callback(self):
        if False:
            return 10
        'Symbolic callbacks'
        em = EventEmitter(type='test_event')
        em.connect((self, 'record_event'))
        ev = em()
        self.assert_result(event=ev)

        def cb(ev):
            if False:
                while True:
                    i = 10
            self.result = 1
        self.result = None
        orig_method = self.record_event
        try:
            self.record_event = cb
            em()
            assert self.result == 1
        finally:
            self.record_event = orig_method

    def test_source_stack_integrity(self):
        if False:
            print('Hello World!')
        'Emitter checks source stack'
        em = EventEmitter(type='test_event')

        def cb(ev):
            if False:
                return 10
            ev._sources.append('x')
        em.connect(cb)
        try:
            em()
        except RuntimeError as err:
            if str(err) != 'Event source-stack mismatch.':
                raise
        em.disconnect()

        def cb(ev):
            if False:
                return 10
            ev._sources = []
        em.connect(cb)
        try:
            em()
        except IndexError:
            pass

    def test_emitter_loop(self):
        if False:
            for i in range(10):
                print('nop')
        'Catch emitter loops'
        em1 = EventEmitter(type='test_event1')
        em2 = EventEmitter(type='test_event2')
        em1.ignore_callback_errors = False
        em2.ignore_callback_errors = False
        em1.connect(em2)
        em2.connect(em1)
        try:
            em1()
        except RuntimeError as err:
            if str(err) != 'EventEmitter loop detected!':
                raise err

    def test_emitter_block(self):
        if False:
            while True:
                i = 10
        'EventEmitter.blocker'
        em = EventEmitter(type='test_event')
        em.connect(self.record_event)
        self.result = None
        with em.blocker():
            em()
        assert self.result is None
        ev = em()
        self.assert_result(event=ev)

    def test_event_handling(self):
        if False:
            return 10
        'Event.handled'
        em = EventEmitter(type='test_event')

        def cb1(ev):
            if False:
                return 10
            ev.handled = True

        def cb2(ev):
            if False:
                i = 10
                return i + 15
            assert ev.handled
            self.result = 1
        em.connect(cb2)
        em.connect(cb1)
        self.result = None
        em()
        assert self.result == 1

    def test_event_block(self):
        if False:
            i = 10
            return i + 15
        'Event.blocked'
        em = EventEmitter(type='test_event')

        def cb1(ev):
            if False:
                i = 10
                return i + 15
            ev.handled = True
            self.result = 1

        def cb2(ev):
            if False:
                return 10
            ev.blocked = True
            self.result = 2
        em.connect(self.record_event)
        em.connect(cb1)
        self.result = None
        em()
        self.assert_result()
        em.connect(cb2)
        self.result = None
        em()
        assert self.result == 2

    def try_emitter(self, em, **kwargs):
        if False:
            i = 10
            return i + 15
        em.connect(self.record_event)
        self.result = None
        return em(**kwargs)

    def record_event(self, ev, key=None):
        if False:
            return 10
        names = [name for name in dir(ev) if name[0] != '_']
        attrs = {}
        for name in names:
            val = getattr(ev, name)
            if name == 'source':
                attrs[name] = val
            elif name == 'sources':
                attrs[name] = val[:]
            else:
                try:
                    attrs[name] = copy.deepcopy(val)
                except Exception:
                    try:
                        attrs[name] = copy.copy(val)
                    except Exception:
                        attrs[name] = val
        if key is None:
            self.result = (ev, attrs)
        else:
            if not hasattr(self, 'result') or self.result is None:
                self.result = {}
            self.result[key] = (ev, attrs)

    def assert_result(self, key=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert hasattr(self, 'result') and self.result is not None, 'No event recorded'
        if key is None:
            (event, event_attrs) = self.result
        else:
            (event, event_attrs) = self.result[key]
        assert isinstance(event, Event), 'Emitted object is not Event instance'
        for (name, val) in kwargs.items():
            if name == 'event':
                assert event is val, 'Event objects do not match'
            elif name == 'event_class':
                assert isinstance(event, val), 'Emitted object is not instance of %s' % val.__name__
            else:
                attr = event_attrs[name]
                assert attr == val, 'Event.%s != %s  (%s)' % (name, str(val), str(attr))

def test_event_connect_order():
    if False:
        return 10
    'Test event connection order'

    def a():
        if False:
            for i in range(10):
                print('nop')
        return

    def b():
        if False:
            for i in range(10):
                print('nop')
        return

    def c():
        if False:
            i = 10
            return i + 15
        return

    def d():
        if False:
            i = 10
            return i + 15
        return

    def e():
        if False:
            for i in range(10):
                print('nop')
        return

    def f():
        if False:
            print('Hello World!')
        return
    em = EventEmitter(type='test_event')
    assert_raises(ValueError, em.connect, c, before=['c', 'foo'])
    assert_raises(ValueError, em.connect, c, position='foo')
    assert_raises(TypeError, em.connect, c, ref=dict())
    em.connect(c, ref=True)
    assert_equal((c,), tuple(em.callbacks))
    em.connect(c)
    assert_equal((c,), tuple(em.callbacks))
    em.connect(d, ref=True, position='last')
    assert_equal((c, d), tuple(em.callbacks))
    em.connect(b, ref=True)
    assert_equal((b, c, d), tuple(em.callbacks))
    assert_raises(RuntimeError, em.connect, a, before='c', after='d')
    em.connect(a, ref=True, before=['c', 'd'])
    assert_equal((a, b, c, d), tuple(em.callbacks))
    em.connect(f, ref=True, after=['c', 'd'])
    assert_equal((a, b, c, d, f), tuple(em.callbacks))
    em.connect(e, ref=True, after='d', before='f')
    assert_equal(('a', 'b', 'c', 'd', 'e', 'f'), tuple(em.callback_refs))
    em.disconnect(e)
    em.connect(e, ref=True, after='a', before='f', position='last')
    assert_equal(('a', 'b', 'c', 'd', 'e', 'f'), tuple(em.callback_refs))
    em.disconnect(e)
    em.connect(e, ref='e', after='d', before='f', position='last')
    assert_equal(('a', 'b', 'c', 'd', 'e', 'f'), tuple(em.callback_refs))
    em.disconnect(e)
    em.connect(e, after='d', before='f', position='first')
    assert_equal(('a', 'b', 'c', 'd', None, 'f'), tuple(em.callback_refs))
    em.disconnect(e)
    assert_raises(ValueError, em.connect, e, ref='d')
    em.connect(e, ref=True, after=[], before='f', position='last')
    assert_equal(('a', 'b', 'c', 'd', 'e', 'f'), tuple(em.callback_refs))
    assert_equal((a, b, c, d, e, f), tuple(em.callbacks))
    old_e = e

    def e():
        if False:
            i = 10
            return i + 15
        return
    assert_raises(ValueError, em.connect, e, ref=True)
    em.connect(e)
    assert_equal((None, 'a', 'b', 'c', 'd', 'e', 'f'), tuple(em.callback_refs))
    assert_equal((e, a, b, c, d, old_e, f), tuple(em.callbacks))

def test_emitter_block():
    if False:
        while True:
            i = 10
    state = [False, False]

    def a(ev):
        if False:
            print('Hello World!')
        state[0] = True

    def b(ev):
        if False:
            i = 10
            return i + 15
        state[1] = True
    e = EventEmitter(source=None, type='event')
    e.connect(a)
    e.connect(b)

    def assert_state(a, b):
        if False:
            while True:
                i = 10
        assert state == [a, b]
        state[0] = False
        state[1] = False
    e()
    assert_state(True, True)
    e.block()
    e()
    assert_state(False, False)
    e.block()
    e()
    assert_state(False, False)
    e.unblock()
    e()
    assert_state(False, False)
    e.unblock()
    e()
    assert_state(True, True)
    try:
        e.unblock()
        raise Exception('Expected RuntimeError')
    except RuntimeError:
        pass
    e.block(a)
    e()
    assert_state(False, True)
    e.block(b)
    e()
    assert_state(False, False)
    e.block(b)
    e()
    assert_state(False, False)
    e.unblock(a)
    e()
    assert_state(True, False)
    e.unblock(b)
    e()
    assert_state(True, False)
    e.unblock(b)
    e()
    assert_state(True, True)
    try:
        e.unblock(a)
        raise Exception('Expected RuntimeError')
    except RuntimeError:
        pass
    with e.blocker():
        e()
        assert_state(False, False)
        with e.blocker():
            e()
            assert_state(False, False)
        e()
        assert_state(False, False)
    e()
    assert_state(True, True)
    with e.blocker(a):
        e()
        assert_state(False, True)
        with e.blocker():
            e()
            assert_state(False, False)
        e()
        assert_state(False, True)
        with e.blocker(a):
            e()
            assert_state(False, True)
        with e.blocker(b):
            e()
            assert_state(False, False)
        e()
        assert_state(False, True)
    e()
    assert_state(True, True)

def test_emitter_reentrance_allowed_when_blocked1():
    if False:
        while True:
            i = 10
    e = EventEmitter(source=None, type='test')
    count = 0

    @e.connect
    def foo(x):
        if False:
            i = 10
            return i + 15
        nonlocal count
        count += 1
        with e.blocker():
            e()
    e()
    assert count == 1

def test_emitter_reentrance_allowed_when_blocked2():
    if False:
        while True:
            i = 10
    e1 = EventEmitter(source=None, type='test1')
    e2 = EventEmitter(source=None, type='test2')
    count = 0

    @e1.connect
    def foo(x):
        if False:
            while True:
                i = 10
        nonlocal count
        count += 1
        with e1.blocker():
            e2()

    @e2.connect
    def bar(x):
        if False:
            while True:
                i = 10
        nonlocal count
        count += 10
        e1()
    e1()
    assert count == 11

def test_emitter_reentrance_allowed_when_blocked3():
    if False:
        for i in range(10):
            print('nop')
    e1 = EventEmitter(source=None, type='test1')
    e2 = EventEmitter(source=None, type='test2')
    count = 0

    @e1.connect
    def foo(x):
        if False:
            i = 10
            return i + 15
        nonlocal count
        count += 1
        with e1.blocker(foo):
            e2()

    @e2.connect
    def bar(x):
        if False:
            for i in range(10):
                print('nop')
        nonlocal count
        count += 10
        e1()

    @e2.connect
    def eggs(x):
        if False:
            print('Hello World!')
        nonlocal count
        count += 100
        e1()
    e1()
    assert count == 111
run_tests_if_main()