from __future__ import annotations
import pytest
pytest
from functools import partial
from bokeh.document import Document
from bokeh.model.util import HasDocumentRef
import bokeh.util.callback_manager as cbm

class _GoodPropertyCallback:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.last_name = None
        self.last_old = None
        self.last_new = None

    def __call__(self, name, old, new):
        if False:
            i = 10
            return i + 15
        self.method(name, old, new)

    def method(self, name, old, new):
        if False:
            print('Hello World!')
        self.last_name = name
        self.last_old = old
        self.last_new = new

    def partially_good(self, name, old, new, newer):
        if False:
            i = 10
            return i + 15
        pass

    def just_fine(self, name, old, new, extra='default'):
        if False:
            i = 10
            return i + 15
        pass

class _BadPropertyCallback:

    def __call__(self, x, y):
        if False:
            print('Hello World!')
        pass

    def method(self, x, y):
        if False:
            print('Hello World!')
        pass

def _good_property(x, y, z):
    if False:
        for i in range(10):
            print('nop')
    pass

def _bad_property(x, y):
    if False:
        print('Hello World!')
    pass

def _partially_good_property(w, x, y, z):
    if False:
        print('Hello World!')
    pass

def _just_fine_property(w, x, y, z='default'):
    if False:
        return 10
    pass

class _GoodEventCallback:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.last_name = None
        self.last_old = None
        self.last_new = None

    def __call__(self, event):
        if False:
            while True:
                i = 10
        self.method(event)

    def method(self, event):
        if False:
            print('Hello World!')
        self.event = event

    def partially_good(self, arg, event):
        if False:
            i = 10
            return i + 15
        pass

class _BadEventCallback:

    def __call__(self):
        if False:
            while True:
                i = 10
        pass

    def method(self):
        if False:
            i = 10
            return i + 15
        pass

def _good_event(event):
    if False:
        while True:
            i = 10
    pass

def _bad_event(x, y, z):
    if False:
        i = 10
        return i + 15
    pass

def _partially_good_event(arg, event):
    if False:
        while True:
            i = 10
    pass

def _partially_bad_event(event):
    if False:
        return 10
    pass

class TestPropertyCallbackManager:

    def test_creation(self) -> None:
        if False:
            return 10
        m = cbm.PropertyCallbackManager()
        assert len(m._callbacks) == 0

    def test_on_change_good_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        m = cbm.PropertyCallbackManager()
        good = _GoodPropertyCallback()
        m.on_change('foo', good.method)
        assert len(m._callbacks) == 1
        assert m._callbacks['foo'] == [good.method]

    def test_on_change_good_partial_function(self) -> None:
        if False:
            i = 10
            return i + 15
        m = cbm.PropertyCallbackManager()
        p = partial(_partially_good_property, 'foo')
        m.on_change('bar', p)
        assert len(m._callbacks) == 1

    def test_on_change_good_partial_method(self) -> None:
        if False:
            return 10
        m = cbm.PropertyCallbackManager()
        good = _GoodPropertyCallback()
        p = partial(good.partially_good, 'foo')
        m.on_change('bar', p)
        assert len(m._callbacks) == 1

    def test_on_change_good_extra_kwargs_function(self) -> None:
        if False:
            while True:
                i = 10
        m = cbm.PropertyCallbackManager()
        m.on_change('bar', _just_fine_property)
        assert len(m._callbacks) == 1

    def test_on_change_good_extra_kwargs_method(self) -> None:
        if False:
            i = 10
            return i + 15
        m = cbm.PropertyCallbackManager()
        good = _GoodPropertyCallback()
        m.on_change('bar', good.just_fine)
        assert len(m._callbacks) == 1

    def test_on_change_good_functor(self) -> None:
        if False:
            print('Hello World!')
        m = cbm.PropertyCallbackManager()
        good = _GoodPropertyCallback()
        m.on_change('foo', good)
        assert len(m._callbacks) == 1
        assert m._callbacks['foo'] == [good]

    def test_on_change_good_function(self) -> None:
        if False:
            print('Hello World!')
        m = cbm.PropertyCallbackManager()
        m.on_change('foo', _good_property)
        assert len(m._callbacks) == 1
        assert m._callbacks['foo'] == [_good_property]

    def test_on_change_good_lambda(self) -> None:
        if False:
            return 10
        m = cbm.PropertyCallbackManager()
        good = lambda x, y, z: x
        m.on_change('foo', good)
        assert len(m._callbacks) == 1
        assert m._callbacks['foo'] == [good]

    def test_on_change_good_closure(self) -> None:
        if False:
            while True:
                i = 10

        def good(x, y, z):
            if False:
                while True:
                    i = 10
            pass
        m = cbm.PropertyCallbackManager()
        m.on_change('foo', good)
        assert len(m._callbacks) == 1
        assert len(m._callbacks['foo']) == 1

    def test_on_change_bad_method(self) -> None:
        if False:
            return 10
        m = cbm.PropertyCallbackManager()
        bad = _BadPropertyCallback()
        with pytest.raises(ValueError):
            m.on_change('foo', bad.method)
        assert len(m._callbacks) == 1
        assert len(m._callbacks['foo']) == 0

    def test_on_change_bad_functor(self) -> None:
        if False:
            return 10
        m = cbm.PropertyCallbackManager()
        bad = _BadPropertyCallback()
        with pytest.raises(ValueError):
            m.on_change('foo', bad)
        assert len(m._callbacks) == 1
        assert len(m._callbacks['foo']) == 0

    def test_on_change_bad_function(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        m = cbm.PropertyCallbackManager()
        with pytest.raises(ValueError):
            m.on_change('foo', _bad_property)
        assert len(m._callbacks) == 1
        assert len(m._callbacks['foo']) == 0

    def test_on_change_bad_lambda(self) -> None:
        if False:
            print('Hello World!')
        m = cbm.PropertyCallbackManager()
        with pytest.raises(ValueError):
            m.on_change('foo', lambda x, y: x)
        assert len(m._callbacks) == 1
        assert len(m._callbacks['foo']) == 0

    def test_on_change_bad_closure(self) -> None:
        if False:
            print('Hello World!')

        def bad(x, y):
            if False:
                while True:
                    i = 10
            pass
        m = cbm.PropertyCallbackManager()
        with pytest.raises(ValueError):
            m.on_change('foo', bad)
        assert len(m._callbacks) == 1
        assert len(m._callbacks['foo']) == 0

    def test_on_change_same_attr_twice_multiple_calls(self) -> None:
        if False:
            print('Hello World!')

        def good1(x, y, z):
            if False:
                print('Hello World!')
            pass

        def good2(x, y, z):
            if False:
                return 10
            pass
        m1 = cbm.PropertyCallbackManager()
        m1.on_change('foo', good1)
        m1.on_change('foo', good2)
        assert len(m1._callbacks) == 1
        assert m1._callbacks['foo'] == [good1, good2]

    def test_on_change_same_attr_twice_one_call(self) -> None:
        if False:
            print('Hello World!')

        def good1(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def good2(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            pass
        m2 = cbm.PropertyCallbackManager()
        m2.on_change('foo', good1, good2)
        assert len(m2._callbacks) == 1
        assert m2._callbacks['foo'] == [good1, good2]

    def test_on_change_different_attrs(self) -> None:
        if False:
            i = 10
            return i + 15

        def good1(x, y, z):
            if False:
                print('Hello World!')
            pass

        def good2(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            pass
        m1 = cbm.PropertyCallbackManager()
        m1.on_change('foo', good1)
        m1.on_change('bar', good2)
        assert len(m1._callbacks) == 2
        assert m1._callbacks['foo'] == [good1]
        assert m1._callbacks['bar'] == [good2]

    def test_trigger(self) -> None:
        if False:
            i = 10
            return i + 15

        class Modelish(HasDocumentRef, cbm.PropertyCallbackManager):
            pass
        m = Modelish()
        good = _GoodPropertyCallback()
        m.on_change('foo', good.method)
        m.trigger('foo', 42, 43)
        assert good.last_name == 'foo'
        assert good.last_old == 42
        assert good.last_new == 43

    def test_trigger_with_two_callbacks(self) -> None:
        if False:
            return 10

        class Modelish(HasDocumentRef, cbm.PropertyCallbackManager):
            pass
        m = Modelish()
        good1 = _GoodPropertyCallback()
        good2 = _GoodPropertyCallback()
        m.on_change('foo', good1.method)
        m.on_change('foo', good2.method)
        m.trigger('foo', 42, 43)
        assert good1.last_name == 'foo'
        assert good1.last_old == 42
        assert good1.last_new == 43
        assert good2.last_name == 'foo'
        assert good2.last_old == 42
        assert good2.last_new == 43

class TestEventCallbackManager:

    def test_creation(self) -> None:
        if False:
            while True:
                i = 10
        m = cbm.EventCallbackManager()
        assert len(m._event_callbacks) == 0

    def test_on_change_good_method(self) -> None:
        if False:
            return 10
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        good = _GoodEventCallback()
        m.on_event('foo', good.method)
        assert len(m._event_callbacks) == 1
        assert m._event_callbacks['foo'] == [good.method]

    def test_on_change_good_partial_function(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        m = cbm.EventCallbackManager()
        p = partial(_partially_good_event, 'foo')
        m.subscribed_events = set()
        m.on_event('foo', p)
        assert len(m._event_callbacks) == 1
        assert m._event_callbacks['foo'] == [p]

    def test_on_change_bad_partial_function(self) -> None:
        if False:
            while True:
                i = 10
        m = cbm.EventCallbackManager()
        p = partial(_partially_bad_event, 'foo')
        m.subscribed_events = set()
        m.on_event('foo', p)
        assert len(m._event_callbacks) == 1

    def test_on_change_good_partial_method(self) -> None:
        if False:
            return 10
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        good = _GoodEventCallback()
        p = partial(good.partially_good, 'foo')
        m.on_event('foo', p)
        assert len(m._event_callbacks) == 1

    def test_on_change_good_functor(self) -> None:
        if False:
            while True:
                i = 10
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        good = _GoodEventCallback()
        m.on_event('foo', good)
        assert len(m._event_callbacks) == 1
        assert m._event_callbacks['foo'] == [good]

    def test_on_change_good_function(self) -> None:
        if False:
            while True:
                i = 10
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        m.on_event('foo', _good_event)
        assert len(m._event_callbacks) == 1
        assert m._event_callbacks['foo'] == [_good_event]

    def test_on_change_unicode_event_name(self) -> None:
        if False:
            i = 10
            return i + 15
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        m.on_event('foo', _good_event)
        assert len(m._event_callbacks) == 1
        assert m._event_callbacks['foo'] == [_good_event]

    def test_on_change_good_lambda(self) -> None:
        if False:
            i = 10
            return i + 15
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        good = lambda event: event
        m.on_event('foo', good)
        assert len(m._event_callbacks) == 1
        assert m._event_callbacks['foo'] == [good]

    def test_on_change_good_closure(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def good(event):
            if False:
                while True:
                    i = 10
            pass
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        m.on_event('foo', good)
        assert len(m._event_callbacks) == 1
        assert len(m._event_callbacks['foo']) == 1

    def test_on_change_bad_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        bad = _BadEventCallback()
        m.on_event('foo', bad.method)
        assert len(m._event_callbacks) == 1

    def test_on_change_bad_functor(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        bad = _BadEventCallback()
        m.on_event('foo', bad)
        assert len(m._event_callbacks) == 1

    def test_on_change_bad_function(self) -> None:
        if False:
            i = 10
            return i + 15
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        with pytest.raises(ValueError):
            m.on_event('foo', _bad_event)
        assert len(m._event_callbacks) == 0

    def test_on_change_bad_lambda(self) -> None:
        if False:
            i = 10
            return i + 15
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        with pytest.raises(ValueError):
            m.on_event('foo', lambda x, y: x)
        assert len(m._event_callbacks) == 0

    def test_on_change_bad_closure(self) -> None:
        if False:
            i = 10
            return i + 15

        def bad(event, y):
            if False:
                while True:
                    i = 10
            pass
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        with pytest.raises(ValueError):
            m.on_event('foo', bad)
        assert len(m._event_callbacks) == 0

    def test_on_change_with_two_callbacks(self) -> None:
        if False:
            while True:
                i = 10
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        good1 = _GoodEventCallback()
        good2 = _GoodEventCallback()
        m.on_event('foo', good1.method)
        m.on_event('foo', good2.method)

    def test_on_change_with_two_callbacks_one_bad(self) -> None:
        if False:
            while True:
                i = 10
        m = cbm.EventCallbackManager()
        m.subscribed_events = set()
        good = _GoodEventCallback()
        bad = _BadEventCallback()
        m.on_event('foo', good.method, bad.method)
        assert len(m._event_callbacks) == 1

    def test__trigger_event_wraps_curdoc(self) -> None:
        if False:
            return 10
        from bokeh.io import curdoc
        from bokeh.io.doc import set_curdoc
        oldcd = curdoc()
        d1 = Document()
        d2 = Document()
        set_curdoc(d1)
        out = {}

        def cb():
            if False:
                for i in range(10):
                    print('nop')
            out['curdoc'] = curdoc()

        class Modelish(HasDocumentRef, cbm.EventCallbackManager):
            pass
        m = Modelish()
        m.subscribed_events = set()
        m.on_event('foo', cb)
        m.id = 10
        m._document = d2
        assert len(m._event_callbacks) == 1
        assert m._event_callbacks['foo'] == [cb]

        class ev:
            model = m
            event_name = 'foo'
        m._trigger_event(ev())
        assert out['curdoc'] is d2
        set_curdoc(oldcd)