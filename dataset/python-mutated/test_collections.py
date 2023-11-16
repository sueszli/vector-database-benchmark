import pickle
from collections.abc import Mapping
from itertools import count
from time import monotonic
from unittest.mock import Mock
import pytest
from billiard.einfo import ExceptionInfo
import t.skip
from celery.utils.collections import AttributeDict, BufferMap, ChainMap, ConfigurationView, DictAttribute, LimitedSet, Messagebuffer
from celery.utils.objects import Bunch

class test_DictAttribute:

    def test_get_set_keys_values_items(self):
        if False:
            return 10
        x = DictAttribute(Bunch())
        x['foo'] = 'The quick brown fox'
        assert x['foo'] == 'The quick brown fox'
        assert x['foo'] == x.obj.foo
        assert x.get('foo') == 'The quick brown fox'
        assert x.get('bar') is None
        with pytest.raises(KeyError):
            x['bar']
        x.foo = 'The quick yellow fox'
        assert x['foo'] == 'The quick yellow fox'
        assert ('foo', 'The quick yellow fox') in list(x.items())
        assert 'foo' in list(x.keys())
        assert 'The quick yellow fox' in list(x.values())

    def test_setdefault(self):
        if False:
            for i in range(10):
                print('nop')
        x = DictAttribute(Bunch())
        x.setdefault('foo', 'NEW')
        assert x['foo'] == 'NEW'
        x.setdefault('foo', 'XYZ')
        assert x['foo'] == 'NEW'

    def test_contains(self):
        if False:
            i = 10
            return i + 15
        x = DictAttribute(Bunch())
        x['foo'] = 1
        assert 'foo' in x
        assert 'bar' not in x

    def test_items(self):
        if False:
            print('Hello World!')
        obj = Bunch(attr1=1)
        x = DictAttribute(obj)
        x['attr2'] = 2
        assert x['attr1'] == 1
        assert x['attr2'] == 2

class test_ConfigurationView:

    def setup_method(self):
        if False:
            return 10
        self.view = ConfigurationView({'changed_key': 1, 'both': 2}, [{'default_key': 1, 'both': 1}])

    def test_setdefault(self):
        if False:
            while True:
                i = 10
        self.view.setdefault('both', 36)
        assert self.view['both'] == 2
        self.view.setdefault('new', 36)
        assert self.view['new'] == 36

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.view.get('both') == 2
        sp = object()
        assert self.view.get('nonexisting', sp) is sp

    def test_update(self):
        if False:
            i = 10
            return i + 15
        changes = dict(self.view.changes)
        self.view.update(a=1, b=2, c=3)
        assert self.view.changes == dict(changes, a=1, b=2, c=3)

    def test_contains(self):
        if False:
            return 10
        assert 'changed_key' in self.view
        assert 'default_key' in self.view
        assert 'new' not in self.view

    def test_repr(self):
        if False:
            while True:
                i = 10
        assert 'changed_key' in repr(self.view)
        assert 'default_key' in repr(self.view)

    def test_iter(self):
        if False:
            while True:
                i = 10
        expected = {'changed_key': 1, 'default_key': 1, 'both': 2}
        assert dict(self.view.items()) == expected
        assert sorted(list(iter(self.view))) == sorted(list(expected.keys()))
        assert sorted(list(self.view.keys())) == sorted(list(expected.keys()))
        assert sorted(list(self.view.values())) == sorted(list(expected.values()))
        assert 'changed_key' in list(self.view.keys())
        assert 2 in list(self.view.values())
        assert ('both', 2) in list(self.view.items())

    def test_add_defaults_dict(self):
        if False:
            return 10
        defaults = {'foo': 10}
        self.view.add_defaults(defaults)
        assert self.view.foo == 10

    def test_add_defaults_object(self):
        if False:
            for i in range(10):
                print('nop')
        defaults = Bunch(foo=10)
        self.view.add_defaults(defaults)
        assert self.view.foo == 10

    def test_clear(self):
        if False:
            print('Hello World!')
        self.view.clear()
        assert self.view.both == 1
        assert 'changed_key' not in self.view

    def test_bool(self):
        if False:
            for i in range(10):
                print('nop')
        assert bool(self.view)
        self.view.maps[:] = []
        assert not bool(self.view)

    def test_len(self):
        if False:
            while True:
                i = 10
        assert len(self.view) == 3
        self.view.KEY = 33
        assert len(self.view) == 4
        self.view.clear()
        assert len(self.view) == 2

    def test_isa_mapping(self):
        if False:
            print('Hello World!')
        from collections.abc import Mapping
        assert issubclass(ConfigurationView, Mapping)

    def test_isa_mutable_mapping(self):
        if False:
            i = 10
            return i + 15
        from collections.abc import MutableMapping
        assert issubclass(ConfigurationView, MutableMapping)

class test_ExceptionInfo:

    def test_exception_info(self):
        if False:
            while True:
                i = 10
        try:
            raise LookupError('The quick brown fox jumps...')
        except Exception:
            einfo = ExceptionInfo()
            assert str(einfo) == einfo.traceback
            assert isinstance(einfo.exception.exc, LookupError)
            assert einfo.exception.exc.args == ('The quick brown fox jumps...',)
            assert einfo.traceback
            assert repr(einfo)

@t.skip.if_win32
class test_LimitedSet:

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        s = LimitedSet(maxlen=2)
        s.add('foo')
        s.add('bar')
        for n in ('foo', 'bar'):
            assert n in s
        s.add('baz')
        for n in ('bar', 'baz'):
            assert n in s
        assert 'foo' not in s
        s = LimitedSet(maxlen=10)
        for i in range(150):
            s.add(i)
        assert len(s) <= 10
        assert len(s._heap) < len(s) * (100.0 + s.max_heap_percent_overload) / 100

    def test_purge(self):
        if False:
            while True:
                i = 10
        s = LimitedSet(maxlen=10)
        [s.add(i) for i in range(10)]
        s.maxlen = 2
        s.purge()
        assert len(s) == 2
        s = LimitedSet(maxlen=10, expires=1)
        [s.add(i) for i in range(10)]
        s.maxlen = 2
        s.purge(now=monotonic() + 100)
        assert len(s) == 0
        s = LimitedSet(maxlen=None, expires=1)
        [s.add(i) for i in range(10)]
        s.maxlen = 2
        s.purge(now=lambda : monotonic() - 100)
        assert len(s) == 2
        s = LimitedSet(maxlen=10, minlen=10, expires=1)
        [s.add(i) for i in range(20)]
        s.minlen = 3
        s.purge(now=monotonic() + 3)
        assert s.minlen == len(s)
        assert len(s._heap) <= s.maxlen * (100.0 + s.max_heap_percent_overload) / 100

    def test_pickleable(self):
        if False:
            for i in range(10):
                print('nop')
        s = LimitedSet(maxlen=2)
        s.add('foo')
        s.add('bar')
        assert pickle.loads(pickle.dumps(s)) == s

    def test_iter(self):
        if False:
            while True:
                i = 10
        s = LimitedSet(maxlen=3)
        items = ['foo', 'bar', 'baz', 'xaz']
        for item in items:
            s.add(item)
        l = list(iter(s))
        for item in items[1:]:
            assert item in l
        assert 'foo' not in l
        assert l == items[1:], 'order by insertion time'

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        s = LimitedSet(maxlen=2)
        items = ('foo', 'bar')
        for item in items:
            s.add(item)
        assert 'LimitedSet(' in repr(s)

    def test_discard(self):
        if False:
            return 10
        s = LimitedSet(maxlen=2)
        s.add('foo')
        s.discard('foo')
        assert 'foo' not in s
        assert len(s._data) == 0
        s.discard('foo')

    def test_clear(self):
        if False:
            return 10
        s = LimitedSet(maxlen=2)
        s.add('foo')
        s.add('bar')
        assert len(s) == 2
        s.clear()
        assert not s

    def test_update(self):
        if False:
            while True:
                i = 10
        s1 = LimitedSet(maxlen=2)
        s1.add('foo')
        s1.add('bar')
        s2 = LimitedSet(maxlen=2)
        s2.update(s1)
        assert sorted(list(s2)) == ['bar', 'foo']
        s2.update(['bla'])
        assert sorted(list(s2)) == ['bar', 'bla']
        s2.update(['do', 're'])
        assert sorted(list(s2)) == ['do', 're']
        s1 = LimitedSet(maxlen=10, expires=None)
        s2 = LimitedSet(maxlen=10, expires=None)
        s3 = LimitedSet(maxlen=10, expires=None)
        s4 = LimitedSet(maxlen=10, expires=None)
        s5 = LimitedSet(maxlen=10, expires=None)
        for i in range(12):
            s1.add(i)
            s2.add(i * i)
        s3.update(s1)
        s3.update(s2)
        s4.update(s1.as_dict())
        s4.update(s2.as_dict())
        s5.update(s1._data)
        s5.update(s2._data)
        assert s3 == s4
        assert s3 == s5
        s2.update(s4)
        s4.update(s2)
        assert s2 == s4

    def test_iterable_and_ordering(self):
        if False:
            return 10
        s = LimitedSet(maxlen=35, expires=None)
        clock = count(1)
        for i in reversed(range(15)):
            s.add(i, now=next(clock))
        j = 40
        for i in s:
            assert i < j
            j = i
        assert i == 0

    def test_pop_and_ordering_again(self):
        if False:
            for i in range(10):
                print('nop')
        s = LimitedSet(maxlen=5)
        for i in range(10):
            s.add(i)
        j = -1
        for _ in range(5):
            i = s.pop()
            assert j < i
        i = s.pop()
        assert i is None

    def test_as_dict(self):
        if False:
            return 10
        s = LimitedSet(maxlen=2)
        s.add('foo')
        assert isinstance(s.as_dict(), Mapping)

    def test_add_removes_duplicate_from_small_heap(self):
        if False:
            i = 10
            return i + 15
        s = LimitedSet(maxlen=2)
        s.add('foo')
        s.add('foo')
        s.add('foo')
        assert len(s) == 1
        assert len(s._data) == 1
        assert len(s._heap) == 1

    def test_add_removes_duplicate_from_big_heap(self):
        if False:
            while True:
                i = 10
        s = LimitedSet(maxlen=1000)
        [s.add(i) for i in range(2000)]
        assert len(s) == 1000
        [s.add('foo') for i in range(1000)]
        assert len(s._heap) < 1150
        [s.add('foo') for i in range(1000)]
        assert len(s._heap) < 1150

class test_AttributeDict:

    def test_getattr__setattr(self):
        if False:
            while True:
                i = 10
        x = AttributeDict({'foo': 'bar'})
        assert x['foo'] == 'bar'
        with pytest.raises(AttributeError):
            x.bar
        x.bar = 'foo'
        assert x['bar'] == 'foo'

class test_Messagebuffer:

    def assert_size_and_first(self, buf, size, expected_first_item):
        if False:
            return 10
        assert len(buf) == size
        assert buf.take() == expected_first_item

    def test_append_limited(self):
        if False:
            return 10
        b = Messagebuffer(10)
        for i in range(20):
            b.put(i)
        self.assert_size_and_first(b, 10, 10)

    def test_append_unlimited(self):
        if False:
            while True:
                i = 10
        b = Messagebuffer(None)
        for i in range(20):
            b.put(i)
        self.assert_size_and_first(b, 20, 0)

    def test_extend_limited(self):
        if False:
            return 10
        b = Messagebuffer(10)
        b.extend(list(range(20)))
        self.assert_size_and_first(b, 10, 10)

    def test_extend_unlimited(self):
        if False:
            print('Hello World!')
        b = Messagebuffer(None)
        b.extend(list(range(20)))
        self.assert_size_and_first(b, 20, 0)

    def test_extend_eviction_time_limited(self):
        if False:
            while True:
                i = 10
        b = Messagebuffer(3000)
        b.extend(range(10000))
        assert len(b) > 3000
        b.evict()
        assert len(b) == 3000

    def test_pop_empty_with_default(self):
        if False:
            print('Hello World!')
        b = Messagebuffer(10)
        sentinel = object()
        assert b.take(sentinel) is sentinel

    def test_pop_empty_no_default(self):
        if False:
            print('Hello World!')
        b = Messagebuffer(10)
        with pytest.raises(b.Empty):
            b.take()

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        assert repr(Messagebuffer(10, [1, 2, 3]))

    def test_iter(self):
        if False:
            i = 10
            return i + 15
        b = Messagebuffer(10, list(range(10)))
        assert len(b) == 10
        for (i, item) in enumerate(b):
            assert item == i
        assert len(b) == 0

    def test_contains(self):
        if False:
            i = 10
            return i + 15
        b = Messagebuffer(10, list(range(10)))
        assert 5 in b

    def test_reversed(self):
        if False:
            return 10
        assert list(reversed(Messagebuffer(10, list(range(10))))) == list(reversed(range(10)))

    def test_getitem(self):
        if False:
            while True:
                i = 10
        b = Messagebuffer(10, list(range(10)))
        for i in range(10):
            assert b[i] == i

class test_BufferMap:

    def test_append_limited(self):
        if False:
            return 10
        b = BufferMap(10)
        for i in range(20):
            b.put(i, i)
        self.assert_size_and_first(b, 10, 10)

    def assert_size_and_first(self, buf, size, expected_first_item):
        if False:
            return 10
        assert buf.total == size
        assert buf._LRUpop() == expected_first_item

    def test_append_unlimited(self):
        if False:
            print('Hello World!')
        b = BufferMap(None)
        for i in range(20):
            b.put(i, i)
        self.assert_size_and_first(b, 20, 0)

    def test_extend_limited(self):
        if False:
            while True:
                i = 10
        b = BufferMap(10)
        b.extend(1, list(range(20)))
        self.assert_size_and_first(b, 10, 10)

    def test_extend_unlimited(self):
        if False:
            while True:
                i = 10
        b = BufferMap(None)
        b.extend(1, list(range(20)))
        self.assert_size_and_first(b, 20, 0)

    def test_pop_empty_with_default(self):
        if False:
            while True:
                i = 10
        b = BufferMap(10)
        sentinel = object()
        assert b.take(1, sentinel) is sentinel

    def test_pop_empty_no_default(self):
        if False:
            for i in range(10):
                print('nop')
        b = BufferMap(10)
        with pytest.raises(b.Empty):
            b.take(1)

    def test_repr(self):
        if False:
            return 10
        assert repr(Messagebuffer(10, [1, 2, 3]))

class test_ChainMap:

    def test_observers_not_shared(self):
        if False:
            print('Hello World!')
        a = ChainMap()
        b = ChainMap()
        callback = Mock()
        a.bind_to(callback)
        b.update(x=1)
        callback.assert_not_called()
        a.update(x=1)
        callback.assert_called_once_with(x=1)