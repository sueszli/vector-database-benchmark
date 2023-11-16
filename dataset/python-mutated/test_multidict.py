import pytest
from mitmproxy.coretypes import multidict

class _TMulti:

    @staticmethod
    def _kconv(key):
        if False:
            for i in range(10):
                print('nop')
        return key.lower()

class TMultiDict(_TMulti, multidict.MultiDict):
    pass

class TestMultiDict:

    @staticmethod
    def _multi():
        if False:
            for i in range(10):
                print('nop')
        return TMultiDict((('foo', 'bar'), ('bar', 'baz'), ('Bar', 'bam')))

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        md = TMultiDict()
        assert len(md) == 0
        md = TMultiDict([('foo', 'bar')])
        assert len(md) == 1
        assert md.fields == (('foo', 'bar'),)

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        assert repr(self._multi()) == "TMultiDict[('foo', 'bar'), ('bar', 'baz'), ('Bar', 'bam')]"

    def test_getitem(self):
        if False:
            return 10
        md = TMultiDict([('foo', 'bar')])
        assert 'foo' in md
        assert 'Foo' in md
        assert md['foo'] == 'bar'
        with pytest.raises(KeyError):
            assert md['bar']
        md_multi = TMultiDict([('foo', 'a'), ('foo', 'b')])
        assert md_multi['foo'] == 'a'

    def test_setitem(self):
        if False:
            print('Hello World!')
        md = TMultiDict()
        md['foo'] = 'bar'
        assert md.fields == (('foo', 'bar'),)
        md['foo'] = 'baz'
        assert md.fields == (('foo', 'baz'),)
        md['bar'] = 'bam'
        assert md.fields == (('foo', 'baz'), ('bar', 'bam'))

    def test_delitem(self):
        if False:
            return 10
        md = self._multi()
        del md['foo']
        assert 'foo' not in md
        assert 'bar' in md
        with pytest.raises(KeyError):
            del md['foo']
        del md['bar']
        assert md.fields == ()

    def test_iter(self):
        if False:
            while True:
                i = 10
        md = self._multi()
        assert list(md.__iter__()) == ['foo', 'bar']

    def test_len(self):
        if False:
            while True:
                i = 10
        md = TMultiDict()
        assert len(md) == 0
        md = self._multi()
        assert len(md) == 2

    def test_eq(self):
        if False:
            return 10
        assert TMultiDict() == TMultiDict()
        assert not TMultiDict() == 42
        md1 = self._multi()
        md2 = self._multi()
        assert md1 == md2
        md1.fields = md1.fields[1:] + md1.fields[:1]
        assert not md1 == md2

    def test_hash(self):
        if False:
            print('Hello World!')
        "\n        If a class defines mutable objects and implements an __eq__() method,\n        it should not implement __hash__(), since the implementation of hashable\n        collections requires that a key's hash value is immutable.\n        "
        with pytest.raises(TypeError):
            assert hash(TMultiDict())

    def test_get_all(self):
        if False:
            for i in range(10):
                print('nop')
        md = self._multi()
        assert md.get_all('foo') == ['bar']
        assert md.get_all('bar') == ['baz', 'bam']
        assert md.get_all('baz') == []

    def test_set_all(self):
        if False:
            i = 10
            return i + 15
        md = TMultiDict()
        md.set_all('foo', ['bar', 'baz'])
        assert md.fields == (('foo', 'bar'), ('foo', 'baz'))
        md = TMultiDict((('a', 'b'), ('x', 'x'), ('c', 'd'), ('X', 'X'), ('e', 'f')))
        md.set_all('x', ['1', '2', '3'])
        assert md.fields == (('a', 'b'), ('x', '1'), ('c', 'd'), ('X', '2'), ('e', 'f'), ('x', '3'))
        md.set_all('x', ['4'])
        assert md.fields == (('a', 'b'), ('x', '4'), ('c', 'd'), ('e', 'f'))

    def test_add(self):
        if False:
            print('Hello World!')
        md = self._multi()
        md.add('foo', 'foo')
        assert md.fields == (('foo', 'bar'), ('bar', 'baz'), ('Bar', 'bam'), ('foo', 'foo'))

    def test_insert(self):
        if False:
            print('Hello World!')
        md = TMultiDict([('b', 'b')])
        md.insert(0, 'a', 'a')
        md.insert(2, 'c', 'c')
        assert md.fields == (('a', 'a'), ('b', 'b'), ('c', 'c'))

    def test_keys(self):
        if False:
            print('Hello World!')
        md = self._multi()
        assert list(md.keys()) == ['foo', 'bar']
        assert list(md.keys(multi=True)) == ['foo', 'bar', 'Bar']

    def test_values(self):
        if False:
            while True:
                i = 10
        md = self._multi()
        assert list(md.values()) == ['bar', 'baz']
        assert list(md.values(multi=True)) == ['bar', 'baz', 'bam']

    def test_items(self):
        if False:
            while True:
                i = 10
        md = self._multi()
        assert list(md.items()) == [('foo', 'bar'), ('bar', 'baz')]
        assert list(md.items(multi=True)) == [('foo', 'bar'), ('bar', 'baz'), ('Bar', 'bam')]

    def test_state(self):
        if False:
            return 10
        md = self._multi()
        assert len(md.get_state()) == 3
        assert md == TMultiDict.from_state(md.get_state())
        md2 = TMultiDict()
        assert md != md2
        md2.set_state(md.get_state())
        assert md == md2

class TParent:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.vals = tuple()

    def setter(self, vals):
        if False:
            return 10
        self.vals = vals

    def getter(self):
        if False:
            print('Hello World!')
        return self.vals

class TestMultiDictView:

    def test_modify(self):
        if False:
            i = 10
            return i + 15
        p = TParent()
        tv = multidict.MultiDictView(p.getter, p.setter)
        assert len(tv) == 0
        tv['a'] = 'b'
        assert p.vals == (('a', 'b'),)
        tv['c'] = 'b'
        assert p.vals == (('a', 'b'), ('c', 'b'))
        assert tv['a'] == 'b'

    def test_copy(self):
        if False:
            return 10
        p = TParent()
        tv = multidict.MultiDictView(p.getter, p.setter)
        c = tv.copy()
        assert isinstance(c, multidict.MultiDict)
        assert tv.items() == c.items()
        c['foo'] = 'bar'
        assert tv.items() != c.items()