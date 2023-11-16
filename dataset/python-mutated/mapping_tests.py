import unittest
import collections
import sys

class BasicTestMappingProtocol(unittest.TestCase):
    type2test = None

    def _reference(self):
        if False:
            print('Hello World!')
        'Return a dictionary of values which are invariant by storage\n        in the object under test.'
        return {'1': '2', 'key1': 'value1', 'key2': (1, 2, 3)}

    def _empty_mapping(self):
        if False:
            print('Hello World!')
        'Return an empty mapping object'
        return self.type2test()

    def _full_mapping(self, data):
        if False:
            i = 10
            return i + 15
        'Return a mapping object with the value contained in data\n        dictionary'
        x = self._empty_mapping()
        for (key, value) in data.items():
            x[key] = value
        return x

    def __init__(self, *args, **kw):
        if False:
            while True:
                i = 10
        unittest.TestCase.__init__(self, *args, **kw)
        self.reference = self._reference().copy()
        (key, value) = self.reference.popitem()
        self.other = {key: value}
        (key, value) = self.reference.popitem()
        self.inmapping = {key: value}
        self.reference[key] = value

    def test_read(self):
        if False:
            return 10
        p = self._empty_mapping()
        p1 = dict(p)
        d = self._full_mapping(self.reference)
        if d is p:
            p = p1
        for (key, value) in self.reference.items():
            self.assertEqual(d[key], value)
        knownkey = list(self.other.keys())[0]
        self.assertRaises(KeyError, lambda : d[knownkey])
        self.assertEqual(len(p), 0)
        self.assertEqual(len(d), len(self.reference))
        for k in self.reference:
            self.assertIn(k, d)
        for k in self.other:
            self.assertNotIn(k, d)
        self.assertEqual(p, p)
        self.assertEqual(d, d)
        self.assertNotEqual(p, d)
        self.assertNotEqual(d, p)
        if p:
            self.fail('Empty mapping must compare to False')
        if not d:
            self.fail('Full mapping must compare to True')

        def check_iterandlist(iter, lst, ref):
            if False:
                i = 10
                return i + 15
            self.assertTrue(hasattr(iter, '__next__'))
            self.assertTrue(hasattr(iter, '__iter__'))
            x = list(iter)
            self.assertTrue(set(x) == set(lst) == set(ref))
        check_iterandlist(iter(d.keys()), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d.values()), list(d.values()), self.reference.values())
        check_iterandlist(iter(d.items()), list(d.items()), self.reference.items())
        (key, value) = next(iter(d.items()))
        (knownkey, knownvalue) = next(iter(self.other.items()))
        self.assertEqual(d.get(key, knownvalue), value)
        self.assertEqual(d.get(knownkey, knownvalue), knownvalue)
        self.assertNotIn(knownkey, d)

    def test_write(self):
        if False:
            return 10
        p = self._empty_mapping()
        for (key, value) in self.reference.items():
            p[key] = value
            self.assertEqual(p[key], value)
        for key in self.reference.keys():
            del p[key]
            self.assertRaises(KeyError, lambda : p[key])
        p = self._empty_mapping()
        p.update(self.reference)
        self.assertEqual(dict(p), self.reference)
        items = list(p.items())
        p = self._empty_mapping()
        p.update(items)
        self.assertEqual(dict(p), self.reference)
        d = self._full_mapping(self.reference)
        (key, value) = next(iter(d.items()))
        (knownkey, knownvalue) = next(iter(self.other.items()))
        self.assertEqual(d.setdefault(key, knownvalue), value)
        self.assertEqual(d[key], value)
        self.assertEqual(d.setdefault(knownkey, knownvalue), knownvalue)
        self.assertEqual(d[knownkey], knownvalue)
        self.assertEqual(d.pop(knownkey), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertRaises(KeyError, d.pop, knownkey)
        default = 909
        d[knownkey] = knownvalue
        self.assertEqual(d.pop(knownkey, default), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertEqual(d.pop(knownkey, default), default)
        (key, value) = d.popitem()
        self.assertNotIn(key, d)
        self.assertEqual(value, self.reference[key])
        p = self._empty_mapping()
        self.assertRaises(KeyError, p.popitem)

    def test_constructor(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self._empty_mapping(), self._empty_mapping())

    def test_bool(self):
        if False:
            return 10
        self.assertTrue(not self._empty_mapping())
        self.assertTrue(self.reference)
        self.assertTrue(bool(self._empty_mapping()) is False)
        self.assertTrue(bool(self.reference) is True)

    def test_keys(self):
        if False:
            return 10
        d = self._empty_mapping()
        self.assertEqual(list(d.keys()), [])
        d = self.reference
        self.assertIn(list(self.inmapping.keys())[0], d.keys())
        self.assertNotIn(list(self.other.keys())[0], d.keys())
        self.assertRaises(TypeError, d.keys, None)

    def test_values(self):
        if False:
            while True:
                i = 10
        d = self._empty_mapping()
        self.assertEqual(list(d.values()), [])
        self.assertRaises(TypeError, d.values, None)

    def test_items(self):
        if False:
            while True:
                i = 10
        d = self._empty_mapping()
        self.assertEqual(list(d.items()), [])
        self.assertRaises(TypeError, d.items, None)

    def test_len(self):
        if False:
            i = 10
            return i + 15
        d = self._empty_mapping()
        self.assertEqual(len(d), 0)

    def test_getitem(self):
        if False:
            return 10
        d = self.reference
        self.assertEqual(d[list(self.inmapping.keys())[0]], list(self.inmapping.values())[0])
        self.assertRaises(TypeError, d.__getitem__)

    def test_update(self):
        if False:
            while True:
                i = 10
        d = self._empty_mapping()
        d.update(self.other)
        self.assertEqual(list(d.items()), list(self.other.items()))
        d = self._empty_mapping()
        d.update()
        self.assertEqual(d, self._empty_mapping())
        d = self._empty_mapping()
        d.update(self.other.items())
        self.assertEqual(list(d.items()), list(self.other.items()))
        d = self._empty_mapping()
        d.update(self.other.items())
        self.assertEqual(list(d.items()), list(self.other.items()))
        self.assertRaises((TypeError, AttributeError), d.update, 42)
        outerself = self

        class SimpleUserDict:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.d = outerself.reference

            def keys(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.d.keys()

            def __getitem__(self, i):
                if False:
                    i = 10
                    return i + 15
                return self.d[i]
        d.clear()
        d.update(SimpleUserDict())
        i1 = sorted(d.items())
        i2 = sorted(self.reference.items())
        self.assertEqual(i1, i2)

        class Exc(Exception):
            pass
        d = self._empty_mapping()

        class FailingUserDict:

            def keys(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise Exc
        self.assertRaises(Exc, d.update, FailingUserDict())
        d.clear()

        class FailingUserDict:

            def keys(self):
                if False:
                    print('Hello World!')

                class BogonIter:

                    def __init__(self):
                        if False:
                            i = 10
                            return i + 15
                        self.i = 1

                    def __iter__(self):
                        if False:
                            return 10
                        return self

                    def __next__(self):
                        if False:
                            return 10
                        if self.i:
                            self.i = 0
                            return 'a'
                        raise Exc
                return BogonIter()

            def __getitem__(self, key):
                if False:
                    i = 10
                    return i + 15
                return key
        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:

            def keys(self):
                if False:
                    while True:
                        i = 10

                class BogonIter:

                    def __init__(self):
                        if False:
                            for i in range(10):
                                print('nop')
                        self.i = ord('a')

                    def __iter__(self):
                        if False:
                            print('Hello World!')
                        return self

                    def __next__(self):
                        if False:
                            for i in range(10):
                                print('nop')
                        if self.i <= ord('z'):
                            rtn = chr(self.i)
                            self.i += 1
                            return rtn
                        raise StopIteration
                return BogonIter()

            def __getitem__(self, key):
                if False:
                    i = 10
                    return i + 15
                raise Exc
        self.assertRaises(Exc, d.update, FailingUserDict())
        d = self._empty_mapping()

        class badseq(object):

            def __iter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self

            def __next__(self):
                if False:
                    i = 10
                    return i + 15
                raise Exc()
        self.assertRaises(Exc, d.update, badseq())
        self.assertRaises(ValueError, d.update, [(1, 2, 3)])

    def test_get(self):
        if False:
            return 10
        d = self._empty_mapping()
        self.assertTrue(d.get(list(self.other.keys())[0]) is None)
        self.assertEqual(d.get(list(self.other.keys())[0], 3), 3)
        d = self.reference
        self.assertTrue(d.get(list(self.other.keys())[0]) is None)
        self.assertEqual(d.get(list(self.other.keys())[0], 3), 3)
        self.assertEqual(d.get(list(self.inmapping.keys())[0]), list(self.inmapping.values())[0])
        self.assertEqual(d.get(list(self.inmapping.keys())[0], 3), list(self.inmapping.values())[0])
        self.assertRaises(TypeError, d.get)
        self.assertRaises(TypeError, d.get, None, None, None)

    def test_setdefault(self):
        if False:
            i = 10
            return i + 15
        d = self._empty_mapping()
        self.assertRaises(TypeError, d.setdefault)

    def test_popitem(self):
        if False:
            for i in range(10):
                print('nop')
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)
        self.assertRaises(TypeError, d.popitem, 42)

    def test_pop(self):
        if False:
            return 10
        d = self._empty_mapping()
        (k, v) = list(self.inmapping.items())[0]
        d[k] = v
        self.assertRaises(KeyError, d.pop, list(self.other.keys())[0])
        self.assertEqual(d.pop(k), v)
        self.assertEqual(len(d), 0)
        self.assertRaises(KeyError, d.pop, k)

class TestMappingProtocol(BasicTestMappingProtocol):

    def test_constructor(self):
        if False:
            i = 10
            return i + 15
        BasicTestMappingProtocol.test_constructor(self)
        self.assertTrue(self._empty_mapping() is not self._empty_mapping())
        self.assertEqual(self.type2test(x=1, y=2), {'x': 1, 'y': 2})

    def test_bool(self):
        if False:
            while True:
                i = 10
        BasicTestMappingProtocol.test_bool(self)
        self.assertTrue(not self._empty_mapping())
        self.assertTrue(self._full_mapping({'x': 'y'}))
        self.assertTrue(bool(self._empty_mapping()) is False)
        self.assertTrue(bool(self._full_mapping({'x': 'y'})) is True)

    def test_keys(self):
        if False:
            return 10
        BasicTestMappingProtocol.test_keys(self)
        d = self._empty_mapping()
        self.assertEqual(list(d.keys()), [])
        d = self._full_mapping({'a': 1, 'b': 2})
        k = d.keys()
        self.assertIn('a', k)
        self.assertIn('b', k)
        self.assertNotIn('c', k)

    def test_values(self):
        if False:
            while True:
                i = 10
        BasicTestMappingProtocol.test_values(self)
        d = self._full_mapping({1: 2})
        self.assertEqual(list(d.values()), [2])

    def test_items(self):
        if False:
            print('Hello World!')
        BasicTestMappingProtocol.test_items(self)
        d = self._full_mapping({1: 2})
        self.assertEqual(list(d.items()), [(1, 2)])

    def test_contains(self):
        if False:
            print('Hello World!')
        d = self._empty_mapping()
        self.assertNotIn('a', d)
        self.assertTrue(not 'a' in d)
        self.assertTrue('a' not in d)
        d = self._full_mapping({'a': 1, 'b': 2})
        self.assertIn('a', d)
        self.assertIn('b', d)
        self.assertNotIn('c', d)
        self.assertRaises(TypeError, d.__contains__)

    def test_len(self):
        if False:
            for i in range(10):
                print('nop')
        BasicTestMappingProtocol.test_len(self)
        d = self._full_mapping({'a': 1, 'b': 2})
        self.assertEqual(len(d), 2)

    def test_getitem(self):
        if False:
            return 10
        BasicTestMappingProtocol.test_getitem(self)
        d = self._full_mapping({'a': 1, 'b': 2})
        self.assertEqual(d['a'], 1)
        self.assertEqual(d['b'], 2)
        d['c'] = 3
        d['a'] = 4
        self.assertEqual(d['c'], 3)
        self.assertEqual(d['a'], 4)
        del d['b']
        self.assertEqual(d, {'a': 4, 'c': 3})
        self.assertRaises(TypeError, d.__getitem__)

    def test_clear(self):
        if False:
            i = 10
            return i + 15
        d = self._full_mapping({1: 1, 2: 2, 3: 3})
        d.clear()
        self.assertEqual(d, {})
        self.assertRaises(TypeError, d.clear, None)

    def test_update(self):
        if False:
            while True:
                i = 10
        BasicTestMappingProtocol.test_update(self)
        d = self._empty_mapping()
        d.update({1: 100})
        d.update({2: 20})
        d.update({1: 1, 2: 2, 3: 3})
        self.assertEqual(d, {1: 1, 2: 2, 3: 3})
        d.update()
        self.assertEqual(d, {1: 1, 2: 2, 3: 3})
        d = self._empty_mapping()
        d.update(x=100)
        d.update(y=20)
        d.update(x=1, y=2, z=3)
        self.assertEqual(d, {'x': 1, 'y': 2, 'z': 3})
        d = self._empty_mapping()
        d.update([('x', 100), ('y', 20)])
        self.assertEqual(d, {'x': 100, 'y': 20})
        d = self._empty_mapping()
        d.update([('x', 100), ('y', 20)], x=1, y=2)
        self.assertEqual(d, {'x': 1, 'y': 2})
        d = self._full_mapping({1: 3, 2: 4})
        d.update(self._full_mapping({1: 2, 3: 4, 5: 6}).items())
        self.assertEqual(d, {1: 2, 2: 4, 3: 4, 5: 6})

        class SimpleUserDict:

            def __init__(self):
                if False:
                    return 10
                self.d = {1: 1, 2: 2, 3: 3}

            def keys(self):
                if False:
                    i = 10
                    return i + 15
                return self.d.keys()

            def __getitem__(self, i):
                if False:
                    print('Hello World!')
                return self.d[i]
        d.clear()
        d.update(SimpleUserDict())
        self.assertEqual(d, {1: 1, 2: 2, 3: 3})

    def test_fromkeys(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.type2test.fromkeys('abc'), {'a': None, 'b': None, 'c': None})
        d = self._empty_mapping()
        self.assertTrue(not d.fromkeys('abc') is d)
        self.assertEqual(d.fromkeys('abc'), {'a': None, 'b': None, 'c': None})
        self.assertEqual(d.fromkeys((4, 5), 0), {4: 0, 5: 0})
        self.assertEqual(d.fromkeys([]), {})

        def g():
            if False:
                print('Hello World!')
            yield 1
        self.assertEqual(d.fromkeys(g()), {1: None})
        self.assertRaises(TypeError, {}.fromkeys, 3)

        class dictlike(self.type2test):
            pass
        self.assertEqual(dictlike.fromkeys('a'), {'a': None})
        self.assertEqual(dictlike().fromkeys('a'), {'a': None})
        self.assertTrue(dictlike.fromkeys('a').__class__ is dictlike)
        self.assertTrue(dictlike().fromkeys('a').__class__ is dictlike)
        self.assertTrue(type(dictlike.fromkeys('a')) is dictlike)

        class mydict(self.type2test):

            def __new__(cls):
                if False:
                    print('Hello World!')
                return collections.UserDict()
        ud = mydict.fromkeys('ab')
        self.assertEqual(ud, {'a': None, 'b': None})
        self.assertIsInstance(ud, collections.UserDict)
        self.assertRaises(TypeError, dict.fromkeys)

        class Exc(Exception):
            pass

        class baddict1(self.type2test):

            def __init__(self, *args, **kwargs):
                if False:
                    return 10
                raise Exc()
        self.assertRaises(Exc, baddict1.fromkeys, [1])

        class BadSeq(object):

            def __iter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self

            def __next__(self):
                if False:
                    while True:
                        i = 10
                raise Exc()
        self.assertRaises(Exc, self.type2test.fromkeys, BadSeq())

        class baddict2(self.type2test):

            def __setitem__(self, key, value):
                if False:
                    i = 10
                    return i + 15
                raise Exc()
        self.assertRaises(Exc, baddict2.fromkeys, [1])

    def test_copy(self):
        if False:
            print('Hello World!')
        d = self._full_mapping({1: 1, 2: 2, 3: 3})
        self.assertEqual(d.copy(), {1: 1, 2: 2, 3: 3})
        d = self._empty_mapping()
        self.assertEqual(d.copy(), d)
        self.assertIsInstance(d.copy(), d.__class__)
        self.assertRaises(TypeError, d.copy, None)

    def test_get(self):
        if False:
            return 10
        BasicTestMappingProtocol.test_get(self)
        d = self._empty_mapping()
        self.assertTrue(d.get('c') is None)
        self.assertEqual(d.get('c', 3), 3)
        d = self._full_mapping({'a': 1, 'b': 2})
        self.assertTrue(d.get('c') is None)
        self.assertEqual(d.get('c', 3), 3)
        self.assertEqual(d.get('a'), 1)
        self.assertEqual(d.get('a', 3), 1)

    def test_setdefault(self):
        if False:
            i = 10
            return i + 15
        BasicTestMappingProtocol.test_setdefault(self)
        d = self._empty_mapping()
        self.assertTrue(d.setdefault('key0') is None)
        d.setdefault('key0', [])
        self.assertTrue(d.setdefault('key0') is None)
        d.setdefault('key', []).append(3)
        self.assertEqual(d['key'][0], 3)
        d.setdefault('key', []).append(4)
        self.assertEqual(len(d['key']), 2)

    def test_popitem(self):
        if False:
            i = 10
            return i + 15
        BasicTestMappingProtocol.test_popitem(self)
        for copymode in (-1, +1):
            for log2size in range(12):
                size = 2 ** log2size
                a = self._empty_mapping()
                b = self._empty_mapping()
                for i in range(size):
                    a[repr(i)] = i
                    if copymode < 0:
                        b[repr(i)] = i
                if copymode > 0:
                    b = a.copy()
                for i in range(size):
                    (ka, va) = ta = a.popitem()
                    self.assertEqual(va, int(ka))
                    (kb, vb) = tb = b.popitem()
                    self.assertEqual(vb, int(kb))
                    self.assertTrue(not (copymode < 0 and ta != tb))
                self.assertTrue(not a)
                self.assertTrue(not b)

    def test_pop(self):
        if False:
            for i in range(10):
                print('nop')
        BasicTestMappingProtocol.test_pop(self)
        d = self._empty_mapping()
        (k, v) = ('abc', 'def')
        self.assertEqual(d.pop(k, v), v)
        d[k] = v
        self.assertEqual(d.pop(k, 1), v)

class TestHashMappingProtocol(TestMappingProtocol):

    def test_getitem(self):
        if False:
            return 10
        TestMappingProtocol.test_getitem(self)

        class Exc(Exception):
            pass

        class BadEq(object):

            def __eq__(self, other):
                if False:
                    print('Hello World!')
                raise Exc()

            def __hash__(self):
                if False:
                    while True:
                        i = 10
                return 24
        d = self._empty_mapping()
        d[BadEq()] = 42
        self.assertRaises(KeyError, d.__getitem__, 23)

        class BadHash(object):
            fail = False

            def __hash__(self):
                if False:
                    print('Hello World!')
                if self.fail:
                    raise Exc()
                else:
                    return 42
        d = self._empty_mapping()
        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.__getitem__, x)

    def test_fromkeys(self):
        if False:
            while True:
                i = 10
        TestMappingProtocol.test_fromkeys(self)

        class mydict(self.type2test):

            def __new__(cls):
                if False:
                    return 10
                return collections.UserDict()
        ud = mydict.fromkeys('ab')
        self.assertEqual(ud, {'a': None, 'b': None})
        self.assertIsInstance(ud, collections.UserDict)

    def test_pop(self):
        if False:
            print('Hello World!')
        TestMappingProtocol.test_pop(self)

        class Exc(Exception):
            pass

        class BadHash(object):
            fail = False

            def __hash__(self):
                if False:
                    while True:
                        i = 10
                if self.fail:
                    raise Exc()
                else:
                    return 42
        d = self._empty_mapping()
        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.pop, x)

    def test_mutatingiteration(self):
        if False:
            return 10
        d = self._empty_mapping()
        d[1] = 1
        try:
            count = 0
            for i in d:
                d[i + 1] = 1
                if count >= 1:
                    self.fail("changing dict size during iteration doesn't raise Error")
                count += 1
        except RuntimeError:
            pass

    def test_repr(self):
        if False:
            while True:
                i = 10
        d = self._empty_mapping()
        self.assertEqual(repr(d), '{}')
        d[1] = 2
        self.assertEqual(repr(d), '{1: 2}')
        d = self._empty_mapping()
        d[1] = d
        self.assertEqual(repr(d), '{1: {...}}')

        class Exc(Exception):
            pass

        class BadRepr(object):

            def __repr__(self):
                if False:
                    while True:
                        i = 10
                raise Exc()
        d = self._full_mapping({1: BadRepr()})
        self.assertRaises(Exc, repr, d)

    def test_repr_deep(self):
        if False:
            while True:
                i = 10
        d = self._empty_mapping()
        for i in range(sys.getrecursionlimit() + 100):
            d0 = d
            d = self._empty_mapping()
            d[1] = d0
        self.assertRaises(RecursionError, repr, d)

    def test_eq(self):
        if False:
            print('Hello World!')
        self.assertEqual(self._empty_mapping(), self._empty_mapping())
        self.assertEqual(self._full_mapping({1: 2}), self._full_mapping({1: 2}))

        class Exc(Exception):
            pass

        class BadCmp(object):

            def __eq__(self, other):
                if False:
                    return 10
                raise Exc()

            def __hash__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 1
        d1 = self._full_mapping({BadCmp(): 1})
        d2 = self._full_mapping({1: 1})
        self.assertRaises(Exc, lambda : BadCmp() == 1)
        self.assertRaises(Exc, lambda : d1 == d2)

    def test_setdefault(self):
        if False:
            return 10
        TestMappingProtocol.test_setdefault(self)

        class Exc(Exception):
            pass

        class BadHash(object):
            fail = False

            def __hash__(self):
                if False:
                    return 10
                if self.fail:
                    raise Exc()
                else:
                    return 42
        d = self._empty_mapping()
        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.setdefault, x, [])