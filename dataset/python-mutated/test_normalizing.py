import unittest
from collections import UserDict
from robot.utils import normalize, NormalizedDict
from robot.utils.asserts import assert_equal, assert_true, assert_false, assert_raises

class TestNormalize(unittest.TestCase):

    def _verify(self, string, expected, **config):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(normalize(string, **config), expected)

    def test_defaults(self):
        if False:
            i = 10
            return i + 15
        for (inp, exp) in [('', ''), ('            ', ''), (' \n\t\r', ''), ('foo', 'foo'), ('BAR', 'bar'), (' f o o ', 'foo'), ('_BAR', '_bar'), ('Fo OBar\r\n', 'foobar'), ('foo\tbar', 'foobar'), ('\n \n \n \n F o O \t\tBaR \r \r \r   ', 'foobar')]:
            self._verify(inp, exp)

    def test_caseless(self):
        if False:
            while True:
                i = 10
        self._verify('Fo o BaR', 'FooBaR', caseless=False)
        self._verify('Fo o BaR', 'foobar', caseless=True)

    def test_caseless_non_ascii(self):
        if False:
            while True:
                i = 10
        self._verify('Äiti', 'Äiti', caseless=False)
        for mother in ['ÄITI', 'ÄiTi', 'äiti', 'äiTi']:
            self._verify(mother, 'äiti', caseless=True)

    def test_casefold(self):
        if False:
            return 10
        self._verify('ß', 'ss', caseless=True)
        self._verify('Straße', 'strasse', caseless=True)
        self._verify('Straße', 'strae', ignore='ß', caseless=True)
        self._verify('Straße', 'trae', ignore='s', caseless=True)

    def test_spaceless(self):
        if False:
            return 10
        self._verify('Fo o BaR', 'fo o bar', spaceless=False)
        self._verify('Fo o BaR', 'foobar', spaceless=True)

    def test_ignore(self):
        if False:
            print('Hello World!')
        self._verify('Foo_ bar', 'fbar', ignore=['_', 'x', 'o'])
        self._verify('Foo_ bar', 'fbar', ignore=('_', 'x', 'o'))
        self._verify('Foo_ bar', 'fbar', ignore='_xo')
        self._verify('Foo_ bar', 'bar', ignore=['_', 'f', 'o'])
        self._verify('Foo_ bar', 'bar', ignore=['_', 'F', 'O'])
        self._verify('Foo_ bar', 'Fbar', ignore=['_', 'f', 'o'], caseless=False)
        self._verify('Foo_\n bar\n', 'foo_ bar', ignore=['\n'], spaceless=False)

    def test_string_subclass_without_compatible_init(self):
        if False:
            i = 10
            return i + 15

        class BrokenLikeSudsText(str):

            def __new__(cls, value):
                if False:
                    i = 10
                    return i + 15
                return str.__new__(cls, value)
        self._verify(BrokenLikeSudsText('suds.sax.text.Text is BROKEN'), 'suds.sax.text.textisbroken')
        self._verify(BrokenLikeSudsText(''), '')

class TestNormalizedDict(unittest.TestCase):

    def test_default_constructor(self):
        if False:
            i = 10
            return i + 15
        nd = NormalizedDict()
        nd['foo bar'] = 'value'
        assert_equal(nd['foobar'], 'value')
        assert_equal(nd['F  oo\nBar'], 'value')

    def test_initial_values_as_dict(self):
        if False:
            while True:
                i = 10
        nd = NormalizedDict({'key': 'value', 'F O\tO': 'bar'})
        assert_equal(nd['key'], 'value')
        assert_equal(nd['K EY'], 'value')
        assert_equal(nd['foo'], 'bar')

    def test_initial_values_as_name_value_pairs(self):
        if False:
            return 10
        nd = NormalizedDict([('key', 'value'), ('F O\tO', 'bar')])
        assert_equal(nd['key'], 'value')
        assert_equal(nd['K EY'], 'value')
        assert_equal(nd['foo'], 'bar')

    def test_initial_values_as_generator(self):
        if False:
            i = 10
            return i + 15
        nd = NormalizedDict((item for item in [('key', 'value'), ('F O\tO', 'bar')]))
        assert_equal(nd['key'], 'value')
        assert_equal(nd['K EY'], 'value')
        assert_equal(nd['foo'], 'bar')

    def test_setdefault(self):
        if False:
            print('Hello World!')
        nd = NormalizedDict({'a': NormalizedDict()})
        nd.setdefault('a').setdefault('B', []).append(1)
        nd.setdefault('A', 'whatever').setdefault('b', []).append(2)
        assert_equal(nd['a']['b'], [1, 2])
        assert_equal(list(nd), ['a'])
        assert_equal(list(nd['a']), ['B'])

    def test_ignore(self):
        if False:
            return 10
        nd = NormalizedDict(ignore=['_'])
        nd['foo_bar'] = 'value'
        assert_equal(nd['foobar'], 'value')
        assert_equal(nd['F  oo\nB   ___a r'], 'value')

    def test_caseless_and_spaceless(self):
        if False:
            i = 10
            return i + 15
        nd1 = NormalizedDict({'F o o BAR': 'value'})
        nd2 = NormalizedDict({'F o o BAR': 'value'}, caseless=False, spaceless=False)
        assert_equal(nd1['F o o BAR'], 'value')
        assert_equal(nd2['F o o BAR'], 'value')
        nd1['FooBAR'] = 'value 2'
        nd2['FooBAR'] = 'value 2'
        assert_equal(nd1['F o o BAR'], 'value 2')
        assert_equal(nd2['F o o BAR'], 'value')
        assert_equal(nd1['FooBAR'], 'value 2')
        assert_equal(nd2['FooBAR'], 'value 2')
        for key in ['foobar', 'f o o b ar', 'Foo BAR']:
            assert_equal(nd1[key], 'value 2')
            assert_raises(KeyError, nd2.__getitem__, key)
            assert_true(key not in nd2)

    def test_caseless_with_non_ascii(self):
        if False:
            while True:
                i = 10
        nd1 = NormalizedDict({'ä': 1})
        assert_equal(nd1['ä'], 1)
        assert_equal(nd1['Ä'], 1)
        assert_true('Ä' in nd1)
        nd2 = NormalizedDict({'ä': 1}, caseless=False)
        assert_equal(nd2['ä'], 1)
        assert_true('Ä' not in nd2)

    def test_contains(self):
        if False:
            while True:
                i = 10
        nd = NormalizedDict({'Foo': 'bar'})
        assert_true('Foo' in nd and 'foo' in nd and ('FOO' in nd))

    def test_original_keys_are_preserved(self):
        if False:
            i = 10
            return i + 15
        nd = NormalizedDict({'low': 1, 'UP': 2})
        nd['up'] = nd['Spa Ce'] = 3
        assert_equal(list(nd.keys()), ['low', 'Spa Ce', 'UP'])
        assert_equal(list(nd.items()), [('low', 1), ('Spa Ce', 3), ('UP', 3)])

    def test_deleting_items(self):
        if False:
            i = 10
            return i + 15
        nd = NormalizedDict({'A': 1, 'b': 2})
        del nd['A']
        del nd['B']
        assert_equal(nd._data, {})
        assert_equal(list(nd.keys()), [])

    def test_pop(self):
        if False:
            print('Hello World!')
        nd = NormalizedDict({'A': 1, 'b': 2})
        assert_equal(nd.pop('A'), 1)
        assert_equal(nd.pop('B'), 2)
        assert_equal(nd._data, {})
        assert_equal(list(nd.keys()), [])

    def test_pop_with_default(self):
        if False:
            print('Hello World!')
        assert_equal(NormalizedDict().pop('nonex', 'default'), 'default')

    def test_popitem(self):
        if False:
            print('Hello World!')
        items = [(str(i), i) for i in range(9)]
        nd = NormalizedDict(items)
        for i in range(9):
            assert_equal(nd.popitem(), items[i])
        assert_equal(nd._data, {})
        assert_equal(list(nd.keys()), [])

    def test_popitem_empty(self):
        if False:
            print('Hello World!')
        assert_raises(KeyError, NormalizedDict().popitem)

    def test_len(self):
        if False:
            while True:
                i = 10
        nd = NormalizedDict()
        assert_equal(len(nd), 0)
        nd['a'] = nd['b'] = nd['B'] = nd['c'] = 'x'
        assert_equal(len(nd), 3)

    def test_truth_value(self):
        if False:
            print('Hello World!')
        assert_false(NormalizedDict())
        assert_true(NormalizedDict({'a': 1}))

    def test_copy(self):
        if False:
            i = 10
            return i + 15
        nd = NormalizedDict({'a': 1, 'B': 1})
        cd = nd.copy()
        assert_equal(nd, cd)
        assert_equal(nd._data, cd._data)
        assert_equal(nd._keys, cd._keys)
        assert_equal(nd._normalize, cd._normalize)
        nd['C'] = 1
        cd['b'] = 2
        assert_equal(nd._keys, {'a': 'a', 'b': 'B', 'c': 'C'})
        assert_equal(nd._data, {'a': 1, 'b': 1, 'c': 1})
        assert_equal(cd._keys, {'a': 'a', 'b': 'B'})
        assert_equal(cd._data, {'a': 1, 'b': 2})

    def test_copy_with_subclass(self):
        if False:
            for i in range(10):
                print('nop')

        class SubClass(NormalizedDict):
            pass
        assert_true(isinstance(SubClass().copy(), SubClass))

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        nd = NormalizedDict({'a': 1, 'B': 2, 'c': '3', 'd': '"', 'E': 5, 'F': 6})
        expected = '{\'a\': 1, \'B\': 2, \'c\': \'3\', \'d\': \'"\', \'E\': 5, \'F\': 6}'
        assert_equal(str(nd), expected)

    def test_repr(self):
        if False:
            while True:
                i = 10
        assert_equal(repr(NormalizedDict()), 'NormalizedDict()')
        assert_equal(repr(NormalizedDict({'a': None, 'b': '"', 'A': 1})), 'NormalizedDict({\'a\': 1, \'b\': \'"\'})')
        assert_equal(repr(type('Extend', (NormalizedDict,), {})()), 'Extend()')

    def test_unicode(self):
        if False:
            print('Hello World!')
        nd = NormalizedDict({'a': 'ä', 'ä': 'a'})
        assert_equal(str(nd), "{'a': 'ä', 'ä': 'a'}")

    def test_update(self):
        if False:
            return 10
        nd = NormalizedDict({'a': 1, 'b': 1, 'c': 1})
        nd.update({'b': 2, 'C': 2, 'D': 2})
        for c in 'bcd':
            assert_equal(nd[c], 2)
            assert_equal(nd[c.upper()], 2)
        keys = list(nd)
        assert_true('b' in keys)
        assert_true('c' in keys)
        assert_true('C' not in keys)
        assert_true('d' not in keys)
        assert_true('D' in keys)

    def test_update_using_another_norm_dict(self):
        if False:
            i = 10
            return i + 15
        nd = NormalizedDict({'a': 1, 'b': 1})
        nd.update(NormalizedDict({'B': 2, 'C': 2}))
        for c in 'bc':
            assert_equal(nd[c], 2)
            assert_equal(nd[c.upper()], 2)
        keys = list(nd)
        assert_true('b' in keys)
        assert_true('B' not in keys)
        assert_true('c' not in keys)
        assert_true('C' in keys)

    def test_update_with_kwargs(self):
        if False:
            print('Hello World!')
        nd = NormalizedDict({'a': 0, 'c': 1})
        nd.update({'b': 2, 'c': 3}, b=4, d=5)
        for (k, v) in [('a', 0), ('b', 4), ('c', 3), ('d', 5)]:
            assert_equal(nd[k], v)
            assert_equal(nd[k.upper()], v)
            assert_true(k in nd)
            assert_true(k.upper() in nd)
            assert_true(k in nd.keys())

    def test_iter(self):
        if False:
            print('Hello World!')
        keys = list('123_aBcDeF')
        nd = NormalizedDict(((k, 1) for k in keys))
        assert_equal(list(nd), keys)
        assert_equal([key for key in nd], keys)

    def test_keys_are_sorted(self):
        if False:
            return 10
        nd = NormalizedDict(((c, None) for c in 'aBcDeFg123XyZ___'))
        assert_equal(list(nd.keys()), list('123_aBcDeFgXyZ'))
        assert_equal(list(nd), list('123_aBcDeFgXyZ'))

    def test_keys_values_and_items_are_returned_in_same_order(self):
        if False:
            return 10
        nd = NormalizedDict()
        for (i, c) in enumerate('abcdefghijklmnopqrstuvwxyz0123456789!"#%&/()=?'):
            nd[c.upper()] = i
            nd[c + str(i)] = 1
        assert_equal(list(nd.items()), list(zip(nd.keys(), nd.values())))

    def test_eq(self):
        if False:
            return 10
        self._verify_eq(NormalizedDict(), NormalizedDict())

    def test_eq_with_normal_dict(self):
        if False:
            while True:
                i = 10
        self._verify_eq(NormalizedDict(), {})

    def test_eq_with_user_dict(self):
        if False:
            print('Hello World!')
        self._verify_eq(NormalizedDict(), UserDict())

    def _verify_eq(self, d1, d2):
        if False:
            for i in range(10):
                print('nop')
        assert_true(d1 == d1 == d2 == d2)
        d1['a'] = 1
        assert_true(d1 == d1 != d2 == d2)
        d2['a'] = 1
        assert_true(d1 == d1 == d2 == d2)
        d1['B'] = 1
        d2['B'] = 1
        assert_true(d1 == d1 == d2 == d2)
        d1['c'] = d2['C'] = 1
        d1['D'] = d2['d'] = 1
        assert_true(d1 == d1 == d2 == d2)

    def test_eq_with_other_objects(self):
        if False:
            return 10
        nd = NormalizedDict()
        for other in ['string', 2, None, [], self.test_clear]:
            assert_false(nd == other, other)
            assert_true(nd != other, other)

    def test_ne(self):
        if False:
            print('Hello World!')
        assert_false(NormalizedDict() != NormalizedDict())
        assert_false(NormalizedDict({'a': 1}) != NormalizedDict({'a': 1}))
        assert_false(NormalizedDict({'a': 1}) != NormalizedDict({'A': 1}))

    def test_hash(self):
        if False:
            print('Hello World!')
        assert_raises(TypeError, hash, NormalizedDict())

    def test_clear(self):
        if False:
            for i in range(10):
                print('nop')
        nd = NormalizedDict({'a': 1, 'B': 2})
        nd.clear()
        assert_equal(nd._data, {})
        assert_equal(nd._keys, {})
if __name__ == '__main__':
    unittest.main()