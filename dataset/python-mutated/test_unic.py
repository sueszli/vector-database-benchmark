import unittest
import re
from robot.utils import safe_str, prepr, DotDict
from robot.utils.asserts import assert_equal, assert_true

class TestSafeStr(unittest.TestCase):

    def test_unicode_nfc_and_nfd_decomposition_equality(self):
        if False:
            print('Hello World!')
        import unicodedata
        text = 'Hyvä'
        assert_equal(safe_str(unicodedata.normalize('NFC', text)), text)
        assert_equal(safe_str(unicodedata.normalize('NFD', text)), text)

    def test_object_containing_unicode_repr(self):
        if False:
            while True:
                i = 10
        assert_equal(safe_str(NonAsciiRepr()), 'Hyvä')

    def test_list_with_objects_containing_unicode_repr(self):
        if False:
            while True:
                i = 10
        objects = [NonAsciiRepr(), NonAsciiRepr()]
        result = safe_str(objects)
        assert_equal(result, '[Hyvä, Hyvä]')

    def test_bytes_below_128(self):
        if False:
            i = 10
            return i + 15
        assert_equal(safe_str('\x00-\x01-\x02-\x7f'), '\x00-\x01-\x02-\x7f')

    def test_bytes_above_128(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(safe_str(b'hyv\xe4'), 'hyv\\xe4')
        assert_equal(safe_str(b'\x00-\x01-\x02-\xe4'), '\x00-\x01-\x02-\\xe4')

    def test_bytes_with_newlines_tabs_etc(self):
        if False:
            print('Hello World!')
        assert_equal(safe_str(b"\x00\xe4\n\t\r\\'"), "\x00\\xe4\n\t\r\\'")

    def test_bytearray(self):
        if False:
            i = 10
            return i + 15
        assert_equal(safe_str(bytearray(b'hyv\xe4')), 'hyv\\xe4')
        assert_equal(safe_str(bytearray(b'\x00-\x01-\x02-\xe4')), '\x00-\x01-\x02-\\xe4')
        assert_equal(safe_str(bytearray(b"\x00\xe4\n\t\r\\'")), "\x00\\xe4\n\t\r\\'")

    def test_failure_in_str(self):
        if False:
            i = 10
            return i + 15
        failing = StrFails()
        assert_equal(safe_str(failing), failing.unrepr)

class TestPrettyRepr(unittest.TestCase):

    def _verify(self, item, expected=None, **config):
        if False:
            print('Hello World!')
        if not expected:
            expected = repr(item).lstrip('')
        assert_equal(prepr(item, **config), expected)
        if isinstance(item, (str, bytes)) and (not config):
            assert_equal(prepr([item]), '[%s]' % expected)
            assert_equal(prepr((item,)), '(%s,)' % expected)
            assert_equal(prepr({item: item}), '{%s: %s}' % (expected, expected))
            assert_equal(prepr({item}), '{%s}' % expected)

    def test_ascii_string(self):
        if False:
            return 10
        self._verify('foo', "'foo'")
        self._verify("f'o'o", '"f\'o\'o"')

    def test_non_ascii_string(self):
        if False:
            return 10
        self._verify('hyvä', "'hyvä'")

    def test_string_in_nfd(self):
        if False:
            print('Hello World!')
        self._verify('hyvä', "'hyvä'")

    def test_ascii_bytes(self):
        if False:
            i = 10
            return i + 15
        self._verify(b'ascii', "b'ascii'")

    def test_non_ascii_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        self._verify(b'non-\xe4scii', "b'non-\\xe4scii'")

    def test_bytearray(self):
        if False:
            print('Hello World!')
        self._verify(bytearray(b'foo'), "bytearray(b'foo')")

    def test_non_strings(self):
        if False:
            for i in range(10):
                print('nop')
        for inp in [1, -2.0, True, None, -2.0, (), [], {}, StrFails()]:
            self._verify(inp)

    def test_failing_repr(self):
        if False:
            for i in range(10):
                print('nop')
        failing = ReprFails()
        self._verify(failing, failing.unrepr)

    def test_non_ascii_repr(self):
        if False:
            i = 10
            return i + 15
        obj = NonAsciiRepr()
        self._verify(obj, 'Hyvä')

    def test_bytes_repr(self):
        if False:
            for i in range(10):
                print('nop')
        obj = BytesRepr()
        self._verify(obj, obj.unrepr)

    def test_collections(self):
        if False:
            return 10
        self._verify(['foo', b'bar', 3], "['foo', b'bar', 3]")
        self._verify(['foo', b'b\xe4r', ('x', b'y')], "['foo', b'b\\xe4r', ('x', b'y')]")
        self._verify({'x': b'\xe4'}, "{'x': b'\\xe4'}")
        self._verify(['ä'], "['ä']")
        self._verify({'ä'}, "{'ä'}")

    def test_dont_sort_dicts_by_default(self):
        if False:
            i = 10
            return i + 15
        self._verify({'x': 1, 'D': 2, 'ä': 3, 'G': 4, 'a': 5}, "{'x': 1, 'D': 2, 'ä': 3, 'G': 4, 'a': 5}")
        self._verify({'a': 1, 1: 'a'}, "{'a': 1, 1: 'a'}")

    def test_allow_sorting_dicts(self):
        if False:
            return 10
        self._verify({'x': 1, 'D': 2, 'ä': 3, 'G': 4, 'a': 5}, "{'D': 2, 'G': 4, 'a': 5, 'x': 1, 'ä': 3}", sort_dicts=True)
        self._verify({'a': 1, 1: 'a'}, "{1: 'a', 'a': 1}", sort_dicts=True)

    def test_dotdict(self):
        if False:
            for i in range(10):
                print('nop')
        self._verify(DotDict({'x': b'\xe4'}), "{'x': b'\\xe4'}")

    def test_recursive(self):
        if False:
            print('Hello World!')
        x = [1, 2]
        x.append(x)
        match = re.match('\\[1, 2. <Recursion on list with id=\\d+>]', prepr(x))
        assert_true(match is not None)

    def test_split_big_collections(self):
        if False:
            return 10
        self._verify(list(range(20)))
        self._verify(list(range(100)), width=400)
        self._verify(list(range(100)), '[%s]' % ',\n '.join((str(i) for i in range(100))))
        self._verify(['Hello, world!'] * 4, '[%s]' % ', '.join(["'Hello, world!'"] * 4))
        self._verify(['Hello, world!'] * 25, '[%s]' % ', '.join(["'Hello, world!'"] * 25), width=500)
        self._verify(['Hello, world!'] * 25, '[%s]' % ',\n '.join(["'Hello, world!'"] * 25))

    def test_dont_split_long_strings(self):
        if False:
            while True:
                i = 10
        self._verify(' '.join(['Hello world!'] * 1000))
        self._verify(b' '.join([b'Hello world!'] * 1000), "b'%s'" % ' '.join(['Hello world!'] * 1000))
        self._verify(bytearray(b' '.join([b'Hello world!'] * 1000)))

class UnRepr:
    error = 'This, of course, should never happen...'

    @property
    def unrepr(self):
        if False:
            i = 10
            return i + 15
        return self.format(type(self).__name__, self.error)

    @staticmethod
    def format(name, error):
        if False:
            i = 10
            return i + 15
        return '<Unrepresentable object %s. Error: %s>' % (name, error)

class StrFails(UnRepr):

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError(self.error)

class ReprFails(UnRepr):

    def __repr__(self):
        if False:
            while True:
                i = 10
        raise RuntimeError(self.error)

class NonAsciiRepr(UnRepr):

    def __init__(self):
        if False:
            return 10
        try:
            repr(self)
        except UnicodeEncodeError as err:
            self.error = f'UnicodeEncodeError: {err}'

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'Hyvä'

class BytesRepr(UnRepr):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            repr(self)
        except TypeError as err:
            self.error = f'TypeError: {err}'

    def __repr__(self):
        if False:
            return 10
        return b'Hyv\xe4'
if __name__ == '__main__':
    unittest.main()