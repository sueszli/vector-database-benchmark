from pytype.pytd import pep484
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd.parse import parser_test_base
import unittest

class TestPEP484(parser_test_base.ParserTest):
    """Test the visitors in optimize.py."""

    def convert(self, t):
        if False:
            while True:
                i = 10
        'Run ConvertTypingToNative and return the result as a string.'
        return pytd_utils.Print(t.Visit(pep484.ConvertTypingToNative(None)))

    def test_convert_optional(self):
        if False:
            print('Hello World!')
        t = pytd.GenericType(pytd.NamedType('typing.Optional'), (pytd.NamedType('str'),))
        self.assertEqual(self.convert(t), 'Optional[str]')

    def test_convert_union(self):
        if False:
            for i in range(10):
                print('nop')
        t = pytd.GenericType(pytd.NamedType('typing.Union'), (pytd.NamedType('str'), pytd.NamedType('float')))
        self.assertEqual(self.convert(t), 'Union[str, float]')

    def test_convert_list(self):
        if False:
            for i in range(10):
                print('nop')
        t = pytd.NamedType('typing.List')
        self.assertEqual(self.convert(t), 'list')

    def test_convert_tuple(self):
        if False:
            while True:
                i = 10
        t = pytd.NamedType('typing.Tuple')
        self.assertEqual(self.convert(t), 'tuple')

    def test_convert_any(self):
        if False:
            i = 10
            return i + 15
        t = pytd.NamedType('typing.Any')
        self.assertEqual(self.convert(t), 'Any')

    def test_convert_anystr(self):
        if False:
            return 10
        t = pytd.NamedType('typing.AnyStr')
        self.assertEqual(self.convert(t), 'AnyStr')
if __name__ == '__main__':
    unittest.main()