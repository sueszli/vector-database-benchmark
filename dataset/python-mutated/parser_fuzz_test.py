"""Fuzz tests for the parser module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import Levenshtein
import six

class ParserFuzzTest(testutils.BaseTestCase):

    @settings(max_examples=10000)
    @given(st.text(min_size=1))
    @example('True')
    @example('"test\\t\\t\\a\\\\a"')
    @example(' "test\\t\\t\\a\\\\a"   ')
    @example('"(1, 2)"')
    @example('(1, 2)')
    @example('(1,                   2)')
    @example('(1,       2) ')
    @example('a,b,c,d')
    @example('(a,b,c,d)')
    @example('[a,b,c,d]')
    @example('{a,b,c,d}')
    @example('test:(a,b,c,d)')
    @example('{test:(a,b,c,d)}')
    @example('{test:a,b,c,d}')
    @example('{test:a,b:(c,d)}')
    @example('0,')
    @example('#')
    @example('A#00000')
    @example('\x80')
    @example(100 * '[' + '0')
    @example('\r\r\r\r1\r\r')
    def testDefaultParseValueFuzz(self, value):
        if False:
            return 10
        try:
            result = parser.DefaultParseValue(value)
        except TypeError:
            if u'\x00' in value:
                return
            raise
        except MemoryError:
            if len(value) > 100:
                return
            raise
        try:
            uvalue = six.text_type(value)
            uresult = six.text_type(result)
        except UnicodeDecodeError:
            return
        distance = Levenshtein.distance(uresult, uvalue)
        max_distance = 2 + sum((c.isspace() for c in value)) + value.count('"') + value.count("'") + 3 * (value.count(',') + 1) + 3 * value.count(':') + 2 * value.count('\\')
        if '#' in value:
            max_distance += len(value) - value.index('#')
        if not isinstance(result, six.string_types):
            max_distance += value.count('0')
        if '{' not in value:
            self.assertLessEqual(distance, max_distance, (distance, max_distance, uvalue, uresult))
if __name__ == '__main__':
    testutils.main()