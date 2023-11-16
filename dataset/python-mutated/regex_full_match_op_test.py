"""Tests for RegexFullMatch op from string_ops."""
from absl.testing import parameterized
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

@parameterized.parameters(gen_string_ops.regex_full_match, gen_string_ops.static_regex_full_match)
class RegexFullMatchOpVariantsTest(test.TestCase, parameterized.TestCase):

    @test_util.run_deprecated_v1
    def testRegexFullMatch(self, op):
        if False:
            for i in range(10):
                print('nop')
        values = ['abaaba', 'abcdabcde']
        with self.cached_session():
            input_tensor = constant_op.constant(values, dtypes.string)
            matched = op(input_tensor, 'a.*a').eval()
            self.assertAllEqual([True, False], matched)

    @test_util.run_deprecated_v1
    def testRegexFullMatchTwoDims(self, op):
        if False:
            return 10
        values = [['abaaba', 'abcdabcde'], ['acdcba', 'ebcda']]
        with self.cached_session():
            input_tensor = constant_op.constant(values, dtypes.string)
            matched = op(input_tensor, 'a.*a').eval()
            self.assertAllEqual([[True, False], [True, False]], matched)

    @test_util.run_deprecated_v1
    def testEmptyMatch(self, op):
        if False:
            return 10
        values = ['abc', '1']
        with self.cached_session():
            input_tensor = constant_op.constant(values, dtypes.string)
            matched = op(input_tensor, '').eval()
            self.assertAllEqual([False, False], matched)

    @test_util.run_deprecated_v1
    def testInvalidPattern(self, op):
        if False:
            print('Hello World!')
        values = ['abc', '1']
        with self.cached_session():
            input_tensor = constant_op.constant(values, dtypes.string)
            invalid_pattern = 'A['
            matched = op(input_tensor, invalid_pattern)
            with self.assertRaisesOpError('Invalid pattern'):
                self.evaluate(matched)

class RegexFullMatchOpTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testRegexFullMatchDelegation(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            input_tensor = constant_op.constant('foo', dtypes.string)
            pattern = '[a-z]'
            op = string_ops.regex_full_match(input_tensor, pattern)
            self.assertFalse(op.name.startswith('RegexFullMatch'), op.name)
            pattern_tensor = constant_op.constant('[a-z]*', dtypes.string)
            op_tensor = string_ops.regex_full_match(input_tensor, pattern_tensor)
            self.assertTrue(op_tensor.name.startswith('RegexFullMatch'), op.name)

    @test_util.run_deprecated_v1
    def testStaticRegexFullMatchDelegation(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            input_tensor = constant_op.constant('foo', dtypes.string)
            pattern = '[a-z]*'
            op = string_ops.regex_full_match(input_tensor, pattern)
            self.assertTrue(op.name.startswith('StaticRegexFullMatch'), op.name)
            pattern_tensor = constant_op.constant('[a-z]*', dtypes.string)
            op_vec = string_ops.regex_full_match(input_tensor, pattern_tensor)
            self.assertTrue(op_vec.name.startswith('RegexFullMatch'), op.name)
if __name__ == '__main__':
    test.main()