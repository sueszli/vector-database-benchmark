import os
import sys
import torch
from typing import List
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestStringFormatting(JitTestCase):

    def test_modulo_operator(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(dividend: int, divisor: int) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return dividend % divisor
        self.checkScript(fn, (5, 2))

    def test_string_interpolation_with_string_placeholder_and_string_variable(self):
        if False:
            return 10

        def fn(arg1: str):
            if False:
                print('Hello World!')
            return '%s in template' % arg1
        self.checkScript(fn, ('foo',))

    def test_string_interpolation_with_string_placeholder_and_format_string_variable(self):
        if False:
            return 10

        def fn(arg1: str):
            if False:
                while True:
                    i = 10
            return arg1 % 'foo'
        self.checkScript(fn, ('%s in template',))

    def test_string_interpolation_with_double_percent_in_string(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(arg1: str):
            if False:
                i = 10
                return i + 15
            return '%s in template %%' % arg1
        self.checkScript(fn, ('foo',))

    def test_string_interpolation_with_percent_in_string(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(arg1: str) -> str:
            if False:
                return 10
            return '%s in template %' % arg1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Incomplete format specifier', '"%s in template %" % arg1'):
            fn('foo')

    def test_string_interpolation_with_string_placeholder_and_digit_variable(self):
        if False:
            i = 10
            return i + 15

        def fn(arg1: int) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return '%s in template' % arg1
        self.checkScript(fn, (1,))

    def test_string_interpolation_with_digit_placeholder_and_digit_variable(self):
        if False:
            return 10

        def fn(arg1: int) -> str:
            if False:
                return 10
            return '%d in template' % arg1
        self.checkScript(fn, (1,))

    def test_string_interpolation_with_alternate_digit_placeholder(self):
        if False:
            i = 10
            return i + 15

        def fn(arg1: int) -> str:
            if False:
                return 10
            return '%i in template' % arg1
        self.checkScript(fn, (1,))

    def test_string_interpolation_with_digit_placeholder_and_string_variable(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(arg1: str) -> str:
            if False:
                while True:
                    i = 10
            return '%d in template' % arg1
        with self.assertRaisesRegexWithHighlight(RuntimeError, '%d requires a number for formatting, but got String', '"%d in template" % arg1'):
            fn('1')

    def test_string_interpolation_with_exponent_placeholder_and_string_variable(self):
        if False:
            return 10

        @torch.jit.script
        def fn(arg1: str) -> str:
            if False:
                print('Hello World!')
            return '%e in template' % arg1
        with self.assertRaisesRegexWithHighlight(RuntimeError, '%e requires a number for formatting, but got String', '"%e in template" % arg1'):
            fn('1')

    def test_string_interpolation_with_lowercase_exponent_placeholder_and_digit_variable(self):
        if False:
            while True:
                i = 10

        def fn(arg1: int) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return '%e in template' % arg1
        self.checkScript(fn, (1,))

    def test_string_interpolation_with_capital_exponent_placeholder_and_digit_variable(self):
        if False:
            i = 10
            return i + 15

        def fn(arg1: int) -> str:
            if False:
                while True:
                    i = 10
            return '%E in template' % arg1
        self.checkScript(fn, (1,))

    def test_string_interpolation_with_float_placeholder_and_float_variable(self):
        if False:
            while True:
                i = 10

        def fn(arg1: float) -> str:
            if False:
                i = 10
                return i + 15
            return '%f in template' % arg1
        self.checkScript(fn, (1.0,))

    def test_string_interpolation_with_float_placeholder_and_digit_variable(self):
        if False:
            print('Hello World!')

        def fn(arg1: int) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return '%f in template' % arg1
        self.checkScript(fn, (1,))

    def test_string_interpolation_with_char_placeholder_and_char_variable(self):
        if False:
            return 10

        def fn(arg1: str) -> str:
            if False:
                print('Hello World!')
            return '%c in template' % arg1
        self.checkScript(fn, ('a',))

    def test_string_interpolation_with_char_placeholder_and_digit_variable(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(arg1: int) -> str:
            if False:
                return 10
            return '%c in template' % arg1
        self.checkScript(fn, (97,))

    def test_string_interpolation_with_char_placeholder_and_true_string_variable(self):
        if False:
            return 10

        @torch.jit.script
        def fn(arg1: str) -> str:
            if False:
                print('Hello World!')
            return '%c in template' % arg1
        with self.assertRaisesRegexWithHighlight(RuntimeError, '%c requires an int or char for formatting, but got String', '"%c in template" % arg1'):
            fn('foo')

    def test_string_interpolation_with_multiple_placeholders(self):
        if False:
            print('Hello World!')

        def fn(arg1: str, arg2: int, arg3: float) -> str:
            if False:
                i = 10
                return i + 15
            return '%s %d %f in template' % (arg1, arg2, arg3)
        self.checkScript(fn, ('foo', 1, 1))

    def test_string_interpolation_with_subscript(self):
        if False:
            i = 10
            return i + 15

        def fn(arg1: List[str]) -> str:
            if False:
                return 10
            return '%s in template' % arg1[0]
        self.checkScript(fn, (['foo', 'bar'],))

    def test_string_interpolation_with_too_few_arguments(self):
        if False:
            return 10

        @torch.jit.script
        def fn(arg1: str) -> str:
            if False:
                print('Hello World!')
            return '%s %s in template' % arg1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Too few arguments for format string', '"%s %s in template" % arg1'):
            fn('foo')

    def test_string_interpolation_with_too_many_arguments(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(arg1: str, arg2: str) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return '%s in template' % (arg1, arg2)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Too many arguments for format string', '"%s in template" % (arg1, arg2'):
            fn('foo', 'bar')

    def test_string_interpolation_with_unknown_format_specifier(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(arg1: str) -> str:
            if False:
                while True:
                    i = 10
            return '%a in template' % arg1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'The specifier %a is not supported in TorchScript format strings', '"%a in template" % arg1'):
            fn('foo')