import unittest
import dash.dash_table.Format as f
from dash.dash_table.Format import Format
import dash.dash_table.FormatTemplate as FormatTemplate

class FormatTest(unittest.TestCase):

    def validate_complex(self, res):
        if False:
            i = 10
            return i + 15
        self.assertEqual(res['locale']['symbol'][0], 'a')
        self.assertEqual(res['locale']['symbol'][1], 'bc')
        self.assertEqual(res['locale']['decimal'], 'x')
        self.assertEqual(res['locale']['group'], 'y')
        self.assertEqual(res['nully'], 'N/A')
        self.assertEqual(res['prefix'], None)
        self.assertEqual(res['specifier'], '.^($010,.6s')

    def test_complex_and_valid_in_ctor(self):
        if False:
            for i in range(10):
                print('nop')
        res = Format(align=f.Align.center, fill='.', group=f.Group.yes, padding=True, padding_width=10, precision=6, scheme='s', sign=f.Sign.parantheses, symbol=f.Symbol.yes, symbol_prefix='a', symbol_suffix='bc', decimal_delimiter='x', group_delimiter='y', groups=[2, 2, 2, 3], nully='N/A', si_prefix=f.Prefix.none)
        self.validate_complex(res.to_plotly_json())

    def test_complex_and_valid_in_fluent(self):
        if False:
            i = 10
            return i + 15
        res = Format().align(f.Align.center).fill('.').group(f.Group.yes).padding(True).padding_width(10).precision(6).scheme('s').sign(f.Sign.parantheses).symbol(f.Symbol.yes).symbol_prefix('a').symbol_suffix('bc').decimal_delimiter('x').group_delimiter('y').groups([2, 2, 2, 3]).nully('N/A').si_prefix(f.Prefix.none)
        self.validate_complex(res.to_plotly_json())

    def test_money_template(self):
        if False:
            print('Hello World!')
        res = FormatTemplate.money(2).to_plotly_json()
        self.assertEqual(res['specifier'], '$,.2f')

    def test_percentage_template(self):
        if False:
            while True:
                i = 10
        res = FormatTemplate.percentage(1).to_plotly_json()
        self.assertEqual(res['specifier'], '.1%')

    def test_valid_align_named(self):
        if False:
            print('Hello World!')
        Format().align(f.Align.center)

    def test_valid_align_string(self):
        if False:
            for i in range(10):
                print('nop')
        Format().align('=')

    def test_invalid_align_string(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, Format().align, 'i')

    def test_invalid_align_type(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, Format().align, 7)

    def test_valid_fill(self):
        if False:
            i = 10
            return i + 15
        Format().fill('.')

    def test_invalid_fill_length(self):
        if False:
            return 10
        self.assertRaises(ValueError, Format().fill, 'invalid')

    def test_invalid_fill_type(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, Format().fill, 7)

    def test_valid_group_bool(self):
        if False:
            for i in range(10):
                print('nop')
        Format().group(True)

    def test_valid_group_string(self):
        if False:
            for i in range(10):
                print('nop')
        Format().group(',')

    def test_valid_group_named(self):
        if False:
            return 10
        Format().group(f.Group.no)

    def test_invalid_group_type(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, Format().group, 7)

    def test_invalid_group_string(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, Format().group, 'invalid')

    def test_valid_padding_bool(self):
        if False:
            for i in range(10):
                print('nop')
        Format().padding(False)

    def test_valid_padding_string(self):
        if False:
            while True:
                i = 10
        Format().padding('0')

    def test_valid_padding_named(self):
        if False:
            i = 10
            return i + 15
        Format().padding(f.Padding.no)

    def test_invalid_padding_type(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, Format().padding, 7)

    def test_invalid_padding_string(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, Format().padding, 'invalid')

    def test_valid_padding_width(self):
        if False:
            return 10
        Format().padding_width(10)

    def test_valid_padding_width_0(self):
        if False:
            return 10
        Format().padding_width(0)

    def test_invalid_padding_width_negative(self):
        if False:
            return 10
        self.assertRaises(ValueError, Format().padding_width, -10)

    def test_invalid_padding_width_type(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, Format().padding_width, 7.7)

    def test_valid_precision(self):
        if False:
            for i in range(10):
                print('nop')
        Format().precision(10)

    def test_valid_precision_0(self):
        if False:
            while True:
                i = 10
        Format().precision(0)

    def test_invalid_precision_negative(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, Format().precision, -10)

    def test_invalid_precision_type(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, Format().precision, 7.7)

    def test_valid_prefix_number(self):
        if False:
            print('Hello World!')
        Format().si_prefix(10 ** (-24))

    def test_valid_prefix_named(self):
        if False:
            for i in range(10):
                print('nop')
        Format().si_prefix(f.Prefix.micro)

    def test_invalid_prefix_number(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, Format().si_prefix, 10 ** (-23))

    def test_invalid_prefix_type(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, Format().si_prefix, '10**-23')

    def test_valid_scheme_string(self):
        if False:
            print('Hello World!')
        Format().scheme('s')

    def test_valid_scheme_named(self):
        if False:
            print('Hello World!')
        Format().scheme(f.Scheme.decimal)

    def test_invalid_scheme_string(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, Format().scheme, 'invalid')

    def test_invalid_scheme_type(self):
        if False:
            return 10
        self.assertRaises(TypeError, Format().scheme, 7)

    def test_valid_sign_string(self):
        if False:
            return 10
        Format().sign('+')

    def test_valid_sign_named(self):
        if False:
            i = 10
            return i + 15
        Format().sign(f.Sign.space)

    def test_invalid_sign_string(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, Format().sign, 'invalid')

    def test_invalid_sign_type(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, Format().sign, 7)

    def test_valid_symbol_string(self):
        if False:
            for i in range(10):
                print('nop')
        Format().symbol('$')

    def test_valid_symbol_named(self):
        if False:
            i = 10
            return i + 15
        Format().symbol(f.Symbol.hex)

    def test_invalid_symbol_string(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, Format().symbol, 'invalid')

    def test_invalid_symbol_type(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, Format().symbol, 7)

    def test_valid_symbol_prefix(self):
        if False:
            print('Hello World!')
        Format().symbol_prefix('abc+-')

    def test_invalid_symbol_prefix_type(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, Format().symbol_prefix, 7)

    def test_valid_symbol_suffix(self):
        if False:
            return 10
        Format().symbol_suffix('abc+-')

    def test_invalid_symbol_suffix(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, Format().symbol_suffix, 7)

    def test_valid_trim_boolean(self):
        if False:
            i = 10
            return i + 15
        Format().trim(False)

    def test_valid_trim_string(self):
        if False:
            while True:
                i = 10
        Format().trim('~')

    def test_valid_trim_named(self):
        if False:
            while True:
                i = 10
        Format().trim(f.Trim.yes)

    def test_invalid_trim_string(self):
        if False:
            return 10
        self.assertRaises(TypeError, Format().trim, 'invalid')

    def test_invalid_trim_type(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, Format().trim, 7)

    def test_valid_decimal_delimiter(self):
        if False:
            print('Hello World!')
        Format().decimal_delimiter('x')

    def test_valid_decimal_delimiter_multi_char(self):
        if False:
            return 10
        self.assertRaises(ValueError, Format().decimal_delimiter, 'xyz')

    def test_invalid_decimal_delimiter(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, Format().decimal_delimiter, 7)

    def test_valid_group_delimitator(self):
        if False:
            for i in range(10):
                print('nop')
        Format().group_delimiter('y')

    def test_valid_group_delimitator_multi_char(self):
        if False:
            return 10
        self.assertRaises(ValueError, Format().group_delimiter, 'xyz')

    def test_invalid_group_delimiter(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, Format().group_delimiter, 7)

    def test_valid_groups(self):
        if False:
            print('Hello World!')
        Format().groups([3])

    def test_valid_groups_single(self):
        if False:
            i = 10
            return i + 15
        Format().groups(3)

    def test_valid_groups_multi(self):
        if False:
            while True:
                i = 10
        Format().groups([2, 2, 3])

    def test_invalid_groups_single_0(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ValueError, Format().groups, 0)

    def test_invalid_groups_single_negative(self):
        if False:
            while True:
                i = 10
        self.assertRaises(ValueError, Format().groups, -7)

    def test_invalid_groups_single_type(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, Format().groups, 7.7)

    def test_invalid_groups_empty(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, Format().groups, [])

    def test_invalid_groups_nested_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, Format().groups, [7.7, 7])

    def test_invalid_groups_nested_0(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, Format().groups, [3, 3, 0])

    def test_invalid_groups_nested_negative(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, Format().groups, [3, 3, -7])