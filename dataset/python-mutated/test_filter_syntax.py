from django.template import TemplateSyntaxError
from django.test import SimpleTestCase
from ..utils import SomeClass, SomeOtherException, UTF8Class, setup

class FilterSyntaxTests(SimpleTestCase):

    @setup({'filter-syntax01': '{{ var|upper }}'})
    def test_filter_syntax01(self):
        if False:
            while True:
                i = 10
        '\n        Basic filter usage\n        '
        output = self.engine.render_to_string('filter-syntax01', {'var': 'Django is the greatest!'})
        self.assertEqual(output, 'DJANGO IS THE GREATEST!')

    @setup({'filter-syntax02': '{{ var|upper|lower }}'})
    def test_filter_syntax02(self):
        if False:
            return 10
        '\n        Chained filters\n        '
        output = self.engine.render_to_string('filter-syntax02', {'var': 'Django is the greatest!'})
        self.assertEqual(output, 'django is the greatest!')

    @setup({'filter-syntax03': '{{ var |upper }}'})
    def test_filter_syntax03(self):
        if False:
            print('Hello World!')
        '\n        Allow spaces before the filter pipe\n        '
        output = self.engine.render_to_string('filter-syntax03', {'var': 'Django is the greatest!'})
        self.assertEqual(output, 'DJANGO IS THE GREATEST!')

    @setup({'filter-syntax04': '{{ var| upper }}'})
    def test_filter_syntax04(self):
        if False:
            return 10
        '\n        Allow spaces after the filter pipe\n        '
        output = self.engine.render_to_string('filter-syntax04', {'var': 'Django is the greatest!'})
        self.assertEqual(output, 'DJANGO IS THE GREATEST!')

    @setup({'filter-syntax05': '{{ var|does_not_exist }}'})
    def test_filter_syntax05(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Raise TemplateSyntaxError for a nonexistent filter\n        '
        msg = "Invalid filter: 'does_not_exist'"
        with self.assertRaisesMessage(TemplateSyntaxError, msg):
            self.engine.get_template('filter-syntax05')

    @setup({'filter-syntax06': '{{ var|fil(ter) }}'})
    def test_filter_syntax06(self):
        if False:
            print('Hello World!')
        '\n        Raise TemplateSyntaxError when trying to access a filter containing\n        an illegal character\n        '
        with self.assertRaisesMessage(TemplateSyntaxError, "Invalid filter: 'fil'"):
            self.engine.get_template('filter-syntax06')

    @setup({'filter-syntax07': '{% nothing_to_see_here %}'})
    def test_filter_syntax07(self):
        if False:
            print('Hello World!')
        '\n        Raise TemplateSyntaxError for invalid block tags\n        '
        msg = "Invalid block tag on line 1: 'nothing_to_see_here'. Did you forget to register or load this tag?"
        with self.assertRaisesMessage(TemplateSyntaxError, msg):
            self.engine.get_template('filter-syntax07')

    @setup({'filter-syntax08': '{% %}'})
    def test_filter_syntax08(self):
        if False:
            return 10
        '\n        Raise TemplateSyntaxError for empty block tags\n        '
        with self.assertRaisesMessage(TemplateSyntaxError, 'Empty block tag on line 1'):
            self.engine.get_template('filter-syntax08')

    @setup({'filter-syntax08-multi-line': 'line 1\nline 2\nline 3{% %}\nline 4\nline 5'})
    def test_filter_syntax08_multi_line(self):
        if False:
            i = 10
            return i + 15
        '\n        Raise TemplateSyntaxError for empty block tags in templates with\n        multiple lines.\n        '
        with self.assertRaisesMessage(TemplateSyntaxError, 'Empty block tag on line 3'):
            self.engine.get_template('filter-syntax08-multi-line')

    @setup({'filter-syntax09': '{{ var|cut:"o"|upper|lower }}'})
    def test_filter_syntax09(self):
        if False:
            i = 10
            return i + 15
        '\n        Chained filters, with an argument to the first one\n        '
        output = self.engine.render_to_string('filter-syntax09', {'var': 'Foo'})
        self.assertEqual(output, 'f')

    @setup({'filter-syntax10': '{{ var|default_if_none:" endquote\\" hah" }}'})
    def test_filter_syntax10(self):
        if False:
            i = 10
            return i + 15
        '\n        Literal string as argument is always "safe" from auto-escaping.\n        '
        output = self.engine.render_to_string('filter-syntax10', {'var': None})
        self.assertEqual(output, ' endquote" hah')

    @setup({'filter-syntax11': '{{ var|default_if_none:var2 }}'})
    def test_filter_syntax11(self):
        if False:
            i = 10
            return i + 15
        '\n        Variable as argument\n        '
        output = self.engine.render_to_string('filter-syntax11', {'var': None, 'var2': 'happy'})
        self.assertEqual(output, 'happy')

    @setup({'filter-syntax13': '1{{ var.method3 }}2'})
    def test_filter_syntax13(self):
        if False:
            return 10
        '\n        Fail silently for methods that raise an exception with a\n        `silent_variable_failure` attribute\n        '
        output = self.engine.render_to_string('filter-syntax13', {'var': SomeClass()})
        if self.engine.string_if_invalid:
            self.assertEqual(output, '1INVALID2')
        else:
            self.assertEqual(output, '12')

    @setup({'filter-syntax14': '1{{ var.method4 }}2'})
    def test_filter_syntax14(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        In methods that raise an exception without a\n        `silent_variable_attribute` set to True, the exception propagates\n        '
        with self.assertRaises(SomeOtherException):
            self.engine.render_to_string('filter-syntax14', {'var': SomeClass()})

    @setup({'filter-syntax15': '{{ var|default_if_none:"foo\\bar" }}'})
    def test_filter_syntax15(self):
        if False:
            return 10
        '\n        Escaped backslash in argument\n        '
        output = self.engine.render_to_string('filter-syntax15', {'var': None})
        self.assertEqual(output, 'foo\\bar')

    @setup({'filter-syntax16': '{{ var|default_if_none:"foo\\now" }}'})
    def test_filter_syntax16(self):
        if False:
            print('Hello World!')
        '\n        Escaped backslash using known escape char\n        '
        output = self.engine.render_to_string('filter-syntax16', {'var': None})
        self.assertEqual(output, 'foo\\now')

    @setup({'filter-syntax17': '{{ var|join:"" }}'})
    def test_filter_syntax17(self):
        if False:
            print('Hello World!')
        '\n        Empty strings can be passed as arguments to filters\n        '
        output = self.engine.render_to_string('filter-syntax17', {'var': ['a', 'b', 'c']})
        self.assertEqual(output, 'abc')

    @setup({'filter-syntax18': '{{ var }}'})
    def test_filter_syntax18(self):
        if False:
            while True:
                i = 10
        '\n        Strings are converted to bytestrings in the final output.\n        '
        output = self.engine.render_to_string('filter-syntax18', {'var': UTF8Class()})
        self.assertEqual(output, 'ŠĐĆŽćžšđ')

    @setup({'filter-syntax19': '{{ var|truncatewords:1 }}'})
    def test_filter_syntax19(self):
        if False:
            print('Hello World!')
        '\n        Numbers as filter arguments should work\n        '
        output = self.engine.render_to_string('filter-syntax19', {'var': 'hello world'})
        self.assertEqual(output, 'hello …')

    @setup({'filter-syntax20': '{{ ""|default_if_none:"was none" }}'})
    def test_filter_syntax20(self):
        if False:
            while True:
                i = 10
        '\n        Filters should accept empty string constants\n        '
        output = self.engine.render_to_string('filter-syntax20')
        self.assertEqual(output, '')

    @setup({'filter-syntax21': '1{{ var.silent_fail_key }}2'})
    def test_filter_syntax21(self):
        if False:
            print('Hello World!')
        '\n        Fail silently for non-callable attribute and dict lookups which\n        raise an exception with a "silent_variable_failure" attribute\n        '
        output = self.engine.render_to_string('filter-syntax21', {'var': SomeClass()})
        if self.engine.string_if_invalid:
            self.assertEqual(output, '1INVALID2')
        else:
            self.assertEqual(output, '12')

    @setup({'filter-syntax22': '1{{ var.silent_fail_attribute }}2'})
    def test_filter_syntax22(self):
        if False:
            return 10
        '\n        Fail silently for non-callable attribute and dict lookups which\n        raise an exception with a `silent_variable_failure` attribute\n        '
        output = self.engine.render_to_string('filter-syntax22', {'var': SomeClass()})
        if self.engine.string_if_invalid:
            self.assertEqual(output, '1INVALID2')
        else:
            self.assertEqual(output, '12')

    @setup({'filter-syntax23': '1{{ var.noisy_fail_key }}2'})
    def test_filter_syntax23(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        In attribute and dict lookups that raise an unexpected exception\n        without a `silent_variable_attribute` set to True, the exception\n        propagates\n        '
        with self.assertRaises(SomeOtherException):
            self.engine.render_to_string('filter-syntax23', {'var': SomeClass()})

    @setup({'filter-syntax24': '1{{ var.noisy_fail_attribute }}2'})
    def test_filter_syntax24(self):
        if False:
            i = 10
            return i + 15
        '\n        In attribute and dict lookups that raise an unexpected exception\n        without a `silent_variable_attribute` set to True, the exception\n        propagates\n        '
        with self.assertRaises(SomeOtherException):
            self.engine.render_to_string('filter-syntax24', {'var': SomeClass()})

    @setup({'filter-syntax25': '{{ var.attribute_error_attribute }}'})
    def test_filter_syntax25(self):
        if False:
            return 10
        '\n        #16383 - Attribute errors from an @property value should be\n        reraised.\n        '
        with self.assertRaises(AttributeError):
            self.engine.render_to_string('filter-syntax25', {'var': SomeClass()})

    @setup({'template': '{{ var.type_error_attribute }}'})
    def test_type_error_attribute(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            self.engine.render_to_string('template', {'var': SomeClass()})