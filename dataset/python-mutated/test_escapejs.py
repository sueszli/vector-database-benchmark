from django.template.defaultfilters import escapejs_filter
from django.test import SimpleTestCase
from django.utils.functional import lazy
from ..utils import setup

class EscapejsTests(SimpleTestCase):

    @setup({'escapejs01': '{{ a|escapejs }}'})
    def test_escapejs01(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.engine.render_to_string('escapejs01', {'a': 'testing\r\njavascript \'string" <b>escaping</b>'})
        self.assertEqual(output, 'testing\\u000D\\u000Ajavascript \\u0027string\\u0022 \\u003Cb\\u003Eescaping\\u003C/b\\u003E')

    @setup({'escapejs02': '{% autoescape off %}{{ a|escapejs }}{% endautoescape %}'})
    def test_escapejs02(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('escapejs02', {'a': 'testing\r\njavascript \'string" <b>escaping</b>'})
        self.assertEqual(output, 'testing\\u000D\\u000Ajavascript \\u0027string\\u0022 \\u003Cb\\u003Eescaping\\u003C/b\\u003E')

class FunctionTests(SimpleTestCase):

    def test_quotes(self):
        if False:
            while True:
                i = 10
        self.assertEqual(escapejs_filter('"double quotes" and \'single quotes\''), '\\u0022double quotes\\u0022 and \\u0027single quotes\\u0027')

    def test_backslashes(self):
        if False:
            while True:
                i = 10
        self.assertEqual(escapejs_filter('\\ : backslashes, too'), '\\u005C : backslashes, too')

    def test_whitespace(self):
        if False:
            print('Hello World!')
        self.assertEqual(escapejs_filter('and lots of whitespace: \r\n\t\x0b\x0c\x08'), 'and lots of whitespace: \\u000D\\u000A\\u0009\\u000B\\u000C\\u0008')

    def test_script(self):
        if False:
            print('Hello World!')
        self.assertEqual(escapejs_filter('<script>and this</script>'), '\\u003Cscript\\u003Eand this\\u003C/script\\u003E')

    def test_paragraph_separator(self):
        if False:
            print('Hello World!')
        self.assertEqual(escapejs_filter('paragraph separator:\u2029and line separator:\u2028'), 'paragraph separator:\\u2029and line separator:\\u2028')

    def test_lazy_string(self):
        if False:
            i = 10
            return i + 15
        append_script = lazy(lambda string: '<script>this</script>' + string, str)
        self.assertEqual(escapejs_filter(append_script('whitespace: \r\n\t\x0b\x0c\x08')), '\\u003Cscript\\u003Ethis\\u003C/script\\u003Ewhitespace: \\u000D\\u000A\\u0009\\u000B\\u000C\\u0008')