from django.template.defaultfilters import addslashes
from django.test import SimpleTestCase
from django.utils.safestring import mark_safe
from ..utils import setup

class AddslashesTests(SimpleTestCase):

    @setup({'addslashes01': '{% autoescape off %}{{ a|addslashes }} {{ b|addslashes }}{% endautoescape %}'})
    def test_addslashes01(self):
        if False:
            print('Hello World!')
        output = self.engine.render_to_string('addslashes01', {'a': "<a>'", 'b': mark_safe("<a>'")})
        self.assertEqual(output, "<a>\\' <a>\\'")

    @setup({'addslashes02': '{{ a|addslashes }} {{ b|addslashes }}'})
    def test_addslashes02(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('addslashes02', {'a': "<a>'", 'b': mark_safe("<a>'")})
        self.assertEqual(output, "&lt;a&gt;\\&#x27; <a>\\'")

class FunctionTests(SimpleTestCase):

    def test_quotes(self):
        if False:
            return 10
        self.assertEqual(addslashes('"double quotes" and \'single quotes\''), '\\"double quotes\\" and \\\'single quotes\\\'')

    def test_backslashes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(addslashes('\\ : backslashes, too'), '\\\\ : backslashes, too')

    def test_non_string_input(self):
        if False:
            print('Hello World!')
        self.assertEqual(addslashes(123), '123')