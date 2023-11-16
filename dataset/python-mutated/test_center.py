from django.template.defaultfilters import center
from django.test import SimpleTestCase
from django.utils.safestring import mark_safe
from ..utils import setup

class CenterTests(SimpleTestCase):

    @setup({'center01': '{% autoescape off %}.{{ a|center:"5" }}. .{{ b|center:"5" }}.{% endautoescape %}'})
    def test_center01(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.engine.render_to_string('center01', {'a': 'a&b', 'b': mark_safe('a&b')})
        self.assertEqual(output, '. a&b . . a&b .')

    @setup({'center02': '.{{ a|center:"5" }}. .{{ b|center:"5" }}.'})
    def test_center02(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('center02', {'a': 'a&b', 'b': mark_safe('a&b')})
        self.assertEqual(output, '. a&amp;b . . a&b .')

class FunctionTests(SimpleTestCase):

    def test_center(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(center('test', 6), ' test ')

    def test_non_string_input(self):
        if False:
            while True:
                i = 10
        self.assertEqual(center(123, 5), ' 123 ')