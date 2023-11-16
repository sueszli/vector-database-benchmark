from django.template.defaultfilters import truncatewords
from django.test import SimpleTestCase
from django.utils.safestring import mark_safe
from ..utils import setup

class TruncatewordsTests(SimpleTestCase):

    @setup({'truncatewords01': '{% autoescape off %}{{ a|truncatewords:"2" }} {{ b|truncatewords:"2"}}{% endautoescape %}'})
    def test_truncatewords01(self):
        if False:
            print('Hello World!')
        output = self.engine.render_to_string('truncatewords01', {'a': 'alpha & bravo', 'b': mark_safe('alpha &amp; bravo')})
        self.assertEqual(output, 'alpha & … alpha &amp; …')

    @setup({'truncatewords02': '{{ a|truncatewords:"2" }} {{ b|truncatewords:"2"}}'})
    def test_truncatewords02(self):
        if False:
            return 10
        output = self.engine.render_to_string('truncatewords02', {'a': 'alpha & bravo', 'b': mark_safe('alpha &amp; bravo')})
        self.assertEqual(output, 'alpha &amp; … alpha &amp; …')

class FunctionTests(SimpleTestCase):

    def test_truncate(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(truncatewords('A sentence with a few words in it', 1), 'A …')

    def test_truncate2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(truncatewords('A sentence with a few words in it', 5), 'A sentence with a few …')

    def test_overtruncate(self):
        if False:
            while True:
                i = 10
        self.assertEqual(truncatewords('A sentence with a few words in it', 100), 'A sentence with a few words in it')

    def test_invalid_number(self):
        if False:
            while True:
                i = 10
        self.assertEqual(truncatewords('A sentence with a few words in it', 'not a number'), 'A sentence with a few words in it')

    def test_non_string_input(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(truncatewords(123, 2), '123')