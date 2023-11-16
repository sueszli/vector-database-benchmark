from django.template import TemplateSyntaxError
from django.test import SimpleTestCase
from ..utils import setup

class SimpleTagTests(SimpleTestCase):
    libraries = {'custom': 'template_tests.templatetags.custom'}

    @setup({'simpletag-renamed01': '{% load custom %}{% minusone 7 %}'})
    def test_simpletag_renamed01(self):
        if False:
            print('Hello World!')
        output = self.engine.render_to_string('simpletag-renamed01')
        self.assertEqual(output, '6')

    @setup({'simpletag-renamed02': '{% load custom %}{% minustwo 7 %}'})
    def test_simpletag_renamed02(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('simpletag-renamed02')
        self.assertEqual(output, '5')

    @setup({'simpletag-renamed03': '{% load custom %}{% minustwo_overridden_name 7 %}'})
    def test_simpletag_renamed03(self):
        if False:
            return 10
        with self.assertRaises(TemplateSyntaxError):
            self.engine.get_template('simpletag-renamed03')