from django.template.defaultfilters import force_escape
from django.test import SimpleTestCase
from django.utils.safestring import SafeData
from ..utils import setup

class ForceEscapeTests(SimpleTestCase):
    """
    Force_escape is applied immediately. It can be used to provide
    double-escaping, for example.
    """

    @setup({'force-escape01': '{% autoescape off %}{{ a|force_escape }}{% endautoescape %}'})
    def test_force_escape01(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('force-escape01', {'a': 'x&y'})
        self.assertEqual(output, 'x&amp;y')

    @setup({'force-escape02': '{{ a|force_escape }}'})
    def test_force_escape02(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('force-escape02', {'a': 'x&y'})
        self.assertEqual(output, 'x&amp;y')

    @setup({'force-escape03': '{% autoescape off %}{{ a|force_escape|force_escape }}{% endautoescape %}'})
    def test_force_escape03(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('force-escape03', {'a': 'x&y'})
        self.assertEqual(output, 'x&amp;amp;y')

    @setup({'force-escape04': '{{ a|force_escape|force_escape }}'})
    def test_force_escape04(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.engine.render_to_string('force-escape04', {'a': 'x&y'})
        self.assertEqual(output, 'x&amp;amp;y')

    @setup({'force-escape05': '{% autoescape off %}{{ a|force_escape|escape }}{% endautoescape %}'})
    def test_force_escape05(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('force-escape05', {'a': 'x&y'})
        self.assertEqual(output, 'x&amp;y')

    @setup({'force-escape06': '{{ a|force_escape|escape }}'})
    def test_force_escape06(self):
        if False:
            return 10
        output = self.engine.render_to_string('force-escape06', {'a': 'x&y'})
        self.assertEqual(output, 'x&amp;y')

    @setup({'force-escape07': '{% autoescape off %}{{ a|escape|force_escape }}{% endautoescape %}'})
    def test_force_escape07(self):
        if False:
            return 10
        output = self.engine.render_to_string('force-escape07', {'a': 'x&y'})
        self.assertEqual(output, 'x&amp;amp;y')

    @setup({'force-escape08': '{{ a|escape|force_escape }}'})
    def test_force_escape08(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('force-escape08', {'a': 'x&y'})
        self.assertEqual(output, 'x&amp;amp;y')

class FunctionTests(SimpleTestCase):

    def test_escape(self):
        if False:
            i = 10
            return i + 15
        escaped = force_escape('<some html & special characters > here')
        self.assertEqual(escaped, '&lt;some html &amp; special characters &gt; here')
        self.assertIsInstance(escaped, SafeData)

    def test_unicode(self):
        if False:
            print('Hello World!')
        self.assertEqual(force_escape('<some html & special characters > here ĐÅ€£'), '&lt;some html &amp; special characters &gt; here ĐÅ€£')