from django.template import TemplateSyntaxError
from django.test import SimpleTestCase
from ..utils import setup

class TemplateTagTests(SimpleTestCase):

    @setup({'templatetag01': '{% templatetag openblock %}'})
    def test_templatetag01(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.engine.render_to_string('templatetag01')
        self.assertEqual(output, '{%')

    @setup({'templatetag02': '{% templatetag closeblock %}'})
    def test_templatetag02(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('templatetag02')
        self.assertEqual(output, '%}')

    @setup({'templatetag03': '{% templatetag openvariable %}'})
    def test_templatetag03(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('templatetag03')
        self.assertEqual(output, '{{')

    @setup({'templatetag04': '{% templatetag closevariable %}'})
    def test_templatetag04(self):
        if False:
            return 10
        output = self.engine.render_to_string('templatetag04')
        self.assertEqual(output, '}}')

    @setup({'templatetag05': '{% templatetag %}'})
    def test_templatetag05(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TemplateSyntaxError):
            self.engine.get_template('templatetag05')

    @setup({'templatetag06': '{% templatetag foo %}'})
    def test_templatetag06(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TemplateSyntaxError):
            self.engine.get_template('templatetag06')

    @setup({'templatetag07': '{% templatetag openbrace %}'})
    def test_templatetag07(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('templatetag07')
        self.assertEqual(output, '{')

    @setup({'templatetag08': '{% templatetag closebrace %}'})
    def test_templatetag08(self):
        if False:
            print('Hello World!')
        output = self.engine.render_to_string('templatetag08')
        self.assertEqual(output, '}')

    @setup({'templatetag09': '{% templatetag openbrace %}{% templatetag openbrace %}'})
    def test_templatetag09(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('templatetag09')
        self.assertEqual(output, '{{')

    @setup({'templatetag10': '{% templatetag closebrace %}{% templatetag closebrace %}'})
    def test_templatetag10(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('templatetag10')
        self.assertEqual(output, '}}')

    @setup({'templatetag11': '{% templatetag opencomment %}'})
    def test_templatetag11(self):
        if False:
            return 10
        output = self.engine.render_to_string('templatetag11')
        self.assertEqual(output, '{#')

    @setup({'templatetag12': '{% templatetag closecomment %}'})
    def test_templatetag12(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('templatetag12')
        self.assertEqual(output, '#}')