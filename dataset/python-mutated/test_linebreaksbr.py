from django.template.defaultfilters import linebreaksbr
from django.test import SimpleTestCase
from django.utils.safestring import mark_safe
from ..utils import setup

class LinebreaksbrTests(SimpleTestCase):
    """
    The contents in "linebreaksbr" are escaped according to the current
    autoescape setting.
    """

    @setup({'linebreaksbr01': '{{ a|linebreaksbr }} {{ b|linebreaksbr }}'})
    def test_linebreaksbr01(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('linebreaksbr01', {'a': 'x&\ny', 'b': mark_safe('x&\ny')})
        self.assertEqual(output, 'x&amp;<br>y x&<br>y')

    @setup({'linebreaksbr02': '{% autoescape off %}{{ a|linebreaksbr }} {{ b|linebreaksbr }}{% endautoescape %}'})
    def test_linebreaksbr02(self):
        if False:
            print('Hello World!')
        output = self.engine.render_to_string('linebreaksbr02', {'a': 'x&\ny', 'b': mark_safe('x&\ny')})
        self.assertEqual(output, 'x&<br>y x&<br>y')

class FunctionTests(SimpleTestCase):

    def test_newline(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(linebreaksbr('line 1\nline 2'), 'line 1<br>line 2')

    def test_carriage(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(linebreaksbr('line 1\rline 2'), 'line 1<br>line 2')

    def test_carriage_newline(self):
        if False:
            print('Hello World!')
        self.assertEqual(linebreaksbr('line 1\r\nline 2'), 'line 1<br>line 2')

    def test_non_string_input(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(linebreaksbr(123), '123')

    def test_autoescape(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(linebreaksbr('foo\n<a>bar</a>\nbuz'), 'foo<br>&lt;a&gt;bar&lt;/a&gt;<br>buz')

    def test_autoescape_off(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(linebreaksbr('foo\n<a>bar</a>\nbuz', autoescape=False), 'foo<br><a>bar</a><br>buz')