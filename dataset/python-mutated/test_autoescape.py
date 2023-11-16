from django.template import TemplateSyntaxError
from django.test import SimpleTestCase
from django.utils.safestring import mark_safe
from ..utils import SafeClass, UnsafeClass, setup

class AutoescapeTagTests(SimpleTestCase):

    @setup({'autoescape-tag01': '{% autoescape off %}hello{% endautoescape %}'})
    def test_autoescape_tag01(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.engine.render_to_string('autoescape-tag01')
        self.assertEqual(output, 'hello')

    @setup({'autoescape-tag02': '{% autoescape off %}{{ first }}{% endautoescape %}'})
    def test_autoescape_tag02(self):
        if False:
            return 10
        output = self.engine.render_to_string('autoescape-tag02', {'first': '<b>hello</b>'})
        self.assertEqual(output, '<b>hello</b>')

    @setup({'autoescape-tag03': '{% autoescape on %}{{ first }}{% endautoescape %}'})
    def test_autoescape_tag03(self):
        if False:
            print('Hello World!')
        output = self.engine.render_to_string('autoescape-tag03', {'first': '<b>hello</b>'})
        self.assertEqual(output, '&lt;b&gt;hello&lt;/b&gt;')

    @setup({'autoescape-tag04': '{% autoescape off %}{{ first }} {% autoescape on %}{{ first }}{% endautoescape %}{% endautoescape %}'})
    def test_autoescape_tag04(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.engine.render_to_string('autoescape-tag04', {'first': '<a>'})
        self.assertEqual(output, '<a> &lt;a&gt;')

    @setup({'autoescape-tag05': '{% autoescape on %}{{ first }}{% endautoescape %}'})
    def test_autoescape_tag05(self):
        if False:
            return 10
        output = self.engine.render_to_string('autoescape-tag05', {'first': '<b>first</b>'})
        self.assertEqual(output, '&lt;b&gt;first&lt;/b&gt;')

    @setup({'autoescape-tag06': '{{ first }}'})
    def test_autoescape_tag06(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('autoescape-tag06', {'first': mark_safe('<b>first</b>')})
        self.assertEqual(output, '<b>first</b>')

    @setup({'autoescape-tag07': '{% autoescape on %}{{ first }}{% endautoescape %}'})
    def test_autoescape_tag07(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('autoescape-tag07', {'first': mark_safe('<b>Apple</b>')})
        self.assertEqual(output, '<b>Apple</b>')

    @setup({'autoescape-tag08': '{% autoescape on %}{{ var|default_if_none:" endquote\\" hah" }}{% endautoescape %}'})
    def test_autoescape_tag08(self):
        if False:
            print('Hello World!')
        '\n        Literal string arguments to filters, if used in the result, are safe.\n        '
        output = self.engine.render_to_string('autoescape-tag08', {'var': None})
        self.assertEqual(output, ' endquote" hah')

    @setup({'autoescape-tag09': '{{ unsafe }}'})
    def test_autoescape_tag09(self):
        if False:
            return 10
        output = self.engine.render_to_string('autoescape-tag09', {'unsafe': UnsafeClass()})
        self.assertEqual(output, 'you &amp; me')

    @setup({'autoescape-tag10': '{{ safe }}'})
    def test_autoescape_tag10(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.engine.render_to_string('autoescape-tag10', {'safe': SafeClass()})
        self.assertEqual(output, 'you &gt; me')

    @setup({'autoescape-filtertag01': '{{ first }}{% filter safe %}{{ first }} x<y{% endfilter %}'})
    def test_autoescape_filtertag01(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The "safe" and "escape" filters cannot work due to internal\n        implementation details (fortunately, the (no)autoescape block\n        tags can be used in those cases)\n        '
        with self.assertRaises(TemplateSyntaxError):
            self.engine.render_to_string('autoescape-filtertag01', {'first': '<a>'})

    @setup({'autoescape-filters01': '{{ var|cut:"&" }}'})
    def test_autoescape_filters01(self):
        if False:
            return 10
        output = self.engine.render_to_string('autoescape-filters01', {'var': 'this & that'})
        self.assertEqual(output, 'this  that')

    @setup({'autoescape-filters02': '{{ var|join:" & " }}'})
    def test_autoescape_filters02(self):
        if False:
            return 10
        output = self.engine.render_to_string('autoescape-filters02', {'var': ('Tom', 'Dick', 'Harry')})
        self.assertEqual(output, 'Tom & Dick & Harry')

    @setup({'autoescape-literals01': '{{ "this & that" }}'})
    def test_autoescape_literals01(self):
        if False:
            i = 10
            return i + 15
        '\n        Literal strings are safe.\n        '
        output = self.engine.render_to_string('autoescape-literals01')
        self.assertEqual(output, 'this & that')

    @setup({'autoescape-stringiterations01': '{% for l in var %}{{ l }},{% endfor %}'})
    def test_autoescape_stringiterations01(self):
        if False:
            print('Hello World!')
        '\n        Iterating over strings outputs safe characters.\n        '
        output = self.engine.render_to_string('autoescape-stringiterations01', {'var': 'K&R'})
        self.assertEqual(output, 'K,&amp;,R,')

    @setup({'autoescape-lookup01': '{{ var.key }}'})
    def test_autoescape_lookup01(self):
        if False:
            return 10
        '\n        Escape requirement survives lookup.\n        '
        output = self.engine.render_to_string('autoescape-lookup01', {'var': {'key': 'this & that'}})
        self.assertEqual(output, 'this &amp; that')

    @setup({'autoescape-incorrect-arg': '{% autoescape true %}{{ var.key }}{% endautoescape %}'})
    def test_invalid_arg(self):
        if False:
            print('Hello World!')
        msg = "'autoescape' argument should be 'on' or 'off'"
        with self.assertRaisesMessage(TemplateSyntaxError, msg):
            self.engine.render_to_string('autoescape-incorrect-arg', {'var': {'key': 'this & that'}})

    @setup({'autoescape-incorrect-arg': '{% autoescape %}{{ var.key }}{% endautoescape %}'})
    def test_no_arg(self):
        if False:
            while True:
                i = 10
        msg = "'autoescape' tag requires exactly one argument."
        with self.assertRaisesMessage(TemplateSyntaxError, msg):
            self.engine.render_to_string('autoescape-incorrect-arg', {'var': {'key': 'this & that'}})