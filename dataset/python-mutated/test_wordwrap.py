from django.template.defaultfilters import wordwrap
from django.test import SimpleTestCase
from django.utils.functional import lazystr
from django.utils.safestring import mark_safe
from ..utils import setup

class WordwrapTests(SimpleTestCase):

    @setup({'wordwrap01': '{% autoescape off %}{{ a|wordwrap:"3" }} {{ b|wordwrap:"3" }}{% endautoescape %}'})
    def test_wordwrap01(self):
        if False:
            i = 10
            return i + 15
        output = self.engine.render_to_string('wordwrap01', {'a': 'a & b', 'b': mark_safe('a & b')})
        self.assertEqual(output, 'a &\nb a &\nb')

    @setup({'wordwrap02': '{{ a|wordwrap:"3" }} {{ b|wordwrap:"3" }}'})
    def test_wordwrap02(self):
        if False:
            return 10
        output = self.engine.render_to_string('wordwrap02', {'a': 'a & b', 'b': mark_safe('a & b')})
        self.assertEqual(output, 'a &amp;\nb a &\nb')

class FunctionTests(SimpleTestCase):

    def test_wrap(self):
        if False:
            return 10
        self.assertEqual(wordwrap("this is a long paragraph of text that really needs to be wrapped I'm afraid", 14), "this is a long\nparagraph of\ntext that\nreally needs\nto be wrapped\nI'm afraid")

    def test_indent(self):
        if False:
            return 10
        self.assertEqual(wordwrap('this is a short paragraph of text.\n  But this line should be indented', 14), 'this is a\nshort\nparagraph of\ntext.\n  But this\nline should be\nindented')

    def test_indent2(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(wordwrap('this is a short paragraph of text.\n  But this line should be indented', 15), 'this is a short\nparagraph of\ntext.\n  But this line\nshould be\nindented')

    def test_non_string_input(self):
        if False:
            while True:
                i = 10
        self.assertEqual(wordwrap(123, 2), '123')

    def test_wrap_lazy_string(self):
        if False:
            while True:
                i = 10
        self.assertEqual(wordwrap(lazystr("this is a long paragraph of text that really needs to be wrapped I'm afraid"), 14), "this is a long\nparagraph of\ntext that\nreally needs\nto be wrapped\nI'm afraid")