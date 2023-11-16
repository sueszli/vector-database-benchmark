from django.template import Context, Template
from django.test import SimpleTestCase
from django.utils import html, translation
from django.utils.functional import Promise, lazy, lazystr
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.translation import gettext_lazy

class customescape(str):

    def __html__(self):
        if False:
            i = 10
            return i + 15
        return self.replace('<', '<<').replace('>', '>>')

class SafeStringTest(SimpleTestCase):

    def assertRenderEqual(self, tpl, expected, **context):
        if False:
            for i in range(10):
                print('nop')
        context = Context(context)
        tpl = Template(tpl)
        self.assertEqual(tpl.render(context), expected)

    def test_mark_safe(self):
        if False:
            for i in range(10):
                print('nop')
        s = mark_safe('a&b')
        self.assertRenderEqual('{{ s }}', 'a&b', s=s)
        self.assertRenderEqual('{{ s|force_escape }}', 'a&amp;b', s=s)

    def test_mark_safe_str(self):
        if False:
            print('Hello World!')
        "\n        Calling str() on a SafeString instance doesn't lose the safe status.\n        "
        s = mark_safe('a&b')
        self.assertIsInstance(str(s), type(s))

    def test_mark_safe_object_implementing_dunder_html(self):
        if False:
            while True:
                i = 10
        e = customescape('<a&b>')
        s = mark_safe(e)
        self.assertIs(s, e)
        self.assertRenderEqual('{{ s }}', '<<a&b>>', s=s)
        self.assertRenderEqual('{{ s|force_escape }}', '&lt;a&amp;b&gt;', s=s)

    def test_mark_safe_lazy(self):
        if False:
            print('Hello World!')
        safe_s = mark_safe(lazystr('a&b'))
        self.assertIsInstance(safe_s, Promise)
        self.assertRenderEqual('{{ s }}', 'a&b', s=safe_s)
        self.assertIsInstance(str(safe_s), SafeData)

    def test_mark_safe_lazy_i18n(self):
        if False:
            while True:
                i = 10
        s = mark_safe(gettext_lazy('name'))
        tpl = Template('{{ s }}')
        with translation.override('fr'):
            self.assertEqual(tpl.render(Context({'s': s})), 'nom')

    def test_mark_safe_object_implementing_dunder_str(self):
        if False:
            while True:
                i = 10

        class Obj:

            def __str__(self):
                if False:
                    i = 10
                    return i + 15
                return '<obj>'
        s = mark_safe(Obj())
        self.assertRenderEqual('{{ s }}', '<obj>', s=s)

    def test_mark_safe_result_implements_dunder_html(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(mark_safe('a&b').__html__(), 'a&b')

    def test_mark_safe_lazy_result_implements_dunder_html(self):
        if False:
            while True:
                i = 10
        self.assertEqual(mark_safe(lazystr('a&b')).__html__(), 'a&b')

    def test_add_lazy_safe_text_and_safe_text(self):
        if False:
            return 10
        s = html.escape(lazystr('a'))
        s += mark_safe('&b')
        self.assertRenderEqual('{{ s }}', 'a&b', s=s)
        s = html.escapejs(lazystr('a'))
        s += mark_safe('&b')
        self.assertRenderEqual('{{ s }}', 'a&b', s=s)

    def test_mark_safe_as_decorator(self):
        if False:
            print('Hello World!')
        '\n        mark_safe used as a decorator leaves the result of a function\n        unchanged.\n        '

        def clean_string_provider():
            if False:
                print('Hello World!')
            return '<html><body>dummy</body></html>'
        self.assertEqual(mark_safe(clean_string_provider)(), clean_string_provider())

    def test_mark_safe_decorator_does_not_affect_dunder_html(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        mark_safe doesn't affect a callable that has an __html__() method.\n        "

        class SafeStringContainer:

            def __html__(self):
                if False:
                    while True:
                        i = 10
                return '<html></html>'
        self.assertIs(mark_safe(SafeStringContainer), SafeStringContainer)

    def test_mark_safe_decorator_does_not_affect_promises(self):
        if False:
            print('Hello World!')
        "\n        mark_safe doesn't affect lazy strings (Promise objects).\n        "

        def html_str():
            if False:
                return 10
            return '<html></html>'
        lazy_str = lazy(html_str, str)()
        self.assertEqual(mark_safe(lazy_str), html_str())

    def test_default_additional_attrs(self):
        if False:
            return 10
        s = SafeString('a&b')
        msg = "object has no attribute 'dynamic_attr'"
        with self.assertRaisesMessage(AttributeError, msg):
            s.dynamic_attr = True

    def test_default_safe_data_additional_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        s = SafeData()
        msg = "object has no attribute 'dynamic_attr'"
        with self.assertRaisesMessage(AttributeError, msg):
            s.dynamic_attr = True