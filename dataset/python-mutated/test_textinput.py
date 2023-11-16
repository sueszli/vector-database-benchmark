from django.forms import CharField, Form, TextInput
from django.utils.safestring import mark_safe
from .base import WidgetTest

class TextInputTest(WidgetTest):
    widget = TextInput()

    def test_render(self):
        if False:
            print('Hello World!')
        self.check_html(self.widget, 'email', '', html='<input type="text" name="email">')

    def test_render_none(self):
        if False:
            return 10
        self.check_html(self.widget, 'email', None, html='<input type="text" name="email">')

    def test_render_value(self):
        if False:
            return 10
        self.check_html(self.widget, 'email', 'test@example.com', html='<input type="text" name="email" value="test@example.com">')

    def test_render_boolean(self):
        if False:
            i = 10
            return i + 15
        '\n        Boolean values are rendered to their string forms ("True" and\n        "False").\n        '
        self.check_html(self.widget, 'get_spam', False, html='<input type="text" name="get_spam" value="False">')
        self.check_html(self.widget, 'get_spam', True, html='<input type="text" name="get_spam" value="True">')

    def test_render_quoted(self):
        if False:
            while True:
                i = 10
        self.check_html(self.widget, 'email', 'some "quoted" & ampersanded value', html='<input type="text" name="email" value="some &quot;quoted&quot; &amp; ampersanded value">')

    def test_render_custom_attrs(self):
        if False:
            while True:
                i = 10
        self.check_html(self.widget, 'email', 'test@example.com', attrs={'class': 'fun'}, html='<input type="text" name="email" value="test@example.com" class="fun">')

    def test_render_unicode(self):
        if False:
            i = 10
            return i + 15
        self.check_html(self.widget, 'email', 'ŠĐĆŽćžšđ', attrs={'class': 'fun'}, html='<input type="text" name="email" value="ŠĐĆŽćžšđ" class="fun">')

    def test_constructor_attrs(self):
        if False:
            print('Hello World!')
        widget = TextInput(attrs={'class': 'fun', 'type': 'email'})
        self.check_html(widget, 'email', '', html='<input type="email" class="fun" name="email">')
        self.check_html(widget, 'email', 'foo@example.com', html='<input type="email" class="fun" value="foo@example.com" name="email">')

    def test_attrs_precedence(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        `attrs` passed to render() get precedence over those passed to the\n        constructor\n        '
        widget = TextInput(attrs={'class': 'pretty'})
        self.check_html(widget, 'email', '', attrs={'class': 'special'}, html='<input type="text" class="special" name="email">')

    def test_attrs_safestring(self):
        if False:
            for i in range(10):
                print('nop')
        widget = TextInput(attrs={'onBlur': mark_safe("function('foo')")})
        self.check_html(widget, 'email', '', html='<input onBlur="function(\'foo\')" type="text" name="email">')

    def test_use_required_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(self.widget.use_required_attribute(None), True)
        self.assertIs(self.widget.use_required_attribute(''), True)
        self.assertIs(self.widget.use_required_attribute('resume.txt'), True)

    def test_fieldset(self):
        if False:
            return 10

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = CharField(widget=self.widget)
        form = TestForm()
        self.assertIs(self.widget.use_fieldset, False)
        self.assertHTMLEqual('<div><label for="id_field">Field:</label><input type="text" name="field" required id="id_field"></div>', form.render())