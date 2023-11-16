from django.forms import BooleanField, CheckboxInput, Form
from .base import WidgetTest

class CheckboxInputTest(WidgetTest):
    widget = CheckboxInput()

    def test_render_empty(self):
        if False:
            return 10
        self.check_html(self.widget, 'is_cool', '', html='<input type="checkbox" name="is_cool">')

    def test_render_none(self):
        if False:
            return 10
        self.check_html(self.widget, 'is_cool', None, html='<input type="checkbox" name="is_cool">')

    def test_render_false(self):
        if False:
            while True:
                i = 10
        self.check_html(self.widget, 'is_cool', False, html='<input type="checkbox" name="is_cool">')

    def test_render_true(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_html(self.widget, 'is_cool', True, html='<input checked type="checkbox" name="is_cool">')

    def test_render_value(self):
        if False:
            print('Hello World!')
        "\n        Using any value that's not in ('', None, False, True) will check the\n        checkbox and set the 'value' attribute.\n        "
        self.check_html(self.widget, 'is_cool', 'foo', html='<input checked type="checkbox" name="is_cool" value="foo">')

    def test_render_int(self):
        if False:
            return 10
        '\n        Integers are handled by value, not as booleans (#17114).\n        '
        self.check_html(self.widget, 'is_cool', 0, html='<input checked type="checkbox" name="is_cool" value="0">')
        self.check_html(self.widget, 'is_cool', 1, html='<input checked type="checkbox" name="is_cool" value="1">')

    def test_render_check_test(self):
        if False:
            i = 10
            return i + 15
        "\n        You can pass 'check_test' to the constructor. This is a callable that\n        takes the value and returns True if the box should be checked.\n        "
        widget = CheckboxInput(check_test=lambda value: value.startswith('hello'))
        self.check_html(widget, 'greeting', '', html='<input type="checkbox" name="greeting">')
        self.check_html(widget, 'greeting', 'hello', html='<input checked type="checkbox" name="greeting" value="hello">')
        self.check_html(widget, 'greeting', 'hello there', html='<input checked type="checkbox" name="greeting" value="hello there">')
        self.check_html(widget, 'greeting', 'hello & goodbye', html='<input checked type="checkbox" name="greeting" value="hello &amp; goodbye">')

    def test_render_check_exception(self):
        if False:
            while True:
                i = 10
        "\n        Calling check_test() shouldn't swallow exceptions (#17888).\n        "
        widget = CheckboxInput(check_test=lambda value: value.startswith('hello'))
        with self.assertRaises(AttributeError):
            widget.render('greeting', True)

    def test_value_from_datadict(self):
        if False:
            i = 10
            return i + 15
        "\n        The CheckboxInput widget will return False if the key is not found in\n        the data dictionary (because HTML form submission doesn't send any\n        result for unchecked checkboxes).\n        "
        self.assertFalse(self.widget.value_from_datadict({}, {}, 'testing'))

    def test_value_from_datadict_string_int(self):
        if False:
            return 10
        value = self.widget.value_from_datadict({'testing': '0'}, {}, 'testing')
        self.assertIs(value, True)

    def test_value_omitted_from_data(self):
        if False:
            print('Hello World!')
        self.assertIs(self.widget.value_omitted_from_data({'field': 'value'}, {}, 'field'), False)
        self.assertIs(self.widget.value_omitted_from_data({}, {}, 'field'), False)

    def test_get_context_does_not_mutate_attrs(self):
        if False:
            return 10
        attrs = {'checked': False}
        self.widget.get_context('name', True, attrs)
        self.assertIs(attrs['checked'], False)

    def test_fieldset(self):
        if False:
            return 10

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = BooleanField(widget=self.widget)
        form = TestForm()
        self.assertIs(self.widget.use_fieldset, False)
        self.assertHTMLEqual(form.render(), '<div><label for="id_field">Field:</label><input id="id_field" name="field" required type="checkbox"></div>')