from datetime import date
from django.forms import CharField, DateInput, Form
from django.utils import translation
from .base import WidgetTest

class DateInputTest(WidgetTest):
    widget = DateInput()

    def test_render_none(self):
        if False:
            while True:
                i = 10
        self.check_html(self.widget, 'date', None, html='<input type="text" name="date">')

    def test_render_value(self):
        if False:
            while True:
                i = 10
        d = date(2007, 9, 17)
        self.assertEqual(str(d), '2007-09-17')
        self.check_html(self.widget, 'date', d, html='<input type="text" name="date" value="2007-09-17">')
        self.check_html(self.widget, 'date', date(2007, 9, 17), html='<input type="text" name="date" value="2007-09-17">')

    def test_string(self):
        if False:
            print('Hello World!')
        '\n        Should be able to initialize from a string value.\n        '
        self.check_html(self.widget, 'date', '2007-09-17', html='<input type="text" name="date" value="2007-09-17">')

    def test_format(self):
        if False:
            i = 10
            return i + 15
        "\n        Use 'format' to change the way a value is displayed.\n        "
        d = date(2007, 9, 17)
        widget = DateInput(format='%d/%m/%Y', attrs={'type': 'date'})
        self.check_html(widget, 'date', d, html='<input type="date" name="date" value="17/09/2007">')

    @translation.override('de-at')
    def test_l10n(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_html(self.widget, 'date', date(2007, 9, 17), html='<input type="text" name="date" value="17.09.2007">')

    def test_fieldset(self):
        if False:
            for i in range(10):
                print('nop')

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = CharField(widget=self.widget)
        form = TestForm()
        self.assertIs(self.widget.use_fieldset, False)
        self.assertHTMLEqual(form.render(), '<div><label for="id_field">Field:</label><input id="id_field" name="field" required type="text"></div>')