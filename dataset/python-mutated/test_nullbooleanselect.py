from django.forms import Form, NullBooleanField, NullBooleanSelect
from django.utils import translation
from .base import WidgetTest

class NullBooleanSelectTest(WidgetTest):
    widget = NullBooleanSelect()

    def test_render_true(self):
        if False:
            return 10
        self.check_html(self.widget, 'is_cool', True, html='<select name="is_cool">\n            <option value="unknown">Unknown</option>\n            <option value="true" selected>Yes</option>\n            <option value="false">No</option>\n            </select>')

    def test_render_false(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_html(self.widget, 'is_cool', False, html='<select name="is_cool">\n            <option value="unknown">Unknown</option>\n            <option value="true">Yes</option>\n            <option value="false" selected>No</option>\n            </select>')

    def test_render_none(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_html(self.widget, 'is_cool', None, html='<select name="is_cool">\n            <option value="unknown" selected>Unknown</option>\n            <option value="true">Yes</option>\n            <option value="false">No</option>\n            </select>')

    def test_render_value_unknown(self):
        if False:
            while True:
                i = 10
        self.check_html(self.widget, 'is_cool', 'unknown', html='<select name="is_cool">\n            <option value="unknown" selected>Unknown</option>\n            <option value="true">Yes</option>\n            <option value="false">No</option>\n            </select>')

    def test_render_value_true(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_html(self.widget, 'is_cool', 'true', html='<select name="is_cool">\n            <option value="unknown">Unknown</option>\n            <option value="true" selected>Yes</option>\n            <option value="false">No</option>\n            </select>')

    def test_render_value_false(self):
        if False:
            while True:
                i = 10
        self.check_html(self.widget, 'is_cool', 'false', html='<select name="is_cool">\n            <option value="unknown">Unknown</option>\n            <option value="true">Yes</option>\n            <option value="false" selected>No</option>\n            </select>')

    def test_render_value_1(self):
        if False:
            while True:
                i = 10
        self.check_html(self.widget, 'is_cool', '1', html='<select name="is_cool">\n            <option value="unknown" selected>Unknown</option>\n            <option value="true">Yes</option>\n            <option value="false">No</option>\n            </select>')

    def test_render_value_2(self):
        if False:
            return 10
        self.check_html(self.widget, 'is_cool', '2', html='<select name="is_cool">\n            <option value="unknown">Unknown</option>\n            <option value="true" selected>Yes</option>\n            <option value="false">No</option>\n            </select>')

    def test_render_value_3(self):
        if False:
            i = 10
            return i + 15
        self.check_html(self.widget, 'is_cool', '3', html='<select name="is_cool">\n            <option value="unknown">Unknown</option>\n            <option value="true">Yes</option>\n            <option value="false" selected>No</option>\n            </select>')

    def test_l10n(self):
        if False:
            print('Hello World!')
        "\n        The NullBooleanSelect widget's options are lazily localized (#17190).\n        "
        widget = NullBooleanSelect()
        with translation.override('de-at'):
            self.check_html(widget, 'id_bool', True, html='\n                <select name="id_bool">\n                    <option value="unknown">Unbekannt</option>\n                    <option value="true" selected>Ja</option>\n                    <option value="false">Nein</option>\n                </select>\n                ')

    def test_fieldset(self):
        if False:
            while True:
                i = 10

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = NullBooleanField(widget=self.widget)
        form = TestForm()
        self.assertIs(self.widget.use_fieldset, False)
        self.assertHTMLEqual('<div><label for="id_field">Field:</label><select name="field" id="id_field"><option value="unknown" selected>Unknown</option><option value="true">Yes</option><option value="false">No</option></select></div>', form.render())