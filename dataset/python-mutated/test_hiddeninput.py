from django.forms import CharField, Form, HiddenInput
from .base import WidgetTest

class HiddenInputTest(WidgetTest):
    widget = HiddenInput()

    def test_render(self):
        if False:
            while True:
                i = 10
        self.check_html(self.widget, 'email', '', html='<input type="hidden" name="email">')

    def test_use_required_attribute(self):
        if False:
            i = 10
            return i + 15
        self.assertIs(self.widget.use_required_attribute(None), False)
        self.assertIs(self.widget.use_required_attribute(''), False)
        self.assertIs(self.widget.use_required_attribute('foo'), False)

    def test_fieldset(self):
        if False:
            for i in range(10):
                print('nop')

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = CharField(widget=self.widget)
        form = TestForm()
        self.assertIs(self.widget.use_fieldset, False)
        self.assertHTMLEqual('<input type="hidden" name="field" id="id_field">', form.render())