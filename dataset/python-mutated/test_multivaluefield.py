from datetime import datetime
from django.core.exceptions import ValidationError
from django.forms import CharField, Form, MultipleChoiceField, MultiValueField, MultiWidget, SelectMultiple, SplitDateTimeField, SplitDateTimeWidget, TextInput
from django.test import SimpleTestCase
beatles = (('J', 'John'), ('P', 'Paul'), ('G', 'George'), ('R', 'Ringo'))

class PartiallyRequiredField(MultiValueField):

    def compress(self, data_list):
        if False:
            for i in range(10):
                print('nop')
        return ','.join(data_list) if data_list else None

class PartiallyRequiredForm(Form):
    f = PartiallyRequiredField(fields=(CharField(required=True), CharField(required=False)), required=True, require_all_fields=False, widget=MultiWidget(widgets=[TextInput(), TextInput()]))

class ComplexMultiWidget(MultiWidget):

    def __init__(self, attrs=None):
        if False:
            return 10
        widgets = (TextInput(), SelectMultiple(choices=beatles), SplitDateTimeWidget())
        super().__init__(widgets, attrs)

    def decompress(self, value):
        if False:
            i = 10
            return i + 15
        if value:
            data = value.split(',')
            return [data[0], list(data[1]), datetime.strptime(data[2], '%Y-%m-%d %H:%M:%S')]
        return [None, None, None]

class ComplexField(MultiValueField):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        fields = (CharField(), MultipleChoiceField(choices=beatles), SplitDateTimeField())
        super().__init__(fields, **kwargs)

    def compress(self, data_list):
        if False:
            print('Hello World!')
        if data_list:
            return '%s,%s,%s' % (data_list[0], ''.join(data_list[1]), data_list[2])
        return None

class ComplexFieldForm(Form):
    field1 = ComplexField(widget=ComplexMultiWidget())

class MultiValueFieldTest(SimpleTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.field = ComplexField(widget=ComplexMultiWidget())
        super().setUpClass()

    def test_clean(self):
        if False:
            return 10
        self.assertEqual(self.field.clean(['some text', ['J', 'P'], ['2007-04-25', '6:24:00']]), 'some text,JP,2007-04-25 06:24:00')

    def test_clean_disabled_multivalue(self):
        if False:
            i = 10
            return i + 15

        class ComplexFieldForm(Form):
            f = ComplexField(disabled=True, widget=ComplexMultiWidget)
        inputs = ('some text,JP,2007-04-25 06:24:00', ['some text', ['J', 'P'], ['2007-04-25', '6:24:00']])
        for data in inputs:
            with self.subTest(data=data):
                form = ComplexFieldForm({}, initial={'f': data})
                form.full_clean()
                self.assertEqual(form.errors, {})
                self.assertEqual(form.cleaned_data, {'f': inputs[0]})

    def test_bad_choice(self):
        if False:
            print('Hello World!')
        msg = "'Select a valid choice. X is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            self.field.clean(['some text', ['X'], ['2007-04-25', '6:24:00']])

    def test_no_value(self):
        if False:
            i = 10
            return i + 15
        '\n        If insufficient data is provided, None is substituted.\n        '
        msg = "'This field is required.'"
        with self.assertRaisesMessage(ValidationError, msg):
            self.field.clean(['some text', ['JP']])

    def test_has_changed_no_initial(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.field.has_changed(None, ['some text', ['J', 'P'], ['2007-04-25', '6:24:00']]))

    def test_has_changed_same(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.field.has_changed('some text,JP,2007-04-25 06:24:00', ['some text', ['J', 'P'], ['2007-04-25', '6:24:00']]))

    def test_has_changed_first_widget(self):
        if False:
            while True:
                i = 10
        "\n        Test when the first widget's data has changed.\n        "
        self.assertTrue(self.field.has_changed('some text,JP,2007-04-25 06:24:00', ['other text', ['J', 'P'], ['2007-04-25', '6:24:00']]))

    def test_has_changed_last_widget(self):
        if False:
            return 10
        "\n        Test when the last widget's data has changed. This ensures that it is\n        not short circuiting while testing the widgets.\n        "
        self.assertTrue(self.field.has_changed('some text,JP,2007-04-25 06:24:00', ['some text', ['J', 'P'], ['2009-04-25', '11:44:00']]))

    def test_disabled_has_changed(self):
        if False:
            i = 10
            return i + 15
        f = MultiValueField(fields=(CharField(), CharField()), disabled=True)
        self.assertIs(f.has_changed(['x', 'x'], ['y', 'y']), False)

    def test_form_as_table(self):
        if False:
            while True:
                i = 10
        form = ComplexFieldForm()
        self.assertHTMLEqual(form.as_table(), '\n            <tr><th><label>Field1:</label></th>\n            <td><input type="text" name="field1_0" id="id_field1_0" required>\n            <select multiple name="field1_1" id="id_field1_1" required>\n            <option value="J">John</option>\n            <option value="P">Paul</option>\n            <option value="G">George</option>\n            <option value="R">Ringo</option>\n            </select>\n            <input type="text" name="field1_2_0" id="id_field1_2_0" required>\n            <input type="text" name="field1_2_1" id="id_field1_2_1" required></td></tr>\n            ')

    def test_form_as_table_data(self):
        if False:
            for i in range(10):
                print('nop')
        form = ComplexFieldForm({'field1_0': 'some text', 'field1_1': ['J', 'P'], 'field1_2_0': '2007-04-25', 'field1_2_1': '06:24:00'})
        self.assertHTMLEqual(form.as_table(), '\n            <tr><th><label>Field1:</label></th>\n            <td><input type="text" name="field1_0" value="some text" id="id_field1_0"\n                required>\n            <select multiple name="field1_1" id="id_field1_1" required>\n            <option value="J" selected>John</option>\n            <option value="P" selected>Paul</option>\n            <option value="G">George</option>\n            <option value="R">Ringo</option>\n            </select>\n            <input type="text" name="field1_2_0" value="2007-04-25" id="id_field1_2_0"\n                required>\n            <input type="text" name="field1_2_1" value="06:24:00" id="id_field1_2_1"\n                required></td></tr>\n            ')

    def test_form_cleaned_data(self):
        if False:
            for i in range(10):
                print('nop')
        form = ComplexFieldForm({'field1_0': 'some text', 'field1_1': ['J', 'P'], 'field1_2_0': '2007-04-25', 'field1_2_1': '06:24:00'})
        form.is_valid()
        self.assertEqual(form.cleaned_data['field1'], 'some text,JP,2007-04-25 06:24:00')

    def test_render_required_attributes(self):
        if False:
            while True:
                i = 10
        form = PartiallyRequiredForm({'f_0': 'Hello', 'f_1': ''})
        self.assertTrue(form.is_valid())
        self.assertInHTML('<input type="text" name="f_0" value="Hello" required id="id_f_0">', form.as_p())
        self.assertInHTML('<input type="text" name="f_1" id="id_f_1">', form.as_p())
        form = PartiallyRequiredForm({'f_0': '', 'f_1': ''})
        self.assertFalse(form.is_valid())