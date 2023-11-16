import copy
from datetime import datetime
from django.forms import CharField, FileInput, Form, MultipleChoiceField, MultiValueField, MultiWidget, RadioSelect, SelectMultiple, SplitDateTimeField, SplitDateTimeWidget, TextInput
from .base import WidgetTest

class MyMultiWidget(MultiWidget):

    def decompress(self, value):
        if False:
            return 10
        if value:
            return value.split('__')
        return ['', '']

class ComplexMultiWidget(MultiWidget):

    def __init__(self, attrs=None):
        if False:
            i = 10
            return i + 15
        widgets = (TextInput(), SelectMultiple(choices=WidgetTest.beatles), SplitDateTimeWidget())
        super().__init__(widgets, attrs)

    def decompress(self, value):
        if False:
            return 10
        if value:
            data = value.split(',')
            return [data[0], list(data[1]), datetime.strptime(data[2], '%Y-%m-%d %H:%M:%S')]
        return [None, None, None]

class ComplexField(MultiValueField):

    def __init__(self, required=True, widget=None, label=None, initial=None):
        if False:
            i = 10
            return i + 15
        fields = (CharField(), MultipleChoiceField(choices=WidgetTest.beatles), SplitDateTimeField())
        super().__init__(fields, required=required, widget=widget, label=label, initial=initial)

    def compress(self, data_list):
        if False:
            while True:
                i = 10
        if data_list:
            return '%s,%s,%s' % (data_list[0], ''.join(data_list[1]), data_list[2])
        return None

class DeepCopyWidget(MultiWidget):
    """
    Used to test MultiWidget.__deepcopy__().
    """

    def __init__(self, choices=[]):
        if False:
            i = 10
            return i + 15
        widgets = [RadioSelect(choices=choices), TextInput]
        super().__init__(widgets)

    def _set_choices(self, choices):
        if False:
            print('Hello World!')
        '\n        When choices are set for this widget, we want to pass those along to\n        the Select widget.\n        '
        self.widgets[0].choices = choices

    def _get_choices(self):
        if False:
            return 10
        "\n        The choices for this widget are the Select widget's choices.\n        "
        return self.widgets[0].choices
    choices = property(_get_choices, _set_choices)

class MultiWidgetTest(WidgetTest):

    def test_subwidgets_name(self):
        if False:
            i = 10
            return i + 15
        widget = MultiWidget(widgets={'': TextInput(), 'big': TextInput(attrs={'class': 'big'}), 'small': TextInput(attrs={'class': 'small'})})
        self.check_html(widget, 'name', ['John', 'George', 'Paul'], html='<input type="text" name="name" value="John"><input type="text" name="name_big" value="George" class="big"><input type="text" name="name_small" value="Paul" class="small">')

    def test_text_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        widget = MyMultiWidget(widgets=(TextInput(attrs={'class': 'big'}), TextInput(attrs={'class': 'small'})))
        self.check_html(widget, 'name', ['john', 'lennon'], html='<input type="text" class="big" value="john" name="name_0"><input type="text" class="small" value="lennon" name="name_1">')
        self.check_html(widget, 'name', ('john', 'lennon'), html='<input type="text" class="big" value="john" name="name_0"><input type="text" class="small" value="lennon" name="name_1">')
        self.check_html(widget, 'name', 'john__lennon', html='<input type="text" class="big" value="john" name="name_0"><input type="text" class="small" value="lennon" name="name_1">')
        self.check_html(widget, 'name', 'john__lennon', attrs={'id': 'foo'}, html='<input id="foo_0" type="text" class="big" value="john" name="name_0"><input id="foo_1" type="text" class="small" value="lennon" name="name_1">')

    def test_constructor_attrs(self):
        if False:
            return 10
        widget = MyMultiWidget(widgets=(TextInput(attrs={'class': 'big'}), TextInput(attrs={'class': 'small'})), attrs={'id': 'bar'})
        self.check_html(widget, 'name', ['john', 'lennon'], html='<input id="bar_0" type="text" class="big" value="john" name="name_0"><input id="bar_1" type="text" class="small" value="lennon" name="name_1">')

    def test_constructor_attrs_with_type(self):
        if False:
            i = 10
            return i + 15
        attrs = {'type': 'number'}
        widget = MyMultiWidget(widgets=(TextInput, TextInput()), attrs=attrs)
        self.check_html(widget, 'code', ['1', '2'], html='<input type="number" value="1" name="code_0"><input type="number" value="2" name="code_1">')
        widget = MyMultiWidget(widgets=(TextInput(attrs), TextInput(attrs)), attrs={'class': 'bar'})
        self.check_html(widget, 'code', ['1', '2'], html='<input type="number" value="1" name="code_0" class="bar"><input type="number" value="2" name="code_1" class="bar">')

    def test_value_omitted_from_data(self):
        if False:
            i = 10
            return i + 15
        widget = MyMultiWidget(widgets=(TextInput(), TextInput()))
        self.assertIs(widget.value_omitted_from_data({}, {}, 'field'), True)
        self.assertIs(widget.value_omitted_from_data({'field_0': 'x'}, {}, 'field'), False)
        self.assertIs(widget.value_omitted_from_data({'field_1': 'y'}, {}, 'field'), False)
        self.assertIs(widget.value_omitted_from_data({'field_0': 'x', 'field_1': 'y'}, {}, 'field'), False)

    def test_value_from_datadict_subwidgets_name(self):
        if False:
            for i in range(10):
                print('nop')
        widget = MultiWidget(widgets={'x': TextInput(), '': TextInput()})
        tests = [({}, [None, None]), ({'field': 'x'}, [None, 'x']), ({'field_x': 'y'}, ['y', None]), ({'field': 'x', 'field_x': 'y'}, ['y', 'x'])]
        for (data, expected) in tests:
            with self.subTest(data):
                self.assertEqual(widget.value_from_datadict(data, {}, 'field'), expected)

    def test_value_omitted_from_data_subwidgets_name(self):
        if False:
            i = 10
            return i + 15
        widget = MultiWidget(widgets={'x': TextInput(), '': TextInput()})
        tests = [({}, True), ({'field': 'x'}, False), ({'field_x': 'y'}, False), ({'field': 'x', 'field_x': 'y'}, False)]
        for (data, expected) in tests:
            with self.subTest(data):
                self.assertIs(widget.value_omitted_from_data(data, {}, 'field'), expected)

    def test_needs_multipart_true(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        needs_multipart_form should be True if any widgets need it.\n        '
        widget = MyMultiWidget(widgets=(TextInput(), FileInput()))
        self.assertTrue(widget.needs_multipart_form)

    def test_needs_multipart_false(self):
        if False:
            print('Hello World!')
        '\n        needs_multipart_form should be False if no widgets need it.\n        '
        widget = MyMultiWidget(widgets=(TextInput(), TextInput()))
        self.assertFalse(widget.needs_multipart_form)

    def test_nested_multiwidget(self):
        if False:
            return 10
        '\n        MultiWidgets can be composed of other MultiWidgets.\n        '
        widget = ComplexMultiWidget()
        self.check_html(widget, 'name', 'some text,JP,2007-04-25 06:24:00', html='\n            <input type="text" name="name_0" value="some text">\n            <select multiple name="name_1">\n                <option value="J" selected>John</option>\n                <option value="P" selected>Paul</option>\n                <option value="G">George</option>\n                <option value="R">Ringo</option>\n            </select>\n            <input type="text" name="name_2_0" value="2007-04-25">\n            <input type="text" name="name_2_1" value="06:24:00">\n            ')

    def test_no_whitespace_between_widgets(self):
        if False:
            for i in range(10):
                print('nop')
        widget = MyMultiWidget(widgets=(TextInput, TextInput()))
        self.check_html(widget, 'code', None, html='<input type="text" name="code_0"><input type="text" name="code_1">', strict=True)

    def test_deepcopy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        MultiWidget should define __deepcopy__() (#12048).\n        '
        w1 = DeepCopyWidget(choices=[1, 2, 3])
        w2 = copy.deepcopy(w1)
        w2.choices = [4, 5, 6]
        self.assertEqual(w1.choices, [1, 2, 3])

    def test_fieldset(self):
        if False:
            for i in range(10):
                print('nop')

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = ComplexField(widget=ComplexMultiWidget)
        form = TestForm()
        self.assertIs(form['field'].field.widget.use_fieldset, True)
        self.assertHTMLEqual('<div><fieldset><legend>Field:</legend><input type="text" name="field_0" required id="id_field_0"><select name="field_1" required id="id_field_1" multiple><option value="J">John</option><option value="P">Paul</option><option value="G">George</option><option value="R">Ringo</option></select><input type="text" name="field_2_0" required id="id_field_2_0"><input type="text" name="field_2_1" required id="id_field_2_1"></fieldset></div>', form.render())