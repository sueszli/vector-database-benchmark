from django.forms import ChoiceField, Form, SelectMultiple
from .base import WidgetTest

class SelectMultipleTest(WidgetTest):
    widget = SelectMultiple
    numeric_choices = (('0', '0'), ('1', '1'), ('2', '2'), ('3', '3'), ('0', 'extra'))

    def test_format_value(self):
        if False:
            print('Hello World!')
        widget = self.widget(choices=self.numeric_choices)
        self.assertEqual(widget.format_value(None), [])
        self.assertEqual(widget.format_value(''), [''])
        self.assertEqual(widget.format_value([3, 0, 1]), ['3', '0', '1'])

    def test_render_selected(self):
        if False:
            print('Hello World!')
        self.check_html(self.widget(choices=self.beatles), 'beatles', ['J'], html='<select multiple name="beatles">\n            <option value="J" selected>John</option>\n            <option value="P">Paul</option>\n            <option value="G">George</option>\n            <option value="R">Ringo</option>\n            </select>')

    def test_render_multiple_selected(self):
        if False:
            return 10
        self.check_html(self.widget(choices=self.beatles), 'beatles', ['J', 'P'], html='<select multiple name="beatles">\n            <option value="J" selected>John</option>\n            <option value="P" selected>Paul</option>\n            <option value="G">George</option>\n            <option value="R">Ringo</option>\n            </select>')

    def test_render_none(self):
        if False:
            print('Hello World!')
        '\n        If the value is None, none of the options are selected, even if the\n        choices have an empty option.\n        '
        self.check_html(self.widget(choices=(('', 'Unknown'),) + self.beatles), 'beatles', None, html='<select multiple name="beatles">\n            <option value="">Unknown</option>\n            <option value="J">John</option>\n            <option value="P">Paul</option>\n            <option value="G">George</option>\n            <option value="R">Ringo</option>\n            </select>')

    def test_render_value_label(self):
        if False:
            i = 10
            return i + 15
        '\n        If the value corresponds to a label (but not to an option value), none\n        of the options are selected.\n        '
        self.check_html(self.widget(choices=self.beatles), 'beatles', ['John'], html='<select multiple name="beatles">\n            <option value="J">John</option>\n            <option value="P">Paul</option>\n            <option value="G">George</option>\n            <option value="R">Ringo</option>\n            </select>')

    def test_multiple_options_same_value(self):
        if False:
            while True:
                i = 10
        '\n        Multiple options with the same value can be selected (#8103).\n        '
        self.check_html(self.widget(choices=self.numeric_choices), 'choices', ['0'], html='<select multiple name="choices">\n            <option value="0" selected>0</option>\n            <option value="1">1</option>\n            <option value="2">2</option>\n            <option value="3">3</option>\n            <option value="0" selected>extra</option>\n            </select>')

    def test_multiple_values_invalid(self):
        if False:
            print('Hello World!')
        '\n        If multiple values are given, but some of them are not valid, the valid\n        ones are selected.\n        '
        self.check_html(self.widget(choices=self.beatles), 'beatles', ['J', 'G', 'foo'], html='<select multiple name="beatles">\n            <option value="J" selected>John</option>\n            <option value="P">Paul</option>\n            <option value="G" selected>George</option>\n            <option value="R">Ringo</option>\n            </select>')

    def test_compare_string(self):
        if False:
            i = 10
            return i + 15
        choices = [('1', '1'), ('2', '2'), ('3', '3')]
        self.check_html(self.widget(choices=choices), 'nums', [2], html='<select multiple name="nums">\n            <option value="1">1</option>\n            <option value="2" selected>2</option>\n            <option value="3">3</option>\n            </select>')
        self.check_html(self.widget(choices=choices), 'nums', ['2'], html='<select multiple name="nums">\n            <option value="1">1</option>\n            <option value="2" selected>2</option>\n            <option value="3">3</option>\n            </select>')
        self.check_html(self.widget(choices=choices), 'nums', [2], html='<select multiple name="nums">\n            <option value="1">1</option>\n            <option value="2" selected>2</option>\n            <option value="3">3</option>\n            </select>')

    def test_optgroup_select_multiple(self):
        if False:
            while True:
                i = 10
        widget = SelectMultiple(choices=(('outer1', 'Outer 1'), ('Group "1"', (('inner1', 'Inner 1'), ('inner2', 'Inner 2')))))
        self.check_html(widget, 'nestchoice', ['outer1', 'inner2'], html='<select multiple name="nestchoice">\n            <option value="outer1" selected>Outer 1</option>\n            <optgroup label="Group &quot;1&quot;">\n            <option value="inner1">Inner 1</option>\n            <option value="inner2" selected>Inner 2</option>\n            </optgroup>\n            </select>')

    def test_value_omitted_from_data(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.widget(choices=self.beatles)
        self.assertIs(widget.value_omitted_from_data({}, {}, 'field'), False)
        self.assertIs(widget.value_omitted_from_data({'field': 'value'}, {}, 'field'), False)

    def test_fieldset(self):
        if False:
            for i in range(10):
                print('nop')

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = ChoiceField(widget=self.widget, choices=self.beatles, required=False)
        form = TestForm()
        self.assertIs(self.widget.use_fieldset, False)
        self.assertHTMLEqual('<div><label for="id_field">Field:</label><select multiple name="field" id="id_field"><option value="J">John</option>  <option value="P">Paul</option><option value="G">George</option><option value="R">Ringo</option></select></div>', form.render())