import datetime
from django import forms
from django.forms import CheckboxSelectMultiple, ChoiceField, Form
from django.test import override_settings
from .base import WidgetTest

class CheckboxSelectMultipleTest(WidgetTest):
    widget = CheckboxSelectMultiple

    def test_render_value(self):
        if False:
            i = 10
            return i + 15
        self.check_html(self.widget(choices=self.beatles), 'beatles', ['J'], html='\n            <div>\n            <div><label><input checked type="checkbox" name="beatles" value="J"> John\n            </label></div>\n            <div><label><input type="checkbox" name="beatles" value="P"> Paul\n            </label></div>\n            <div><label><input type="checkbox" name="beatles" value="G"> George\n            </label></div>\n            <div><label><input type="checkbox" name="beatles" value="R"> Ringo\n            </label></div>\n            </div>\n        ')

    def test_render_value_multiple(self):
        if False:
            return 10
        self.check_html(self.widget(choices=self.beatles), 'beatles', ['J', 'P'], html='\n            <div>\n            <div><label><input checked type="checkbox" name="beatles" value="J"> John\n            </label></div>\n            <div><label><input checked type="checkbox" name="beatles" value="P"> Paul\n            </label></div>\n            <div><label><input type="checkbox" name="beatles" value="G"> George\n            </label></div>\n            <div><label><input type="checkbox" name="beatles" value="R"> Ringo\n            </label></div>\n            </div>\n        ')

    def test_render_none(self):
        if False:
            print('Hello World!')
        '\n        If the value is None, none of the options are selected, even if the\n        choices have an empty option.\n        '
        self.check_html(self.widget(choices=(('', 'Unknown'),) + self.beatles), 'beatles', None, html='\n            <div>\n            <div><label><input type="checkbox" name="beatles" value=""> Unknown\n            </label></div>\n            <div><label><input type="checkbox" name="beatles" value="J"> John\n            </label></div>\n            <div><label><input type="checkbox" name="beatles" value="P"> Paul\n            </label></div>\n            <div><label><input type="checkbox" name="beatles" value="G"> George\n            </label></div>\n            <div><label><input type="checkbox" name="beatles" value="R"> Ringo\n            </label></div>\n            </div>\n        ')

    def test_nested_choices(self):
        if False:
            while True:
                i = 10
        nested_choices = (('unknown', 'Unknown'), ('Audio', (('vinyl', 'Vinyl'), ('cd', 'CD'))), ('Video', (('vhs', 'VHS'), ('dvd', 'DVD'))))
        html = '\n        <div id="media">\n        <div> <label for="media_0">\n        <input type="checkbox" name="nestchoice" value="unknown" id="media_0"> Unknown\n        </label></div>\n        <div>\n        <label>Audio</label>\n        <div> <label for="media_1_0">\n        <input checked type="checkbox" name="nestchoice" value="vinyl" id="media_1_0">\n        Vinyl</label></div>\n        <div> <label for="media_1_1">\n        <input type="checkbox" name="nestchoice" value="cd" id="media_1_1"> CD\n        </label></div>\n        </div><div>\n        <label>Video</label>\n        <div> <label for="media_2_0">\n        <input type="checkbox" name="nestchoice" value="vhs" id="media_2_0"> VHS\n        </label></div>\n        <div> <label for="media_2_1">\n        <input type="checkbox" name="nestchoice" value="dvd" id="media_2_1" checked> DVD\n        </label></div>\n        </div>\n        </div>\n        '
        self.check_html(self.widget(choices=nested_choices), 'nestchoice', ('vinyl', 'dvd'), attrs={'id': 'media'}, html=html)

    def test_nested_choices_without_id(self):
        if False:
            i = 10
            return i + 15
        nested_choices = (('unknown', 'Unknown'), ('Audio', (('vinyl', 'Vinyl'), ('cd', 'CD'))), ('Video', (('vhs', 'VHS'), ('dvd', 'DVD'))))
        html = '\n        <div>\n        <div> <label>\n        <input type="checkbox" name="nestchoice" value="unknown"> Unknown</label></div>\n        <div>\n        <label>Audio</label>\n        <div> <label>\n        <input checked type="checkbox" name="nestchoice" value="vinyl"> Vinyl\n        </label></div>\n        <div> <label>\n        <input type="checkbox" name="nestchoice" value="cd"> CD</label></div>\n        </div><div>\n        <label>Video</label>\n        <div> <label>\n        <input type="checkbox" name="nestchoice" value="vhs"> VHS</label></div>\n        <div> <label>\n        <input type="checkbox" name="nestchoice" value="dvd"checked> DVD</label></div>\n        </div>\n        </div>\n        '
        self.check_html(self.widget(choices=nested_choices), 'nestchoice', ('vinyl', 'dvd'), html=html)

    def test_separate_ids(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Each input gets a separate ID.\n        '
        choices = [('a', 'A'), ('b', 'B'), ('c', 'C')]
        html = '\n        <div id="abc">\n        <div>\n        <label for="abc_0">\n        <input checked type="checkbox" name="letters" value="a" id="abc_0"> A</label>\n        </div>\n        <div><label for="abc_1">\n        <input type="checkbox" name="letters" value="b" id="abc_1"> B</label></div>\n        <div>\n        <label for="abc_2">\n        <input checked type="checkbox" name="letters" value="c" id="abc_2"> C</label>\n        </div>\n        </div>\n        '
        self.check_html(self.widget(choices=choices), 'letters', ['a', 'c'], attrs={'id': 'abc'}, html=html)

    def test_separate_ids_constructor(self):
        if False:
            while True:
                i = 10
        '\n        Each input gets a separate ID when the ID is passed to the constructor.\n        '
        widget = CheckboxSelectMultiple(attrs={'id': 'abc'}, choices=[('a', 'A'), ('b', 'B'), ('c', 'C')])
        html = '\n        <div id="abc">\n        <div>\n        <label for="abc_0">\n        <input checked type="checkbox" name="letters" value="a" id="abc_0"> A</label>\n        </div>\n        <div><label for="abc_1">\n        <input type="checkbox" name="letters" value="b" id="abc_1"> B</label></div>\n        <div>\n        <label for="abc_2">\n        <input checked type="checkbox" name="letters" value="c" id="abc_2"> C</label>\n        </div>\n        </div>\n        '
        self.check_html(widget, 'letters', ['a', 'c'], html=html)

    @override_settings(USE_THOUSAND_SEPARATOR=True)
    def test_doesnt_localize_input_value(self):
        if False:
            return 10
        choices = [(1, 'One'), (1000, 'One thousand'), (1000000, 'One million')]
        html = '\n        <div>\n        <div><label><input type="checkbox" name="numbers" value="1"> One</label></div>\n        <div><label>\n        <input type="checkbox" name="numbers" value="1000"> One thousand</label></div>\n        <div><label>\n        <input type="checkbox" name="numbers" value="1000000"> One million</label></div>\n        </div>\n        '
        self.check_html(self.widget(choices=choices), 'numbers', None, html=html)
        choices = [(datetime.time(0, 0), 'midnight'), (datetime.time(12, 0), 'noon')]
        html = '\n        <div>\n        <div><label>\n        <input type="checkbox" name="times" value="00:00:00"> midnight</label></div>\n        <div><label>\n        <input type="checkbox" name="times" value="12:00:00"> noon</label></div>\n        </div>\n        '
        self.check_html(self.widget(choices=choices), 'times', None, html=html)

    def test_use_required_attribute(self):
        if False:
            i = 10
            return i + 15
        widget = self.widget(choices=self.beatles)
        self.assertIs(widget.use_required_attribute(None), False)
        self.assertIs(widget.use_required_attribute([]), False)
        self.assertIs(widget.use_required_attribute(['J', 'P']), False)

    def test_value_omitted_from_data(self):
        if False:
            print('Hello World!')
        widget = self.widget(choices=self.beatles)
        self.assertIs(widget.value_omitted_from_data({}, {}, 'field'), False)
        self.assertIs(widget.value_omitted_from_data({'field': 'value'}, {}, 'field'), False)

    def test_label(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        CheckboxSelectMultiple doesn\'t contain \'for="field_0"\' in the <label>\n        because clicking that would toggle the first checkbox.\n        '

        class TestForm(forms.Form):
            f = forms.MultipleChoiceField(widget=CheckboxSelectMultiple)
        bound_field = TestForm()['f']
        self.assertEqual(bound_field.field.widget.id_for_label('id'), '')
        self.assertEqual(bound_field.label_tag(), '<label>F:</label>')
        self.assertEqual(bound_field.legend_tag(), '<legend>F:</legend>')

    def test_fieldset(self):
        if False:
            i = 10
            return i + 15

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = ChoiceField(widget=self.widget, choices=self.beatles)
        form = TestForm()
        self.assertIs(self.widget.use_fieldset, True)
        self.assertHTMLEqual(form.render(), '<div><fieldset><legend>Field:</legend><div id="id_field"><div><label for="id_field_0"><input type="checkbox" name="field" value="J" id="id_field_0"> John</label></div><div><label for="id_field_1"><input type="checkbox" name="field" value="P" id="id_field_1">Paul</label></div><div><label for="id_field_2"><input type="checkbox" name="field" value="G" id="id_field_2"> George</label></div><div><label for="id_field_3"><input type="checkbox" name="field" value="R" id="id_field_3">Ringo</label></div></div></fieldset></div>')