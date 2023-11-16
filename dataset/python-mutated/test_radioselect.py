import datetime
from django.forms import ChoiceField, Form, MultiWidget, RadioSelect, TextInput
from django.test import override_settings
from django.utils.safestring import mark_safe
from .test_choicewidget import ChoiceWidgetTest
BLANK_CHOICE_DASH = (('', '------'),)

class RadioSelectTest(ChoiceWidgetTest):
    widget = RadioSelect

    def test_render(self):
        if False:
            for i in range(10):
                print('nop')
        html = '\n        <div>\n          <div>\n            <label><input type="radio" name="beatle" value="">------</label>\n          </div>\n          <div>\n            <label><input checked type="radio" name="beatle" value="J">John</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="P">Paul</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="G">George</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="R">Ringo</label>\n          </div>\n        </div>\n        '
        beatles_with_blank = BLANK_CHOICE_DASH + self.beatles
        for choices in (beatles_with_blank, dict(beatles_with_blank)):
            with self.subTest(choices):
                self.check_html(self.widget(choices=choices), 'beatle', 'J', html=html)

    def test_nested_choices(self):
        if False:
            i = 10
            return i + 15
        nested_choices = (('unknown', 'Unknown'), ('Audio', (('vinyl', 'Vinyl'), ('cd', 'CD'))), ('Video', (('vhs', 'VHS'), ('dvd', 'DVD'))))
        html = '\n        <div id="media">\n        <div>\n        <label for="media_0">\n        <input type="radio" name="nestchoice" value="unknown" id="media_0"> Unknown\n        </label></div>\n        <div>\n        <label>Audio</label>\n        <div>\n        <label for="media_1_0">\n        <input type="radio" name="nestchoice" value="vinyl" id="media_1_0"> Vinyl\n        </label></div>\n        <div> <label for="media_1_1">\n        <input type="radio" name="nestchoice" value="cd" id="media_1_1"> CD\n        </label></div>\n        </div><div>\n        <label>Video</label>\n        <div>\n        <label for="media_2_0">\n        <input type="radio" name="nestchoice" value="vhs" id="media_2_0"> VHS\n        </label></div>\n        <div>\n        <label for="media_2_1">\n        <input type="radio" name="nestchoice" value="dvd" id="media_2_1" checked> DVD\n        </label></div>\n        </div>\n        </div>\n        '
        self.check_html(self.widget(choices=nested_choices), 'nestchoice', 'dvd', attrs={'id': 'media'}, html=html)

    def test_render_none(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If value is None, none of the options are selected.\n        '
        choices = BLANK_CHOICE_DASH + self.beatles
        html = '\n        <div>\n          <div>\n            <label><input checked type="radio" name="beatle" value="">------</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="J">John</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="P">Paul</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="G">George</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="R">Ringo</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=choices), 'beatle', None, html=html)

    def test_render_label_value(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the value corresponds to a label (but not to an option value), none\n        of the options are selected.\n        '
        html = '\n        <div>\n          <div>\n            <label><input type="radio" name="beatle" value="J">John</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="P">Paul</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="G">George</label>\n          </div>\n          <div>\n            <label><input type="radio" name="beatle" value="R">Ringo</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=self.beatles), 'beatle', 'Ringo', html=html)

    def test_render_selected(self):
        if False:
            return 10
        '\n        Only one option can be selected.\n        '
        choices = [('0', '0'), ('1', '1'), ('2', '2'), ('3', '3'), ('0', 'extra')]
        html = '\n        <div>\n          <div>\n            <label><input checked type="radio" name="choices" value="0">0</label>\n          </div>\n          <div>\n            <label><input type="radio" name="choices" value="1">1</label>\n          </div>\n          <div>\n            <label><input type="radio" name="choices" value="2">2</label>\n          </div>\n          <div>\n            <label><input type="radio" name="choices" value="3">3</label>\n          </div>\n          <div>\n            <label><input type="radio" name="choices" value="0">extra</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=choices), 'choices', '0', html=html)

    def test_constructor_attrs(self):
        if False:
            while True:
                i = 10
        '\n        Attributes provided at instantiation are passed to the constituent\n        inputs.\n        '
        widget = self.widget(attrs={'id': 'foo'}, choices=self.beatles)
        html = '\n        <div id="foo">\n          <div>\n            <label for="foo_0">\n            <input checked type="radio" id="foo_0" value="J" name="beatle">John</label>\n          </div>\n          <div><label for="foo_1">\n            <input type="radio" id="foo_1" value="P" name="beatle">Paul</label>\n          </div>\n          <div><label for="foo_2">\n            <input type="radio" id="foo_2" value="G" name="beatle">George</label>\n          </div>\n          <div><label for="foo_3">\n            <input type="radio" id="foo_3" value="R" name="beatle">Ringo</label>\n          </div>\n        </div>\n        '
        self.check_html(widget, 'beatle', 'J', html=html)

    def test_compare_to_str(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The value is compared to its str().\n        '
        html = '\n        <div>\n          <div>\n            <label><input type="radio" name="num" value="1">1</label>\n          </div>\n          <div>\n            <label><input type="radio" name="num" value="2">2</label>\n          </div>\n          <div>\n            <label><input checked type="radio" name="num" value="3">3</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=[('1', '1'), ('2', '2'), ('3', '3')]), 'num', 3, html=html)
        self.check_html(self.widget(choices=[(1, 1), (2, 2), (3, 3)]), 'num', '3', html=html)
        self.check_html(self.widget(choices=[(1, 1), (2, 2), (3, 3)]), 'num', 3, html=html)

    def test_choices_constructor(self):
        if False:
            while True:
                i = 10
        widget = self.widget(choices=[(1, 1), (2, 2), (3, 3)])
        html = '\n        <div>\n          <div>\n            <label><input type="radio" name="num" value="1">1</label>\n          </div>\n          <div>\n            <label><input type="radio" name="num" value="2">2</label>\n          </div>\n          <div>\n            <label><input checked type="radio" name="num" value="3">3</label>\n          </div>\n        </div>\n        '
        self.check_html(widget, 'num', 3, html=html)

    def test_choices_constructor_generator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If choices is passed to the constructor and is a generator, it can be\n        iterated over multiple times without getting consumed.\n        '

        def get_choices():
            if False:
                print('Hello World!')
            for i in range(4):
                yield (i, i)
        html = '\n        <div>\n          <div>\n            <label><input type="radio" name="num" value="0">0</label>\n          </div>\n          <div>\n            <label><input type="radio" name="num" value="1">1</label>\n          </div>\n          <div>\n            <label><input type="radio" name="num" value="2">2</label>\n          </div>\n          <div>\n            <label><input checked type="radio" name="num" value="3">3</label>\n          </div>\n        </div>\n        '
        widget = self.widget(choices=get_choices())
        self.check_html(widget, 'num', 3, html=html)

    def test_choices_escaping(self):
        if False:
            i = 10
            return i + 15
        choices = (('bad', 'you & me'), ('good', mark_safe('you &gt; me')))
        html = '\n        <div>\n          <div>\n            <label><input type="radio" name="escape" value="bad">you & me</label>\n          </div>\n          <div>\n            <label><input type="radio" name="escape" value="good">you &gt; me</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=choices), 'escape', None, html=html)

    def test_choices_unicode(self):
        if False:
            print('Hello World!')
        html = '\n        <div>\n          <div>\n            <label>\n            <input checked type="radio" name="email"\n              value="ŠĐĆŽćžšđ">\n            ŠĐabcĆŽćžšđ</label>\n          </div>\n          <div>\n            <label>\n            <input type="radio" name="email" value="ćžšđ">\n            abcćžšđ</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=[('ŠĐĆŽćžšđ', 'ŠĐabcĆŽćžšđ'), ('ćžšđ', 'abcćžšđ')]), 'email', 'ŠĐĆŽćžšđ', html=html)

    def test_choices_optgroup(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Choices can be nested one level in order to create HTML optgroups.\n        '
        html = '\n        <div>\n          <div>\n            <label><input type="radio" name="nestchoice" value="outer1">Outer 1</label>\n          </div>\n          <div>\n            <label>Group &quot;1&quot;</label>\n            <div>\n              <label>\n              <input type="radio" name="nestchoice" value="inner1">Inner 1</label>\n            </div>\n            <div>\n              <label>\n              <input type="radio" name="nestchoice" value="inner2">Inner 2</label>\n            </div>\n          </div>\n        </div>\n        '
        for widget in self.nested_widgets:
            with self.subTest(widget):
                self.check_html(widget, 'nestchoice', None, html=html)

    def test_choices_select_outer(self):
        if False:
            for i in range(10):
                print('nop')
        html = '\n        <div>\n          <div>\n            <label>\n            <input checked type="radio" name="nestchoice" value="outer1">Outer 1</label>\n          </div>\n          <div>\n            <label>Group &quot;1&quot;</label>\n            <div>\n              <label>\n              <input type="radio" name="nestchoice" value="inner1">Inner 1</label>\n            </div>\n            <div>\n              <label>\n              <input type="radio" name="nestchoice" value="inner2">Inner 2</label>\n            </div>\n          </div>\n        </div>\n        '
        for widget in self.nested_widgets:
            with self.subTest(widget):
                self.check_html(widget, 'nestchoice', 'outer1', html=html)

    def test_choices_select_inner(self):
        if False:
            print('Hello World!')
        html = '\n        <div>\n          <div>\n            <label><input type="radio" name="nestchoice" value="outer1">Outer 1</label>\n          </div>\n          <div>\n            <label>Group &quot;1&quot;</label>\n            <div>\n              <label>\n              <input type="radio" name="nestchoice" value="inner1">Inner 1</label>\n            </div>\n            <div>\n              <label>\n                <input checked type="radio" name="nestchoice" value="inner2">Inner 2\n              </label>\n            </div>\n          </div>\n        </div>\n        '
        for widget in self.nested_widgets:
            with self.subTest(widget):
                self.check_html(widget, 'nestchoice', 'inner2', html=html)

    def test_render_attrs(self):
        if False:
            print('Hello World!')
        '\n        Attributes provided at render-time are passed to the constituent\n        inputs.\n        '
        html = '\n        <div id="bar">\n          <div>\n            <label for="bar_0">\n            <input checked type="radio" id="bar_0" value="J" name="beatle">John</label>\n          </div>\n          <div><label for="bar_1">\n            <input type="radio" id="bar_1" value="P" name="beatle">Paul</label>\n          </div>\n          <div><label for="bar_2">\n            <input type="radio" id="bar_2" value="G" name="beatle">George</label>\n          </div>\n          <div><label for="bar_3">\n            <input type="radio" id="bar_3" value="R" name="beatle">Ringo</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=self.beatles), 'beatle', 'J', attrs={'id': 'bar'}, html=html)

    def test_class_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The <div> in the multiple_input.html widget template include the class\n        attribute.\n        '
        html = '\n        <div class="bar">\n          <div><label>\n            <input checked type="radio" class="bar" value="J" name="beatle">John</label>\n          </div>\n          <div><label>\n            <input type="radio" class="bar" value="P" name="beatle">Paul</label>\n          </div>\n          <div><label>\n            <input type="radio" class="bar" value="G" name="beatle">George</label>\n          </div>\n          <div><label>\n            <input type="radio" class="bar" value="R" name="beatle">Ringo</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=self.beatles), 'beatle', 'J', attrs={'class': 'bar'}, html=html)

    @override_settings(USE_THOUSAND_SEPARATOR=True)
    def test_doesnt_localize_input_value(self):
        if False:
            for i in range(10):
                print('nop')
        choices = [(1, 'One'), (1000, 'One thousand'), (1000000, 'One million')]
        html = '\n        <div>\n          <div><label><input type="radio" name="number" value="1">One</label></div>\n          <div>\n            <label><input type="radio" name="number" value="1000">One thousand</label>\n          </div>\n          <div>\n            <label><input type="radio" name="number" value="1000000">One million</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=choices), 'number', None, html=html)
        choices = [(datetime.time(0, 0), 'midnight'), (datetime.time(12, 0), 'noon')]
        html = '\n        <div>\n          <div>\n            <label><input type="radio" name="time" value="00:00:00">midnight</label>\n          </div>\n          <div>\n            <label><input type="radio" name="time" value="12:00:00">noon</label>\n          </div>\n        </div>\n        '
        self.check_html(self.widget(choices=choices), 'time', None, html=html)

    def test_render_as_subwidget(self):
        if False:
            for i in range(10):
                print('nop')
        'A RadioSelect as a subwidget of MultiWidget.'
        choices = BLANK_CHOICE_DASH + self.beatles
        html = '\n        <div>\n          <div><label>\n            <input type="radio" name="beatle_0" value="">------</label>\n          </div>\n          <div><label>\n            <input checked type="radio" name="beatle_0" value="J">John</label>\n          </div>\n          <div><label>\n            <input type="radio" name="beatle_0" value="P">Paul</label>\n          </div>\n          <div><label>\n            <input type="radio" name="beatle_0" value="G">George</label>\n          </div>\n          <div><label>\n            <input type="radio" name="beatle_0" value="R">Ringo</label>\n          </div>\n        </div>\n        <input name="beatle_1" type="text" value="Some text">\n        '
        self.check_html(MultiWidget([self.widget(choices=choices), TextInput()]), 'beatle', ['J', 'Some text'], html=html)

    def test_fieldset(self):
        if False:
            print('Hello World!')

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = ChoiceField(widget=self.widget, choices=self.beatles, required=False)
        form = TestForm()
        self.assertIs(self.widget.use_fieldset, True)
        self.assertHTMLEqual('<div><fieldset><legend>Field:</legend><div id="id_field"><div><label for="id_field_0"><input type="radio" name="field" value="J" id="id_field_0"> John</label></div><div><label for="id_field_1"><input type="radio" name="field" value="P" id="id_field_1">Paul</label></div><div><label for="id_field_2"><input type="radio" name="field" value="G" id="id_field_2"> George</label></div><div><label for="id_field_3"><input type="radio" name="field" value="R" id="id_field_3">Ringo</label></div></div></fieldset></div>', form.render())