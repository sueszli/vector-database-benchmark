from django.core.files.uploadedfile import SimpleUploadedFile
from django.forms import ClearableFileInput, FileField, Form, MultiWidget
from .base import WidgetTest

class FakeFieldFile:
    """
    Quacks like a FieldFile (has a .url and string representation), but
    doesn't require us to care about storages etc.
    """
    url = 'something'

    def __str__(self):
        if False:
            return 10
        return self.url

class ClearableFileInputTest(WidgetTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.widget = ClearableFileInput()

    def test_clear_input_renders(self):
        if False:
            return 10
        '\n        A ClearableFileInput with is_required False and rendered with an\n        initial value that is a file renders a clear checkbox.\n        '
        self.check_html(self.widget, 'myfile', FakeFieldFile(), html='\n            Currently: <a href="something">something</a>\n            <input type="checkbox" name="myfile-clear" id="myfile-clear_id">\n            <label for="myfile-clear_id">Clear</label><br>\n            Change: <input type="file" name="myfile">\n            ')

    def test_html_escaped(self):
        if False:
            while True:
                i = 10
        '\n        A ClearableFileInput should escape name, filename, and URL\n        when rendering HTML (#15182).\n        '

        class StrangeFieldFile:
            url = 'something?chapter=1&sect=2&copy=3&lang=en'

            def __str__(self):
                if False:
                    while True:
                        i = 10
                return 'something<div onclick="alert(\'oops\')">.jpg'
        self.check_html(ClearableFileInput(), 'my<div>file', StrangeFieldFile(), html='\n                Currently:\n                <a href="something?chapter=1&amp;sect=2&amp;copy=3&amp;lang=en">\n                something&lt;div onclick=&quot;alert(&#x27;oops&#x27;)&quot;&gt;.jpg</a>\n                <input type="checkbox" name="my&lt;div&gt;file-clear"\n                    id="my&lt;div&gt;file-clear_id">\n                <label for="my&lt;div&gt;file-clear_id">Clear</label><br>\n                Change: <input type="file" name="my&lt;div&gt;file">\n                ')

    def test_clear_input_renders_only_if_not_required(self):
        if False:
            i = 10
            return i + 15
        '\n        A ClearableFileInput with is_required=True does not render a clear\n        checkbox.\n        '
        widget = ClearableFileInput()
        widget.is_required = True
        self.check_html(widget, 'myfile', FakeFieldFile(), html='\n            Currently: <a href="something">something</a> <br>\n            Change: <input type="file" name="myfile">\n            ')

    def test_clear_input_renders_only_if_initial(self):
        if False:
            return 10
        '\n        A ClearableFileInput instantiated with no initial value does not render\n        a clear checkbox.\n        '
        self.check_html(self.widget, 'myfile', None, html='<input type="file" name="myfile">')

    def test_render_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_html(self.widget, 'myfile', FakeFieldFile(), attrs={'disabled': True}, html='Currently: <a href="something">something</a><input type="checkbox" name="myfile-clear" id="myfile-clear_id" disabled><label for="myfile-clear_id">Clear</label><br>Change: <input type="file" name="myfile" disabled>')

    def test_render_no_disabled(self):
        if False:
            i = 10
            return i + 15

        class TestForm(Form):
            clearable_file = FileField(widget=self.widget, initial=FakeFieldFile(), required=False)
        form = TestForm()
        with self.assertNoLogs('django.template', 'DEBUG'):
            form.render()

    def test_render_as_subwidget(self):
        if False:
            i = 10
            return i + 15
        'A ClearableFileInput as a subwidget of MultiWidget.'
        widget = MultiWidget(widgets=(self.widget,))
        self.check_html(widget, 'myfile', [FakeFieldFile()], html='\n            Currently: <a href="something">something</a>\n            <input type="checkbox" name="myfile_0-clear" id="myfile_0-clear_id">\n            <label for="myfile_0-clear_id">Clear</label><br>\n            Change: <input type="file" name="myfile_0">\n            ')

    def test_clear_input_checked_returns_false(self):
        if False:
            while True:
                i = 10
        '\n        ClearableFileInput.value_from_datadict returns False if the clear\n        checkbox is checked, if not required.\n        '
        value = self.widget.value_from_datadict(data={'myfile-clear': True}, files={}, name='myfile')
        self.assertIs(value, False)
        self.assertIs(self.widget.checked, True)

    def test_clear_input_checked_returns_false_only_if_not_required(self):
        if False:
            while True:
                i = 10
        '\n        ClearableFileInput.value_from_datadict never returns False if the field\n        is required.\n        '
        widget = ClearableFileInput()
        widget.is_required = True
        field = SimpleUploadedFile('something.txt', b'content')
        value = widget.value_from_datadict(data={'myfile-clear': True}, files={'myfile': field}, name='myfile')
        self.assertEqual(value, field)
        self.assertIs(widget.checked, True)

    def test_html_does_not_mask_exceptions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A ClearableFileInput should not mask exceptions produced while\n        checking that it has a value.\n        '

        class FailingURLFieldFile:

            @property
            def url(self):
                if False:
                    while True:
                        i = 10
                raise ValueError('Canary')

            def __str__(self):
                if False:
                    while True:
                        i = 10
                return 'value'
        with self.assertRaisesMessage(ValueError, 'Canary'):
            self.widget.render('myfile', FailingURLFieldFile())

    def test_url_as_property(self):
        if False:
            while True:
                i = 10

        class URLFieldFile:

            @property
            def url(self):
                if False:
                    return 10
                return 'https://www.python.org/'

            def __str__(self):
                if False:
                    print('Hello World!')
                return 'value'
        html = self.widget.render('myfile', URLFieldFile())
        self.assertInHTML('<a href="https://www.python.org/">value</a>', html)

    def test_return_false_if_url_does_not_exists(self):
        if False:
            return 10

        class NoURLFieldFile:

            def __str__(self):
                if False:
                    print('Hello World!')
                return 'value'
        html = self.widget.render('myfile', NoURLFieldFile())
        self.assertHTMLEqual(html, '<input name="myfile" type="file">')

    def test_use_required_attribute(self):
        if False:
            print('Hello World!')
        self.assertIs(self.widget.use_required_attribute(None), True)
        self.assertIs(self.widget.use_required_attribute('resume.txt'), False)

    def test_value_omitted_from_data(self):
        if False:
            return 10
        widget = ClearableFileInput()
        self.assertIs(widget.value_omitted_from_data({}, {}, 'field'), True)
        self.assertIs(widget.value_omitted_from_data({}, {'field': 'x'}, 'field'), False)
        self.assertIs(widget.value_omitted_from_data({'field-clear': 'y'}, {}, 'field'), False)

    def test_fieldset(self):
        if False:
            i = 10
            return i + 15

        class TestForm(Form):
            template_name = 'forms_tests/use_fieldset.html'
            field = FileField(widget=self.widget)
            with_file = FileField(widget=self.widget, initial=FakeFieldFile())
            clearable_file = FileField(widget=self.widget, initial=FakeFieldFile(), required=False)
        form = TestForm()
        self.assertIs(self.widget.use_fieldset, False)
        self.assertHTMLEqual('<div><label for="id_field">Field:</label><input id="id_field" name="field" type="file" required></div><div><label for="id_with_file">With file:</label>Currently: <a href="something">something</a><br>Change:<input type="file" name="with_file" id="id_with_file"></div><div><label for="id_clearable_file">Clearable file:</label>Currently: <a href="something">something</a><input type="checkbox" name="clearable_file-clear" id="clearable_file-clear_id"><label for="clearable_file-clear_id">Clear</label><br>Change:<input type="file" name="clearable_file" id="id_clearable_file"></div>', form.render())

    def test_multiple_error(self):
        if False:
            i = 10
            return i + 15
        msg = "ClearableFileInput doesn't support uploading multiple files."
        with self.assertRaisesMessage(ValueError, msg):
            ClearableFileInput(attrs={'multiple': True})