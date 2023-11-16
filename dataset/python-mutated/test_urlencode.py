from django.template.defaultfilters import urlencode
from django.test import SimpleTestCase
from ..utils import setup

class UrlencodeTests(SimpleTestCase):

    @setup({'urlencode01': '{{ url|urlencode }}'})
    def test_urlencode01(self):
        if False:
            while True:
                i = 10
        output = self.engine.render_to_string('urlencode01', {'url': '/test&"/me?/'})
        self.assertEqual(output, '/test%26%22/me%3F/')

    @setup({'urlencode02': '/test/{{ urlbit|urlencode:"" }}/'})
    def test_urlencode02(self):
        if False:
            return 10
        output = self.engine.render_to_string('urlencode02', {'urlbit': 'escape/slash'})
        self.assertEqual(output, '/test/escape%2Fslash/')

class FunctionTests(SimpleTestCase):

    def test_urlencode(self):
        if False:
            return 10
        self.assertEqual(urlencode('françois & jill'), 'fran%C3%A7ois%20%26%20jill')

    def test_non_string_input(self):
        if False:
            print('Hello World!')
        self.assertEqual(urlencode(1), '1')