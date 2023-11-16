from django.template.defaultfilters import default_if_none
from django.test import SimpleTestCase

class FunctionTests(SimpleTestCase):

    def test_value(self):
        if False:
            print('Hello World!')
        self.assertEqual(default_if_none('val', 'default'), 'val')

    def test_none(self):
        if False:
            return 10
        self.assertEqual(default_if_none(None, 'default'), 'default')

    def test_empty_string(self):
        if False:
            while True:
                i = 10
        self.assertEqual(default_if_none('', 'default'), '')