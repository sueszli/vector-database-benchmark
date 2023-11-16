from django.template.defaultfilters import get_digit
from django.test import SimpleTestCase

class FunctionTests(SimpleTestCase):

    def test_values(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(get_digit(123, 1), 3)
        self.assertEqual(get_digit(123, 2), 2)
        self.assertEqual(get_digit(123, 3), 1)
        self.assertEqual(get_digit(123, 4), 0)
        self.assertEqual(get_digit(123, 0), 123)

    def test_string(self):
        if False:
            print('Hello World!')
        self.assertEqual(get_digit('xyz', 0), 'xyz')