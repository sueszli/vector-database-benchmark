from django.template.defaultfilters import divisibleby
from django.test import SimpleTestCase

class FunctionTests(SimpleTestCase):

    def test_true(self):
        if False:
            while True:
                i = 10
        self.assertIs(divisibleby(4, 2), True)

    def test_false(self):
        if False:
            print('Hello World!')
        self.assertIs(divisibleby(4, 3), False)