from django.template.defaultfilters import yesno
from django.test import SimpleTestCase
from ..utils import setup

class YesNoTests(SimpleTestCase):

    @setup({'t': '{{ var|yesno:"yup,nup,mup" }} {{ var|yesno }}'})
    def test_true(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.engine.render_to_string('t', {'var': True})
        self.assertEqual(output, 'yup yes')

class FunctionTests(SimpleTestCase):

    def test_true(self):
        if False:
            while True:
                i = 10
        self.assertEqual(yesno(True), 'yes')

    def test_false(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(yesno(False), 'no')

    def test_none(self):
        if False:
            return 10
        self.assertEqual(yesno(None), 'maybe')

    def test_true_arguments(self):
        if False:
            while True:
                i = 10
        self.assertEqual(yesno(True, 'certainly,get out of town,perhaps'), 'certainly')

    def test_false_arguments(self):
        if False:
            return 10
        self.assertEqual(yesno(False, 'certainly,get out of town,perhaps'), 'get out of town')

    def test_none_two_arguments(self):
        if False:
            print('Hello World!')
        self.assertEqual(yesno(None, 'certainly,get out of town'), 'get out of town')

    def test_none_three_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(yesno(None, 'certainly,get out of town,perhaps'), 'perhaps')

    def test_invalid_value(self):
        if False:
            return 10
        self.assertIs(yesno(True, 'yes'), True)
        self.assertIs(yesno(False, 'yes'), False)
        self.assertIsNone(yesno(None, 'yes'))