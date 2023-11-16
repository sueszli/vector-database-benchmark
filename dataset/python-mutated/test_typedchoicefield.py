import decimal
from django.core.exceptions import ValidationError
from django.forms import TypedChoiceField
from django.test import SimpleTestCase

class TypedChoiceFieldTest(SimpleTestCase):

    def test_typedchoicefield_1(self):
        if False:
            i = 10
            return i + 15
        f = TypedChoiceField(choices=[(1, '+1'), (-1, '-1')], coerce=int)
        self.assertEqual(1, f.clean('1'))
        msg = "'Select a valid choice. 2 is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('2')

    def test_typedchoicefield_2(self):
        if False:
            return 10
        f = TypedChoiceField(choices=[(1, '+1'), (-1, '-1')], coerce=float)
        self.assertEqual(1.0, f.clean('1'))

    def test_typedchoicefield_3(self):
        if False:
            for i in range(10):
                print('nop')
        f = TypedChoiceField(choices=[(1, '+1'), (-1, '-1')], coerce=bool)
        self.assertTrue(f.clean('-1'))

    def test_typedchoicefield_4(self):
        if False:
            print('Hello World!')
        f = TypedChoiceField(choices=[('A', 'A'), ('B', 'B')], coerce=int)
        msg = "'Select a valid choice. B is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('B')
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean('')

    def test_typedchoicefield_5(self):
        if False:
            while True:
                i = 10
        f = TypedChoiceField(choices=[(1, '+1'), (-1, '-1')], coerce=int, required=False)
        self.assertEqual('', f.clean(''))

    def test_typedchoicefield_6(self):
        if False:
            return 10
        f = TypedChoiceField(choices=[(1, '+1'), (-1, '-1')], coerce=int, required=False, empty_value=None)
        self.assertIsNone(f.clean(''))

    def test_typedchoicefield_has_changed(self):
        if False:
            i = 10
            return i + 15
        f = TypedChoiceField(choices=[(1, '+1'), (-1, '-1')], coerce=int, required=True)
        self.assertFalse(f.has_changed(None, ''))
        self.assertFalse(f.has_changed(1, '1'))
        self.assertFalse(f.has_changed('1', '1'))
        f = TypedChoiceField(choices=[('', '---------'), ('a', 'a'), ('b', 'b')], coerce=str, required=False, initial=None, empty_value=None)
        self.assertFalse(f.has_changed(None, ''))
        self.assertTrue(f.has_changed('', 'a'))
        self.assertFalse(f.has_changed('a', 'a'))

    def test_typedchoicefield_special_coerce(self):
        if False:
            return 10
        '\n        A coerce function which results in a value not present in choices\n        should raise an appropriate error (#21397).\n        '

        def coerce_func(val):
            if False:
                return 10
            return decimal.Decimal('1.%s' % val)
        f = TypedChoiceField(choices=[(1, '1'), (2, '2')], coerce=coerce_func, required=True)
        self.assertEqual(decimal.Decimal('1.2'), f.clean('2'))
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean('')
        msg = "'Select a valid choice. 3 is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('3')