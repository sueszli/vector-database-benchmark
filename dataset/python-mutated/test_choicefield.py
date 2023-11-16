from django.core.exceptions import ValidationError
from django.db import models
from django.forms import ChoiceField, Form
from django.test import SimpleTestCase
from . import FormFieldAssertionsMixin

class ChoiceFieldTest(FormFieldAssertionsMixin, SimpleTestCase):

    def test_choicefield_1(self):
        if False:
            while True:
                i = 10
        f = ChoiceField(choices=[('1', 'One'), ('2', 'Two')])
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean('')
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean(None)
        self.assertEqual('1', f.clean(1))
        self.assertEqual('1', f.clean('1'))
        msg = "'Select a valid choice. 3 is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('3')

    def test_choicefield_2(self):
        if False:
            i = 10
            return i + 15
        f = ChoiceField(choices=[('1', 'One'), ('2', 'Two')], required=False)
        self.assertEqual('', f.clean(''))
        self.assertEqual('', f.clean(None))
        self.assertEqual('1', f.clean(1))
        self.assertEqual('1', f.clean('1'))
        msg = "'Select a valid choice. 3 is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('3')

    def test_choicefield_3(self):
        if False:
            for i in range(10):
                print('nop')
        f = ChoiceField(choices=[('J', 'John'), ('P', 'Paul')])
        self.assertEqual('J', f.clean('J'))
        msg = "'Select a valid choice. John is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('John')

    def test_choicefield_4(self):
        if False:
            for i in range(10):
                print('nop')
        f = ChoiceField(choices=[('Numbers', (('1', 'One'), ('2', 'Two'))), ('Letters', (('3', 'A'), ('4', 'B'))), ('5', 'Other')])
        self.assertEqual('1', f.clean(1))
        self.assertEqual('1', f.clean('1'))
        self.assertEqual('3', f.clean(3))
        self.assertEqual('3', f.clean('3'))
        self.assertEqual('5', f.clean(5))
        self.assertEqual('5', f.clean('5'))
        msg = "'Select a valid choice. 6 is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('6')

    def test_choicefield_choices_default(self):
        if False:
            i = 10
            return i + 15
        f = ChoiceField()
        self.assertEqual(f.choices, [])

    def test_choicefield_callable(self):
        if False:
            i = 10
            return i + 15

        def choices():
            if False:
                while True:
                    i = 10
            return [('J', 'John'), ('P', 'Paul')]
        f = ChoiceField(choices=choices)
        self.assertEqual('J', f.clean('J'))

    def test_choicefield_callable_mapping(self):
        if False:
            for i in range(10):
                print('nop')

        def choices():
            if False:
                for i in range(10):
                    print('nop')
            return {'J': 'John', 'P': 'Paul'}
        f = ChoiceField(choices=choices)
        self.assertEqual('J', f.clean('J'))

    def test_choicefield_callable_grouped_mapping(self):
        if False:
            return 10

        def choices():
            if False:
                print('Hello World!')
            return {'Numbers': {'1': 'One', '2': 'Two'}, 'Letters': {'3': 'A', '4': 'B'}}
        f = ChoiceField(choices=choices)
        for i in ('1', '2', '3', '4'):
            with self.subTest(i):
                self.assertEqual(i, f.clean(i))

    def test_choicefield_mapping(self):
        if False:
            for i in range(10):
                print('nop')
        f = ChoiceField(choices={'J': 'John', 'P': 'Paul'})
        self.assertEqual('J', f.clean('J'))

    def test_choicefield_grouped_mapping(self):
        if False:
            print('Hello World!')
        f = ChoiceField(choices={'Numbers': (('1', 'One'), ('2', 'Two')), 'Letters': (('3', 'A'), ('4', 'B'))})
        for i in ('1', '2', '3', '4'):
            with self.subTest(i):
                self.assertEqual(i, f.clean(i))

    def test_choicefield_grouped_mapping_inner_dict(self):
        if False:
            while True:
                i = 10
        f = ChoiceField(choices={'Numbers': {'1': 'One', '2': 'Two'}, 'Letters': {'3': 'A', '4': 'B'}})
        for i in ('1', '2', '3', '4'):
            with self.subTest(i):
                self.assertEqual(i, f.clean(i))

    def test_choicefield_callable_may_evaluate_to_different_values(self):
        if False:
            while True:
                i = 10
        choices = []

        def choices_as_callable():
            if False:
                while True:
                    i = 10
            return choices

        class ChoiceFieldForm(Form):
            choicefield = ChoiceField(choices=choices_as_callable)
        choices = [('J', 'John')]
        form = ChoiceFieldForm()
        self.assertEqual(choices, list(form.fields['choicefield'].choices))
        self.assertEqual(choices, list(form.fields['choicefield'].widget.choices))
        choices = [('P', 'Paul')]
        form = ChoiceFieldForm()
        self.assertEqual(choices, list(form.fields['choicefield'].choices))
        self.assertEqual(choices, list(form.fields['choicefield'].widget.choices))

    def test_choicefield_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        f = ChoiceField(choices=[('J', 'John'), ('P', 'Paul')], disabled=True)
        self.assertWidgetRendersTo(f, '<select id="id_f" name="f" disabled><option value="J">John</option><option value="P">Paul</option></select>')

    def test_choicefield_enumeration(self):
        if False:
            print('Hello World!')

        class FirstNames(models.TextChoices):
            JOHN = ('J', 'John')
            PAUL = ('P', 'Paul')
        f = ChoiceField(choices=FirstNames)
        self.assertEqual(f.choices, FirstNames.choices)
        self.assertEqual(f.clean('J'), 'J')
        msg = "'Select a valid choice. 3 is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('3')