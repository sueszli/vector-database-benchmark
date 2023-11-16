import sys
from datetime import date, datetime
from django.core.exceptions import ValidationError
from django.forms import DateField, Form, HiddenInput, SelectDateWidget
from django.test import SimpleTestCase
from django.utils import translation

class GetDate(Form):
    mydate = DateField(widget=SelectDateWidget)

class DateFieldTest(SimpleTestCase):

    def test_form_field(self):
        if False:
            print('Hello World!')
        a = GetDate({'mydate_month': '4', 'mydate_day': '1', 'mydate_year': '2008'})
        self.assertTrue(a.is_valid())
        self.assertEqual(a.cleaned_data['mydate'], date(2008, 4, 1))
        self.assertHTMLEqual(a['mydate'].as_hidden(), '<input type="hidden" name="mydate" value="2008-04-01" id="id_mydate">')
        b = GetDate({'mydate': '2008-4-1'})
        self.assertTrue(b.is_valid())
        self.assertEqual(b.cleaned_data['mydate'], date(2008, 4, 1))
        c = GetDate({'mydate_month': '2', 'mydate_day': '31', 'mydate_year': '2010'})
        self.assertFalse(c.is_valid())
        self.assertEqual(c.errors, {'mydate': ['Enter a valid date.']})
        d = GetDate({'mydate_month': '1', 'mydate_day': '1', 'mydate_year': '2010'})
        self.assertIn('<label for="id_mydate_month">', d.as_p())
        e = GetDate({'mydate_month': str(sys.maxsize + 1), 'mydate_day': '31', 'mydate_year': '2010'})
        self.assertIs(e.is_valid(), False)
        self.assertEqual(e.errors, {'mydate': ['Enter a valid date.']})

    @translation.override('nl')
    def test_l10n_date_changed(self):
        if False:
            i = 10
            return i + 15
        '\n        DateField.has_changed() with SelectDateWidget works with a localized\n        date format (#17165).\n        '
        b = GetDate({'mydate_year': '2008', 'mydate_month': '4', 'mydate_day': '1'}, initial={'mydate': date(2008, 4, 1)})
        self.assertFalse(b.has_changed())
        b = GetDate({'mydate_year': '2008', 'mydate_month': '4', 'mydate_day': '2'}, initial={'mydate': date(2008, 4, 1)})
        self.assertTrue(b.has_changed())

        class GetDateShowHiddenInitial(Form):
            mydate = DateField(widget=SelectDateWidget, show_hidden_initial=True)
        b = GetDateShowHiddenInitial({'mydate_year': '2008', 'mydate_month': '4', 'mydate_day': '1', 'initial-mydate': HiddenInput().format_value(date(2008, 4, 1))}, initial={'mydate': date(2008, 4, 1)})
        self.assertFalse(b.has_changed())
        b = GetDateShowHiddenInitial({'mydate_year': '2008', 'mydate_month': '4', 'mydate_day': '22', 'initial-mydate': HiddenInput().format_value(date(2008, 4, 1))}, initial={'mydate': date(2008, 4, 1)})
        self.assertTrue(b.has_changed())
        b = GetDateShowHiddenInitial({'mydate_year': '2008', 'mydate_month': '4', 'mydate_day': '22', 'initial-mydate': HiddenInput().format_value(date(2008, 4, 1))}, initial={'mydate': date(2008, 4, 22)})
        self.assertTrue(b.has_changed())
        b = GetDateShowHiddenInitial({'mydate_year': '2008', 'mydate_month': '4', 'mydate_day': '22', 'initial-mydate': HiddenInput().format_value(date(2008, 4, 22))}, initial={'mydate': date(2008, 4, 1)})
        self.assertFalse(b.has_changed())

    @translation.override('nl')
    def test_l10n_invalid_date_in(self):
        if False:
            i = 10
            return i + 15
        a = GetDate({'mydate_month': '2', 'mydate_day': '31', 'mydate_year': '2010'})
        self.assertFalse(a.is_valid())
        self.assertEqual(a.errors, {'mydate': ['Voer een geldige datum in.']})

    @translation.override('nl')
    def test_form_label_association(self):
        if False:
            print('Hello World!')
        a = GetDate({'mydate_month': '1', 'mydate_day': '1', 'mydate_year': '2010'})
        self.assertIn('<label for="id_mydate_day">', a.as_p())

    def test_datefield_1(self):
        if False:
            return 10
        f = DateField()
        self.assertEqual(date(2006, 10, 25), f.clean(date(2006, 10, 25)))
        self.assertEqual(date(2006, 10, 25), f.clean(datetime(2006, 10, 25, 14, 30)))
        self.assertEqual(date(2006, 10, 25), f.clean(datetime(2006, 10, 25, 14, 30, 59)))
        self.assertEqual(date(2006, 10, 25), f.clean(datetime(2006, 10, 25, 14, 30, 59, 200)))
        self.assertEqual(date(2006, 10, 25), f.clean('2006-10-25'))
        self.assertEqual(date(2006, 10, 25), f.clean('10/25/2006'))
        self.assertEqual(date(2006, 10, 25), f.clean('10/25/06'))
        self.assertEqual(date(2006, 10, 25), f.clean('Oct 25 2006'))
        self.assertEqual(date(2006, 10, 25), f.clean('October 25 2006'))
        self.assertEqual(date(2006, 10, 25), f.clean('October 25, 2006'))
        self.assertEqual(date(2006, 10, 25), f.clean('25 October 2006'))
        self.assertEqual(date(2006, 10, 25), f.clean('25 October, 2006'))
        with self.assertRaisesMessage(ValidationError, "'Enter a valid date.'"):
            f.clean('2006-4-31')
        with self.assertRaisesMessage(ValidationError, "'Enter a valid date.'"):
            f.clean('200a-10-25')
        with self.assertRaisesMessage(ValidationError, "'Enter a valid date.'"):
            f.clean('25/10/06')
        with self.assertRaisesMessage(ValidationError, "'Enter a valid date.'"):
            f.clean('0-0-0')
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean(None)

    def test_datefield_2(self):
        if False:
            while True:
                i = 10
        f = DateField(required=False)
        self.assertIsNone(f.clean(None))
        self.assertEqual('None', repr(f.clean(None)))
        self.assertIsNone(f.clean(''))
        self.assertEqual('None', repr(f.clean('')))

    def test_datefield_3(self):
        if False:
            i = 10
            return i + 15
        f = DateField(input_formats=['%Y %m %d'])
        self.assertEqual(date(2006, 10, 25), f.clean(date(2006, 10, 25)))
        self.assertEqual(date(2006, 10, 25), f.clean(datetime(2006, 10, 25, 14, 30)))
        self.assertEqual(date(2006, 10, 25), f.clean('2006 10 25'))
        with self.assertRaisesMessage(ValidationError, "'Enter a valid date.'"):
            f.clean('2006-10-25')
        with self.assertRaisesMessage(ValidationError, "'Enter a valid date.'"):
            f.clean('10/25/2006')
        with self.assertRaisesMessage(ValidationError, "'Enter a valid date.'"):
            f.clean('10/25/06')

    def test_datefield_4(self):
        if False:
            return 10
        f = DateField()
        self.assertEqual(date(2006, 10, 25), f.clean(' 10/25/2006 '))
        self.assertEqual(date(2006, 10, 25), f.clean(' 10/25/06 '))
        self.assertEqual(date(2006, 10, 25), f.clean(' Oct 25   2006 '))
        self.assertEqual(date(2006, 10, 25), f.clean(' October  25 2006 '))
        self.assertEqual(date(2006, 10, 25), f.clean(' October 25, 2006 '))
        self.assertEqual(date(2006, 10, 25), f.clean(' 25 October 2006 '))
        with self.assertRaisesMessage(ValidationError, "'Enter a valid date.'"):
            f.clean('   ')

    def test_datefield_5(self):
        if False:
            while True:
                i = 10
        f = DateField()
        with self.assertRaisesMessage(ValidationError, "'Enter a valid date.'"):
            f.clean('a\x00b')

    def test_datefield_changed(self):
        if False:
            return 10
        format = '%d/%m/%Y'
        f = DateField(input_formats=[format])
        d = date(2007, 9, 17)
        self.assertFalse(f.has_changed(d, '17/09/2007'))

    def test_datefield_strptime(self):
        if False:
            while True:
                i = 10
        "field.strptime() doesn't raise a UnicodeEncodeError (#16123)"
        f = DateField()
        try:
            f.strptime('31 мая 2011', '%d-%b-%y')
        except Exception as e:
            self.assertEqual(e.__class__, ValueError)