from datetime import date, datetime, time
from django import forms
from django.core.exceptions import ValidationError
from django.test import SimpleTestCase, override_settings
from django.utils import translation
from django.utils.translation import activate, deactivate

class LocalizedTimeTests(SimpleTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        activate('nl')

    def tearDown(self):
        if False:
            print('Hello World!')
        deactivate()

    def test_timeField(self):
        if False:
            i = 10
            return i + 15
        'TimeFields can parse dates in the default format'
        f = forms.TimeField()
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM')
        result = f.clean('13:30:05')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:05')
        result = f.clean('13:30')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:00')
        result = f.clean('13:30:05.000155')
        self.assertEqual(result, time(13, 30, 5, 155))

    def test_localized_timeField(self):
        if False:
            for i in range(10):
                print('nop')
        'Localized TimeFields act as unlocalized widgets'
        f = forms.TimeField(localize=True)
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM')
        result = f.clean('13:30:05')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:05')
        result = f.clean('13:30')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:00')

    def test_timeField_with_inputformat(self):
        if False:
            i = 10
            return i + 15
        'TimeFields with manually specified input formats can accept those formats'
        f = forms.TimeField(input_formats=['%H.%M.%S', '%H.%M'])
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM')
        with self.assertRaises(ValidationError):
            f.clean('13:30:05')
        result = f.clean('13.30.05')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:05')
        result = f.clean('13.30')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:00')

    def test_localized_timeField_with_inputformat(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Localized TimeFields with manually specified input formats can accept\n        those formats.\n        '
        f = forms.TimeField(input_formats=['%H.%M.%S', '%H.%M'], localize=True)
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM')
        with self.assertRaises(ValidationError):
            f.clean('13:30:05')
        result = f.clean('13.30.05')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:05')
        result = f.clean('13.30')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:00')

@translation.override(None)
@override_settings(TIME_INPUT_FORMATS=['%I:%M:%S %p', '%I:%M %p'])
class CustomTimeInputFormatsTests(SimpleTestCase):

    def test_timeField(self):
        if False:
            while True:
                i = 10
        'TimeFields can parse dates in the default format'
        f = forms.TimeField()
        with self.assertRaises(ValidationError):
            f.clean('13:30:05')
        result = f.clean('1:30:05 PM')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:05 PM')
        result = f.clean('1:30 PM')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:00 PM')

    def test_localized_timeField(self):
        if False:
            print('Hello World!')
        'Localized TimeFields act as unlocalized widgets'
        f = forms.TimeField(localize=True)
        with self.assertRaises(ValidationError):
            f.clean('13:30:05')
        result = f.clean('1:30:05 PM')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:05 PM')
        result = f.clean('01:30 PM')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:00 PM')

    def test_timeField_with_inputformat(self):
        if False:
            i = 10
            return i + 15
        'TimeFields with manually specified input formats can accept those formats'
        f = forms.TimeField(input_formats=['%H.%M.%S', '%H.%M'])
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM')
        with self.assertRaises(ValidationError):
            f.clean('13:30:05')
        result = f.clean('13.30.05')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:05 PM')
        result = f.clean('13.30')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:00 PM')

    def test_localized_timeField_with_inputformat(self):
        if False:
            while True:
                i = 10
        '\n        Localized TimeFields with manually specified input formats can accept\n        those formats.\n        '
        f = forms.TimeField(input_formats=['%H.%M.%S', '%H.%M'], localize=True)
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM')
        with self.assertRaises(ValidationError):
            f.clean('13:30:05')
        result = f.clean('13.30.05')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:05 PM')
        result = f.clean('13.30')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:00 PM')

class SimpleTimeFormatTests(SimpleTestCase):

    def test_timeField(self):
        if False:
            while True:
                i = 10
        'TimeFields can parse dates in the default format'
        f = forms.TimeField()
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM')
        result = f.clean('13:30:05')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:05')
        result = f.clean('13:30')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:00')

    def test_localized_timeField(self):
        if False:
            return 10
        'Localized TimeFields in a non-localized environment act as unlocalized widgets'
        f = forms.TimeField()
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM')
        result = f.clean('13:30:05')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:05')
        result = f.clean('13:30')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:00')

    def test_timeField_with_inputformat(self):
        if False:
            return 10
        'TimeFields with manually specified input formats can accept those formats'
        f = forms.TimeField(input_formats=['%I:%M:%S %p', '%I:%M %p'])
        with self.assertRaises(ValidationError):
            f.clean('13:30:05')
        result = f.clean('1:30:05 PM')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:05')
        result = f.clean('1:30 PM')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:00')

    def test_localized_timeField_with_inputformat(self):
        if False:
            return 10
        '\n        Localized TimeFields with manually specified input formats can accept\n        those formats.\n        '
        f = forms.TimeField(input_formats=['%I:%M:%S %p', '%I:%M %p'], localize=True)
        with self.assertRaises(ValidationError):
            f.clean('13:30:05')
        result = f.clean('1:30:05 PM')
        self.assertEqual(result, time(13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:05')
        result = f.clean('1:30 PM')
        self.assertEqual(result, time(13, 30, 0))
        text = f.widget.format_value(result)
        self.assertEqual(text, '13:30:00')

class LocalizedDateTests(SimpleTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        activate('de')

    def tearDown(self):
        if False:
            print('Hello World!')
        deactivate()

    def test_dateField(self):
        if False:
            for i in range(10):
                print('nop')
        'DateFields can parse dates in the default format'
        f = forms.DateField()
        with self.assertRaises(ValidationError):
            f.clean('21/12/2010')
        self.assertEqual(f.clean('2010-12-21'), date(2010, 12, 21))
        result = f.clean('21.12.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')
        result = f.clean('21.12.10')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')

    def test_localized_dateField(self):
        if False:
            while True:
                i = 10
        'Localized DateFields act as unlocalized widgets'
        f = forms.DateField(localize=True)
        with self.assertRaises(ValidationError):
            f.clean('21/12/2010')
        result = f.clean('21.12.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')
        result = f.clean('21.12.10')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')

    def test_dateField_with_inputformat(self):
        if False:
            while True:
                i = 10
        'DateFields with manually specified input formats can accept those formats'
        f = forms.DateField(input_formats=['%m.%d.%Y', '%m-%d-%Y'])
        with self.assertRaises(ValidationError):
            f.clean('2010-12-21')
        with self.assertRaises(ValidationError):
            f.clean('21/12/2010')
        with self.assertRaises(ValidationError):
            f.clean('21.12.2010')
        result = f.clean('12.21.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')
        result = f.clean('12-21-2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')

    def test_localized_dateField_with_inputformat(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Localized DateFields with manually specified input formats can accept\n        those formats.\n        '
        f = forms.DateField(input_formats=['%m.%d.%Y', '%m-%d-%Y'], localize=True)
        with self.assertRaises(ValidationError):
            f.clean('2010-12-21')
        with self.assertRaises(ValidationError):
            f.clean('21/12/2010')
        with self.assertRaises(ValidationError):
            f.clean('21.12.2010')
        result = f.clean('12.21.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')
        result = f.clean('12-21-2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')

@translation.override(None)
@override_settings(DATE_INPUT_FORMATS=['%d.%m.%Y', '%d-%m-%Y'])
class CustomDateInputFormatsTests(SimpleTestCase):

    def test_dateField(self):
        if False:
            i = 10
            return i + 15
        'DateFields can parse dates in the default format'
        f = forms.DateField()
        with self.assertRaises(ValidationError):
            f.clean('2010-12-21')
        result = f.clean('21.12.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')
        result = f.clean('21-12-2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')

    def test_localized_dateField(self):
        if False:
            print('Hello World!')
        'Localized DateFields act as unlocalized widgets'
        f = forms.DateField(localize=True)
        with self.assertRaises(ValidationError):
            f.clean('2010-12-21')
        result = f.clean('21.12.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')
        result = f.clean('21-12-2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')

    def test_dateField_with_inputformat(self):
        if False:
            return 10
        'DateFields with manually specified input formats can accept those formats'
        f = forms.DateField(input_formats=['%m.%d.%Y', '%m-%d-%Y'])
        with self.assertRaises(ValidationError):
            f.clean('21.12.2010')
        with self.assertRaises(ValidationError):
            f.clean('2010-12-21')
        result = f.clean('12.21.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')
        result = f.clean('12-21-2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')

    def test_localized_dateField_with_inputformat(self):
        if False:
            print('Hello World!')
        '\n        Localized DateFields with manually specified input formats can accept\n        those formats.\n        '
        f = forms.DateField(input_formats=['%m.%d.%Y', '%m-%d-%Y'], localize=True)
        with self.assertRaises(ValidationError):
            f.clean('21.12.2010')
        with self.assertRaises(ValidationError):
            f.clean('2010-12-21')
        result = f.clean('12.21.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')
        result = f.clean('12-21-2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010')

class SimpleDateFormatTests(SimpleTestCase):

    def test_dateField(self):
        if False:
            print('Hello World!')
        'DateFields can parse dates in the default format'
        f = forms.DateField()
        with self.assertRaises(ValidationError):
            f.clean('21.12.2010')
        result = f.clean('2010-12-21')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21')
        result = f.clean('12/21/2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21')

    def test_localized_dateField(self):
        if False:
            return 10
        'Localized DateFields in a non-localized environment act as unlocalized widgets'
        f = forms.DateField()
        with self.assertRaises(ValidationError):
            f.clean('21.12.2010')
        result = f.clean('2010-12-21')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21')
        result = f.clean('12/21/2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21')

    def test_dateField_with_inputformat(self):
        if False:
            i = 10
            return i + 15
        'DateFields with manually specified input formats can accept those formats'
        f = forms.DateField(input_formats=['%d.%m.%Y', '%d-%m-%Y'])
        with self.assertRaises(ValidationError):
            f.clean('2010-12-21')
        result = f.clean('21.12.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21')
        result = f.clean('21-12-2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21')

    def test_localized_dateField_with_inputformat(self):
        if False:
            print('Hello World!')
        '\n        Localized DateFields with manually specified input formats can accept\n        those formats.\n        '
        f = forms.DateField(input_formats=['%d.%m.%Y', '%d-%m-%Y'], localize=True)
        with self.assertRaises(ValidationError):
            f.clean('2010-12-21')
        result = f.clean('21.12.2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21')
        result = f.clean('21-12-2010')
        self.assertEqual(result, date(2010, 12, 21))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21')

class LocalizedDateTimeTests(SimpleTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        activate('de')

    def tearDown(self):
        if False:
            while True:
                i = 10
        deactivate()

    def test_dateTimeField(self):
        if False:
            while True:
                i = 10
        'DateTimeFields can parse dates in the default format'
        f = forms.DateTimeField()
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM 21/12/2010')
        self.assertEqual(f.clean('2010-12-21 13:30:05'), datetime(2010, 12, 21, 13, 30, 5))
        result = f.clean('21.12.2010 13:30:05')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010 13:30:05')
        result = f.clean('21.12.2010 13:30')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010 13:30:00')

    def test_localized_dateTimeField(self):
        if False:
            return 10
        'Localized DateTimeFields act as unlocalized widgets'
        f = forms.DateTimeField(localize=True)
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM 21/12/2010')
        result = f.clean('21.12.2010 13:30:05')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010 13:30:05')
        result = f.clean('21.12.2010 13:30')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010 13:30:00')

    def test_dateTimeField_with_inputformat(self):
        if False:
            i = 10
            return i + 15
        'DateTimeFields with manually specified input formats can accept those formats'
        f = forms.DateTimeField(input_formats=['%H.%M.%S %m.%d.%Y', '%H.%M %m-%d-%Y'])
        with self.assertRaises(ValidationError):
            f.clean('2010-12-21 13:30:05 13:30:05')
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM 21/12/2010')
        with self.assertRaises(ValidationError):
            f.clean('13:30:05 21.12.2010')
        result = f.clean('13.30.05 12.21.2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010 13:30:05')
        result = f.clean('13.30 12-21-2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010 13:30:00')

    def test_localized_dateTimeField_with_inputformat(self):
        if False:
            print('Hello World!')
        '\n        Localized DateTimeFields with manually specified input formats can\n        accept those formats.\n        '
        f = forms.DateTimeField(input_formats=['%H.%M.%S %m.%d.%Y', '%H.%M %m-%d-%Y'], localize=True)
        with self.assertRaises(ValidationError):
            f.clean('2010/12/21 13:30:05')
        with self.assertRaises(ValidationError):
            f.clean('1:30:05 PM 21/12/2010')
        with self.assertRaises(ValidationError):
            f.clean('13:30:05 21.12.2010')
        result = f.clean('13.30.05 12.21.2010')
        self.assertEqual(datetime(2010, 12, 21, 13, 30, 5), result)
        self.assertEqual(f.clean('2010-12-21 13:30:05'), datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010 13:30:05')
        result = f.clean('13.30 12-21-2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '21.12.2010 13:30:00')

@translation.override(None)
@override_settings(DATETIME_INPUT_FORMATS=['%I:%M:%S %p %d/%m/%Y', '%I:%M %p %d-%m-%Y'])
class CustomDateTimeInputFormatsTests(SimpleTestCase):

    def test_dateTimeField(self):
        if False:
            i = 10
            return i + 15
        'DateTimeFields can parse dates in the default format'
        f = forms.DateTimeField()
        with self.assertRaises(ValidationError):
            f.clean('2010/12/21 13:30:05')
        result = f.clean('1:30:05 PM 21/12/2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:05 PM 21/12/2010')
        result = f.clean('1:30 PM 21-12-2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:00 PM 21/12/2010')

    def test_localized_dateTimeField(self):
        if False:
            for i in range(10):
                print('nop')
        'Localized DateTimeFields act as unlocalized widgets'
        f = forms.DateTimeField(localize=True)
        with self.assertRaises(ValidationError):
            f.clean('2010/12/21 13:30:05')
        result = f.clean('1:30:05 PM 21/12/2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:05 PM 21/12/2010')
        result = f.clean('1:30 PM 21-12-2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:00 PM 21/12/2010')

    def test_dateTimeField_with_inputformat(self):
        if False:
            return 10
        'DateTimeFields with manually specified input formats can accept those formats'
        f = forms.DateTimeField(input_formats=['%m.%d.%Y %H:%M:%S', '%m-%d-%Y %H:%M'])
        with self.assertRaises(ValidationError):
            f.clean('13:30:05 21.12.2010')
        with self.assertRaises(ValidationError):
            f.clean('2010/12/21 13:30:05')
        result = f.clean('12.21.2010 13:30:05')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:05 PM 21/12/2010')
        result = f.clean('12-21-2010 13:30')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:00 PM 21/12/2010')

    def test_localized_dateTimeField_with_inputformat(self):
        if False:
            print('Hello World!')
        '\n        Localized DateTimeFields with manually specified input formats can\n        accept those formats.\n        '
        f = forms.DateTimeField(input_formats=['%m.%d.%Y %H:%M:%S', '%m-%d-%Y %H:%M'], localize=True)
        with self.assertRaises(ValidationError):
            f.clean('13:30:05 21.12.2010')
        with self.assertRaises(ValidationError):
            f.clean('2010/12/21 13:30:05')
        result = f.clean('12.21.2010 13:30:05')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:05 PM 21/12/2010')
        result = f.clean('12-21-2010 13:30')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '01:30:00 PM 21/12/2010')

class SimpleDateTimeFormatTests(SimpleTestCase):

    def test_dateTimeField(self):
        if False:
            for i in range(10):
                print('nop')
        'DateTimeFields can parse dates in the default format'
        f = forms.DateTimeField()
        with self.assertRaises(ValidationError):
            f.clean('13:30:05 21.12.2010')
        result = f.clean('2010-12-21 13:30:05')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21 13:30:05')
        result = f.clean('12/21/2010 13:30:05')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21 13:30:05')

    def test_localized_dateTimeField(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Localized DateTimeFields in a non-localized environment act as\n        unlocalized widgets.\n        '
        f = forms.DateTimeField()
        with self.assertRaises(ValidationError):
            f.clean('13:30:05 21.12.2010')
        result = f.clean('2010-12-21 13:30:05')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21 13:30:05')
        result = f.clean('12/21/2010 13:30:05')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21 13:30:05')

    def test_dateTimeField_with_inputformat(self):
        if False:
            return 10
        'DateTimeFields with manually specified input formats can accept those formats'
        f = forms.DateTimeField(input_formats=['%I:%M:%S %p %d.%m.%Y', '%I:%M %p %d-%m-%Y'])
        with self.assertRaises(ValidationError):
            f.clean('2010/12/21 13:30:05')
        result = f.clean('1:30:05 PM 21.12.2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21 13:30:05')
        result = f.clean('1:30 PM 21-12-2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21 13:30:00')

    def test_localized_dateTimeField_with_inputformat(self):
        if False:
            while True:
                i = 10
        '\n        Localized DateTimeFields with manually specified input formats can\n        accept those formats.\n        '
        f = forms.DateTimeField(input_formats=['%I:%M:%S %p %d.%m.%Y', '%I:%M %p %d-%m-%Y'], localize=True)
        with self.assertRaises(ValidationError):
            f.clean('2010/12/21 13:30:05')
        result = f.clean('1:30:05 PM 21.12.2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30, 5))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21 13:30:05')
        result = f.clean('1:30 PM 21-12-2010')
        self.assertEqual(result, datetime(2010, 12, 21, 13, 30))
        text = f.widget.format_value(result)
        self.assertEqual(text, '2010-12-21 13:30:00')