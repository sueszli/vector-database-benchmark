from django.core.exceptions import ValidationError
from django.forms import URLField
from django.test import SimpleTestCase, ignore_warnings
from django.utils.deprecation import RemovedInDjango60Warning
from . import FormFieldAssertionsMixin

@ignore_warnings(category=RemovedInDjango60Warning)
class URLFieldTest(FormFieldAssertionsMixin, SimpleTestCase):

    def test_urlfield_widget(self):
        if False:
            return 10
        f = URLField()
        self.assertWidgetRendersTo(f, '<input type="url" name="f" id="id_f" required>')

    def test_urlfield_widget_max_min_length(self):
        if False:
            while True:
                i = 10
        f = URLField(min_length=15, max_length=20)
        self.assertEqual('http://example.com', f.clean('http://example.com'))
        self.assertWidgetRendersTo(f, '<input id="id_f" type="url" name="f" maxlength="20" minlength="15" required>')
        msg = "'Ensure this value has at least 15 characters (it has 12).'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('http://f.com')
        msg = "'Ensure this value has at most 20 characters (it has 37).'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('http://abcdefghijklmnopqrstuvwxyz.com')

    def test_urlfield_clean(self):
        if False:
            while True:
                i = 10
        f = URLField(required=False, assume_scheme='https')
        tests = [('http://localhost', 'http://localhost'), ('http://example.com', 'http://example.com'), ('http://example.com/test', 'http://example.com/test'), ('http://example.com.', 'http://example.com.'), ('http://www.example.com', 'http://www.example.com'), ('http://www.example.com:8000/test', 'http://www.example.com:8000/test'), ('http://example.com?some_param=some_value', 'http://example.com?some_param=some_value'), ('valid-with-hyphens.com', 'https://valid-with-hyphens.com'), ('subdomain.domain.com', 'https://subdomain.domain.com'), ('http://200.8.9.10', 'http://200.8.9.10'), ('http://200.8.9.10:8000/test', 'http://200.8.9.10:8000/test'), ('http://valid-----hyphens.com', 'http://valid-----hyphens.com'), ('http://some.idn.xyzäöüßabc.domain.com:123/blah', 'http://some.idn.xyzäöüßabc.domain.com:123/blah'), ('www.example.com/s/http://code.djangoproject.com/ticket/13804', 'https://www.example.com/s/http://code.djangoproject.com/ticket/13804'), ('http://example.com/     ', 'http://example.com/'), ('http://עברית.idn.icann.org/', 'http://עברית.idn.icann.org/'), ('http://sãopaulo.com/', 'http://sãopaulo.com/'), ('http://sãopaulo.com.br/', 'http://sãopaulo.com.br/'), ('http://пример.испытание/', 'http://пример.испытание/'), ('http://مثال.إختبار/', 'http://مثال.إختبار/'), ('http://例子.测试/', 'http://例子.测试/'), ('http://例子.測試/', 'http://例子.測試/'), ('http://उदाहरण.परीक्षा/', 'http://उदाहरण.परीक्षा/'), ('http://例え.テスト/', 'http://例え.テスト/'), ('http://مثال.آزمایشی/', 'http://مثال.آزمایشی/'), ('http://실례.테스트/', 'http://실례.테스트/'), ('http://العربية.idn.icann.org/', 'http://العربية.idn.icann.org/'), ('http://[12:34::3a53]/', 'http://[12:34::3a53]/'), ('http://[a34:9238::]:8080/', 'http://[a34:9238::]:8080/')]
        for (url, expected) in tests:
            with self.subTest(url=url):
                self.assertEqual(f.clean(url), expected)

    def test_urlfield_clean_invalid(self):
        if False:
            print('Hello World!')
        f = URLField()
        tests = ['foo', 'com.', '.', 'http://', 'http://example', 'http://example.', 'http://.com', 'http://invalid-.com', 'http://-invalid.com', 'http://inv-.alid-.com', 'http://inv-.-alid.com', '[a', 'http://[a', 23, 'http://%s' % ('X' * 60,), 'http://%s' % ('X' * 200,), '////]@N.AN', '#@A.bO']
        msg = "'Enter a valid URL.'"
        for value in tests:
            with self.subTest(value=value):
                with self.assertRaisesMessage(ValidationError, msg):
                    f.clean(value)

    def test_urlfield_clean_required(self):
        if False:
            return 10
        f = URLField()
        msg = "'This field is required.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean(None)
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean('')

    def test_urlfield_clean_not_required(self):
        if False:
            i = 10
            return i + 15
        f = URLField(required=False)
        self.assertEqual(f.clean(None), '')
        self.assertEqual(f.clean(''), '')

    def test_urlfield_strip_on_none_value(self):
        if False:
            return 10
        f = URLField(required=False, empty_value=None)
        self.assertIsNone(f.clean(''))
        self.assertIsNone(f.clean(None))

    def test_urlfield_unable_to_set_strip_kwarg(self):
        if False:
            return 10
        msg = "__init__() got multiple values for keyword argument 'strip'"
        with self.assertRaisesMessage(TypeError, msg):
            URLField(strip=False)

    def test_urlfield_assume_scheme(self):
        if False:
            while True:
                i = 10
        f = URLField()
        self.assertEqual(f.clean('example.com'), 'http://example.com')
        f = URLField(assume_scheme='http')
        self.assertEqual(f.clean('example.com'), 'http://example.com')
        f = URLField(assume_scheme='https')
        self.assertEqual(f.clean('example.com'), 'https://example.com')

class URLFieldAssumeSchemeDeprecationTest(FormFieldAssertionsMixin, SimpleTestCase):

    def test_urlfield_raises_warning(self):
        if False:
            while True:
                i = 10
        msg = "The default scheme will be changed from 'http' to 'https' in Django 6.0. Pass the forms.URLField.assume_scheme argument to silence this warning."
        with self.assertWarnsMessage(RemovedInDjango60Warning, msg):
            f = URLField()
            self.assertEqual(f.clean('example.com'), 'http://example.com')