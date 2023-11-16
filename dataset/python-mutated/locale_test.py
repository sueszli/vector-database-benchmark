import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type

class TranslationLoaderTest(unittest.TestCase):
    SAVE_VARS = ['_translations', '_supported_locales', '_use_gettext']

    def clear_locale_cache(self):
        if False:
            while True:
                i = 10
        tornado.locale.Locale._cache = {}

    def setUp(self):
        if False:
            print('Hello World!')
        self.saved = {}
        for var in TranslationLoaderTest.SAVE_VARS:
            self.saved[var] = getattr(tornado.locale, var)
        self.clear_locale_cache()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        for (k, v) in self.saved.items():
            setattr(tornado.locale, k, v)
        self.clear_locale_cache()

    def test_csv(self):
        if False:
            print('Hello World!')
        tornado.locale.load_translations(os.path.join(os.path.dirname(__file__), 'csv_translations'))
        locale = tornado.locale.get('fr_FR')
        self.assertTrue(isinstance(locale, tornado.locale.CSVLocale))
        self.assertEqual(locale.translate('school'), 'école')

    def test_csv_bom(self):
        if False:
            for i in range(10):
                print('nop')
        with open(os.path.join(os.path.dirname(__file__), 'csv_translations', 'fr_FR.csv'), 'rb') as f:
            char_data = to_unicode(f.read())
        for encoding in ['utf-8-sig', 'utf-16']:
            tmpdir = tempfile.mkdtemp()
            try:
                with open(os.path.join(tmpdir, 'fr_FR.csv'), 'wb') as f:
                    f.write(char_data.encode(encoding))
                tornado.locale.load_translations(tmpdir)
                locale = tornado.locale.get('fr_FR')
                self.assertIsInstance(locale, tornado.locale.CSVLocale)
                self.assertEqual(locale.translate('school'), 'école')
            finally:
                shutil.rmtree(tmpdir)

    def test_gettext(self):
        if False:
            for i in range(10):
                print('nop')
        tornado.locale.load_gettext_translations(os.path.join(os.path.dirname(__file__), 'gettext_translations'), 'tornado_test')
        locale = tornado.locale.get('fr_FR')
        self.assertTrue(isinstance(locale, tornado.locale.GettextLocale))
        self.assertEqual(locale.translate('school'), 'école')
        self.assertEqual(locale.pgettext('law', 'right'), 'le droit')
        self.assertEqual(locale.pgettext('good', 'right'), 'le bien')
        self.assertEqual(locale.pgettext('organization', 'club', 'clubs', 1), 'le club')
        self.assertEqual(locale.pgettext('organization', 'club', 'clubs', 2), 'les clubs')
        self.assertEqual(locale.pgettext('stick', 'club', 'clubs', 1), 'le bâton')
        self.assertEqual(locale.pgettext('stick', 'club', 'clubs', 2), 'les bâtons')

class LocaleDataTest(unittest.TestCase):

    def test_non_ascii_name(self):
        if False:
            for i in range(10):
                print('nop')
        name = tornado.locale.LOCALE_NAMES['es_LA']['name']
        self.assertTrue(isinstance(name, unicode_type))
        self.assertEqual(name, 'Español')
        self.assertEqual(utf8(name), b'Espa\xc3\xb1ol')

class EnglishTest(unittest.TestCase):

    def test_format_date(self):
        if False:
            return 10
        locale = tornado.locale.get('en_US')
        date = datetime.datetime(2013, 4, 28, 18, 35)
        self.assertEqual(locale.format_date(date, full_format=True), 'April 28, 2013 at 6:35 pm')
        aware_dt = datetime.datetime.now(datetime.timezone.utc)
        naive_dt = aware_dt.replace(tzinfo=None)
        for (name, now) in {'aware': aware_dt, 'naive': naive_dt}.items():
            with self.subTest(dt=name):
                self.assertEqual(locale.format_date(now - datetime.timedelta(seconds=2), full_format=False), '2 seconds ago')
                self.assertEqual(locale.format_date(now - datetime.timedelta(minutes=2), full_format=False), '2 minutes ago')
                self.assertEqual(locale.format_date(now - datetime.timedelta(hours=2), full_format=False), '2 hours ago')
                self.assertEqual(locale.format_date(now - datetime.timedelta(days=1), full_format=False, shorter=True), 'yesterday')
                date = now - datetime.timedelta(days=2)
                self.assertEqual(locale.format_date(date, full_format=False, shorter=True), locale._weekdays[date.weekday()])
                date = now - datetime.timedelta(days=300)
                self.assertEqual(locale.format_date(date, full_format=False, shorter=True), '%s %d' % (locale._months[date.month - 1], date.day))
                date = now - datetime.timedelta(days=500)
                self.assertEqual(locale.format_date(date, full_format=False, shorter=True), '%s %d, %d' % (locale._months[date.month - 1], date.day, date.year))

    def test_friendly_number(self):
        if False:
            for i in range(10):
                print('nop')
        locale = tornado.locale.get('en_US')
        self.assertEqual(locale.friendly_number(1000000), '1,000,000')

    def test_list(self):
        if False:
            return 10
        locale = tornado.locale.get('en_US')
        self.assertEqual(locale.list([]), '')
        self.assertEqual(locale.list(['A']), 'A')
        self.assertEqual(locale.list(['A', 'B']), 'A and B')
        self.assertEqual(locale.list(['A', 'B', 'C']), 'A, B and C')

    def test_format_day(self):
        if False:
            print('Hello World!')
        locale = tornado.locale.get('en_US')
        date = datetime.datetime(2013, 4, 28, 18, 35)
        self.assertEqual(locale.format_day(date=date, dow=True), 'Sunday, April 28')
        self.assertEqual(locale.format_day(date=date, dow=False), 'April 28')