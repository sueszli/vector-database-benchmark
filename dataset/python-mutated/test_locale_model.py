from django.conf import settings
from django.test import TestCase, override_settings
from django.utils import translation
from django.utils.translation import gettext_lazy as _
from wagtail.models import Locale, Page
from wagtail.test.i18n.models import TestPage

def make_test_page(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    root_page = Page.objects.get(id=1)
    kwargs.setdefault('title', 'Test page')
    return root_page.add_child(instance=TestPage(**kwargs))

class TestLocaleModel(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        language_codes = dict(settings.LANGUAGES).keys()
        for language_code in language_codes:
            Locale.objects.get_or_create(language_code=language_code)

    def test_default(self):
        if False:
            while True:
                i = 10
        locale = Locale.get_default()
        self.assertEqual(locale.language_code, 'en')

    @override_settings(LANGUAGE_CODE='fr-ca')
    def test_default_doesnt_have_to_be_english(self):
        if False:
            print('Hello World!')
        locale = Locale.get_default()
        self.assertEqual(locale.language_code, 'fr')

    def test_get_active_default(self):
        if False:
            while True:
                i = 10
        self.assertEqual(Locale.get_active().language_code, 'en')

    def test_get_active_overridden(self):
        if False:
            return 10
        with translation.override('fr'):
            self.assertEqual(Locale.get_active().language_code, 'fr')

    def test_language_name(self):
        if False:
            return 10
        for (language_code, expected_result) in (('en', 'English'), ('fr', 'French'), ('zh-hans', 'Simplified Chinese')):
            with self.subTest(language_code):
                locale = Locale(language_code=language_code)
                self.assertEqual(locale.language_name, expected_result)

    def test_language_name_for_unrecognised_language(self):
        if False:
            print('Hello World!')
        locale = Locale(language_code='foo')
        with self.assertRaises(KeyError):
            locale.language_name

    def test_language_name_local(self):
        if False:
            print('Hello World!')
        for (language_code, expected_result) in (('en', 'English'), ('fr', 'français'), ('zh-hans', '简体中文')):
            with self.subTest(language_code):
                locale = Locale(language_code=language_code)
                self.assertEqual(locale.language_name_local, expected_result)

    def test_language_name_local_for_unrecognised_language(self):
        if False:
            while True:
                i = 10
        locale = Locale(language_code='foo')
        with self.assertRaises(KeyError):
            locale.language_name_local

    def test_language_name_localized_reflects_active_language(self):
        if False:
            return 10
        for language_code in ('fr', 'zh-hans', 'ca', 'de'):
            with self.subTest(language_code):
                locale = Locale(language_code=language_code)
                with translation.override('en'):
                    self.assertEqual(locale.language_name_localized, locale.language_name)
                with translation.override(language_code):
                    self.assertEqual(locale.language_name_localized.lower(), locale.language_name_local.lower())

    def test_language_name_localized_for_unconfigured_language(self):
        if False:
            return 10
        locale = Locale(language_code='zh-hans')
        self.assertEqual(locale.language_name_localized, 'Simplified Chinese')
        with translation.override('zh-hans'):
            self.assertEqual(locale.language_name_localized, locale.language_name_local)

    def test_language_name_localized_for_unrecognised_language(self):
        if False:
            print('Hello World!')
        locale = Locale(language_code='foo')
        with self.assertRaises(KeyError):
            locale.language_name_localized

    def test_is_bidi(self):
        if False:
            while True:
                i = 10
        for (language_code, expected_result) in (('en', False), ('ar', True), ('he', True), ('fr', False), ('foo', False)):
            with self.subTest(language_code):
                locale = Locale(language_code=language_code)
                self.assertIs(locale.is_bidi, expected_result)

    def test_is_default(self):
        if False:
            while True:
                i = 10
        for (language_code, expected_result) in ((settings.LANGUAGE_CODE, True), ('zh-hans', False), ('foo', False)):
            with self.subTest(language_code):
                locale = Locale(language_code=language_code)
                self.assertIs(locale.is_default, expected_result)

    def test_is_active(self):
        if False:
            print('Hello World!')
        for (locale_language, active_language, expected_result) in ((settings.LANGUAGE_CODE, settings.LANGUAGE_CODE, True), (settings.LANGUAGE_CODE, 'fr', False), ('zh-hans', settings.LANGUAGE_CODE, False), ('en', 'en-gb', True), ('foo', settings.LANGUAGE_CODE, False)):
            with self.subTest(f'locale={locale_language} active={active_language}'):
                with translation.override(active_language):
                    locale = Locale(language_code=locale_language)
                    self.assertEqual(locale.is_active, expected_result)

    def test_get_display_name(self):
        if False:
            print('Hello World!')
        for (language_code, expected_result) in (('en', 'English'), ('zh-hans', 'Simplified Chinese'), ('foo', 'foo')):
            locale = Locale(language_code=language_code)
            with self.subTest(language_code):
                self.assertEqual(locale.get_display_name(), expected_result)

    def test_str_reflects_get_display(self):
        if False:
            print('Hello World!')
        for language_code in ('en', 'zh-hans', 'foo'):
            locale = Locale(language_code=language_code)
            with self.subTest(language_code):
                self.assertEqual(str(locale), locale.get_display_name())

    @override_settings(LANGUAGES=[('en', _('English')), ('fr', _('French'))])
    def test_str_when_languages_uses_gettext(self):
        if False:
            return 10
        locale = Locale(language_code='en')
        self.assertIsInstance(locale.__str__(), str)

    @override_settings(LANGUAGE_CODE='fr')
    def test_change_root_page_locale_on_locale_deletion(self):
        if False:
            print('Hello World!')
        "\n        On deleting the locale used for the root page (but no 'real' pages), the\n        root page should be reassigned to a new locale (the default one, if possible)\n        "
        Page.objects.filter(depth__gt=1).update(locale=Locale.objects.get(language_code='fr'))
        self.assertEqual(Page.get_first_root_node().locale.language_code, 'en')
        Locale.objects.get(language_code='en').delete()
        self.assertEqual(Page.get_first_root_node().locale.language_code, 'fr')