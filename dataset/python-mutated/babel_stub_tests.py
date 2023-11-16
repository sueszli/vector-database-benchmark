import unittest
from mkdocs.utils.babel_stub import Locale, UnknownLocaleError

class BabelStubTests(unittest.TestCase):

    def test_locale_language_only(self):
        if False:
            i = 10
            return i + 15
        locale = Locale('es')
        self.assertEqual(locale.language, 'es')
        self.assertEqual(locale.territory, '')
        self.assertEqual(str(locale), 'es')

    def test_locale_language_territory(self):
        if False:
            while True:
                i = 10
        locale = Locale('es', 'ES')
        self.assertEqual(locale.language, 'es')
        self.assertEqual(locale.territory, 'ES')
        self.assertEqual(str(locale), 'es_ES')

    def test_parse_locale_language_only(self):
        if False:
            i = 10
            return i + 15
        locale = Locale.parse('fr', '_')
        self.assertEqual(locale.language, 'fr')
        self.assertEqual(locale.territory, '')
        self.assertEqual(str(locale), 'fr')

    def test_parse_locale_language_territory(self):
        if False:
            while True:
                i = 10
        locale = Locale.parse('fr_FR', '_')
        self.assertEqual(locale.language, 'fr')
        self.assertEqual(locale.territory, 'FR')
        self.assertEqual(str(locale), 'fr_FR')

    def test_parse_locale_language_territory_sep(self):
        if False:
            while True:
                i = 10
        locale = Locale.parse('fr-FR', '-')
        self.assertEqual(locale.language, 'fr')
        self.assertEqual(locale.territory, 'FR')
        self.assertEqual(str(locale), 'fr_FR')

    def test_parse_locale_bad_type(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            Locale.parse(['list'], '_')

    def test_parse_locale_invalid_characters(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            Locale.parse('42', '_')

    def test_parse_locale_bad_format(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            Locale.parse('en-GB', '_')

    def test_parse_locale_bad_format_sep(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            Locale.parse('en_GB', '-')

    def test_parse_locale_unknown_locale(self):
        if False:
            return 10
        with self.assertRaises(UnknownLocaleError):
            Locale.parse('foo', '_')