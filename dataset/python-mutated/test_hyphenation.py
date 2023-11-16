import os
import shutil
import unittest
from lxml import etree
from calibre.ebooks.oeb.polish.parsing import parse_html5
from calibre.ptempfile import PersistentTemporaryDirectory
from calibre.utils.hyphenation.dictionaries import dictionary_name_for_locale, get_cache_path, is_cache_up_to_date, path_to_dictionary
from calibre.utils.hyphenation.hyphenate import add_soft_hyphens, add_soft_hyphens_to_html, add_soft_hyphens_to_words, dictionary_for_locale

class TestHyphenation(unittest.TestCase):
    ae = unittest.TestCase.assertEqual

    def setUp(self):
        if False:
            i = 10
            return i + 15
        tdir = PersistentTemporaryDirectory()
        path_to_dictionary.cache_dir = tdir
        dictionary_name_for_locale.cache_clear()
        dictionary_for_locale.cache_clear()
        get_cache_path.cache_clear()
        is_cache_up_to_date.updated = False

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        dictionary_name_for_locale.cache_clear()
        dictionary_for_locale.cache_clear()
        get_cache_path.cache_clear()
        is_cache_up_to_date.updated = False
        try:
            shutil.rmtree(path_to_dictionary.cache_dir)
        except OSError:
            pass
        path_to_dictionary.cache_dir = None

    def test_locale_to_hyphen_dictionary(self):
        if False:
            print('Hello World!')

        def t(x, expected=None):
            if False:
                for i in range(10):
                    print('nop')
            self.ae(dictionary_name_for_locale(x), f'hyph_{expected}.dic' if expected else None)
        t('en', 'en_US')
        t('en_IN', 'en_GB')
        t('de', 'de_DE')
        t('es', 'es')
        t('nl', 'nl_NL')
        t('fr', 'fr')
        t('XXX')
        cache = [False]

        def cache_callback():
            if False:
                print('Hello World!')
            cache[0] = True
        dp = path_to_dictionary(dictionary_name_for_locale('en'), cache_callback)
        self.assertTrue(os.path.exists(dp), f'The dictionary {dp} does not exist')
        self.assertTrue(cache[0])
        cache[0] = False
        self.assertTrue(os.path.exists(path_to_dictionary(dictionary_name_for_locale('es'), cache_callback)))
        self.assertFalse(cache[0])

    def test_add_soft_hyphens(self):
        if False:
            return 10

        def t(word, expected):
            if False:
                i = 10
                return i + 15
            self.ae(add_soft_hyphens(word, dictionary, '='), expected)
        dictionary = dictionary_for_locale('hu')
        t('asszonnyal', 'asszonnyal')
        dictionary = dictionary_for_locale('en')
        t('beautiful', 'beau=ti=ful')
        t('BeauTiful', 'Beau=Ti=ful')

        def w(words, expected):
            if False:
                while True:
                    i = 10
            self.ae(add_soft_hyphens_to_words(words, dictionary, '='), expected)
        w(' A\n beautiful  day. ', ' A\n beau=ti=ful  day. ')

    def test_hyphenate_html(self):
        if False:
            i = 10
            return i + 15
        root = parse_html5('\n<p>beautiful, <span lang="sv"><!-- x -->tillata\n<span lang="en">Expand</span></span> "latitude!', line_numbers=False)
        add_soft_hyphens_to_html(root, hyphen_char='=')
        raw = etree.tostring(root, method='text', encoding='unicode')
        self.ae(raw, 'beau=ti=ful, tilla=ta\nEx=pand "lat=i=tude!')

def find_tests():
    if False:
        for i in range(10):
            print('nop')
    return unittest.defaultTestLoader.loadTestsFromTestCase(TestHyphenation)