import locale
import logging
import os
import subprocess
import sys
import unittest
from collections.abc import Sequence
from shutil import rmtree
from tempfile import mkdtemp
from pelican import Pelican
from pelican.generators import StaticGenerator
from pelican.settings import read_settings
from pelican.tests.support import LoggedTestCase, diff_subproc, locale_available, mute, skipIfNoExecutable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_PATH = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir, 'samples'))
OUTPUT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, 'output'))
INPUT_PATH = os.path.join(SAMPLES_PATH, 'content')
SAMPLE_CONFIG = os.path.join(SAMPLES_PATH, 'pelican.conf.py')
SAMPLE_FR_CONFIG = os.path.join(SAMPLES_PATH, 'pelican.conf_FR.py')

def recursiveDiff(dcmp):
    if False:
        return 10
    diff = {'diff_files': [os.path.join(dcmp.right, f) for f in dcmp.diff_files], 'left_only': [os.path.join(dcmp.right, f) for f in dcmp.left_only], 'right_only': [os.path.join(dcmp.right, f) for f in dcmp.right_only]}
    for sub_dcmp in dcmp.subdirs.values():
        for (k, v) in recursiveDiff(sub_dcmp).items():
            diff[k] += v
    return diff

class TestPelican(LoggedTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.temp_path = mkdtemp(prefix='pelicantests.')
        self.temp_cache = mkdtemp(prefix='pelican_cache.')
        self.maxDiff = None
        self.old_locale = locale.setlocale(locale.LC_ALL)
        locale.setlocale(locale.LC_ALL, 'C')

    def tearDown(self):
        if False:
            return 10
        read_settings()
        rmtree(self.temp_path)
        rmtree(self.temp_cache)
        locale.setlocale(locale.LC_ALL, self.old_locale)
        super().tearDown()

    def assertDirsEqual(self, left_path, right_path, msg=None):
        if False:
            print('Hello World!')
        '\n        Check if the files are the same (ignoring whitespace) below both paths.\n        '
        proc = diff_subproc(left_path, right_path)
        (out, err) = proc.communicate()
        if proc.returncode != 0:
            msg = self._formatMessage(msg, '%s and %s differ:\nstdout:\n%s\nstderr\n%s' % (left_path, right_path, out, err))
            raise self.failureException(msg)

    def test_order_of_generators(self):
        if False:
            while True:
                i = 10
        pelican = Pelican(settings=read_settings(path=None))
        generator_classes = pelican._get_generator_classes()
        self.assertTrue(generator_classes[-1] is StaticGenerator, "StaticGenerator must be the last generator, but it isn't!")
        self.assertIsInstance(generator_classes, Sequence, '_get_generator_classes() must return a Sequence to preserve order')

    @skipIfNoExecutable(['git', '--version'])
    def test_basic_generation_works(self):
        if False:
            i = 10
            return i + 15
        settings = read_settings(path=None, override={'PATH': INPUT_PATH, 'OUTPUT_PATH': self.temp_path, 'CACHE_PATH': self.temp_cache, 'LOCALE': locale.normalize('en_US')})
        pelican = Pelican(settings=settings)
        mute(True)(pelican.run)()
        self.assertDirsEqual(self.temp_path, os.path.join(OUTPUT_PATH, 'basic'))
        self.assertLogCountEqual(count=1, msg='Unable to find.*skipping url replacement', level=logging.WARNING)

    @skipIfNoExecutable(['git', '--version'])
    def test_custom_generation_works(self):
        if False:
            print('Hello World!')
        settings = read_settings(path=SAMPLE_CONFIG, override={'PATH': INPUT_PATH, 'OUTPUT_PATH': self.temp_path, 'CACHE_PATH': self.temp_cache, 'LOCALE': locale.normalize('en_US.UTF-8')})
        pelican = Pelican(settings=settings)
        mute(True)(pelican.run)()
        self.assertDirsEqual(self.temp_path, os.path.join(OUTPUT_PATH, 'custom'))

    @skipIfNoExecutable(['git', '--version'])
    @unittest.skipUnless(locale_available('fr_FR.UTF-8') or locale_available('French'), 'French locale needed')
    def test_custom_locale_generation_works(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that generation with fr_FR.UTF-8 locale works'
        if sys.platform == 'win32':
            our_locale = 'French'
        else:
            our_locale = 'fr_FR.UTF-8'
        settings = read_settings(path=SAMPLE_FR_CONFIG, override={'PATH': INPUT_PATH, 'OUTPUT_PATH': self.temp_path, 'CACHE_PATH': self.temp_cache, 'LOCALE': our_locale})
        pelican = Pelican(settings=settings)
        mute(True)(pelican.run)()
        self.assertDirsEqual(self.temp_path, os.path.join(OUTPUT_PATH, 'custom_locale'))

    def test_theme_static_paths_copy(self):
        if False:
            i = 10
            return i + 15
        settings = read_settings(path=SAMPLE_CONFIG, override={'PATH': INPUT_PATH, 'OUTPUT_PATH': self.temp_path, 'CACHE_PATH': self.temp_cache, 'THEME_STATIC_PATHS': [os.path.join(SAMPLES_PATH, 'very'), os.path.join(SAMPLES_PATH, 'kinda'), os.path.join(SAMPLES_PATH, 'theme_standard')]})
        pelican = Pelican(settings=settings)
        mute(True)(pelican.run)()
        theme_output = os.path.join(self.temp_path, 'theme')
        extra_path = os.path.join(theme_output, 'exciting', 'new', 'files')
        for file in ['a_stylesheet', 'a_template']:
            self.assertTrue(os.path.exists(os.path.join(theme_output, file)))
        for file in ['wow!', 'boom!', 'bap!', 'zap!']:
            self.assertTrue(os.path.exists(os.path.join(extra_path, file)))

    def test_theme_static_paths_copy_single_file(self):
        if False:
            return 10
        settings = read_settings(path=SAMPLE_CONFIG, override={'PATH': INPUT_PATH, 'OUTPUT_PATH': self.temp_path, 'CACHE_PATH': self.temp_cache, 'THEME_STATIC_PATHS': [os.path.join(SAMPLES_PATH, 'theme_standard')]})
        pelican = Pelican(settings=settings)
        mute(True)(pelican.run)()
        theme_output = os.path.join(self.temp_path, 'theme')
        for file in ['a_stylesheet', 'a_template']:
            self.assertTrue(os.path.exists(os.path.join(theme_output, file)))

    def test_cyclic_intersite_links_no_warnings(self):
        if False:
            for i in range(10):
                print('nop')
        settings = read_settings(path=None, override={'PATH': os.path.join(CURRENT_DIR, 'cyclic_intersite_links'), 'OUTPUT_PATH': self.temp_path, 'CACHE_PATH': self.temp_cache})
        pelican = Pelican(settings=settings)
        mute(True)(pelican.run)()
        self.assertLogCountEqual(count=1, msg="Unable to find '.*\\.rst', skipping url replacement.", level=logging.WARNING)

    def test_md_extensions_deprecation(self):
        if False:
            i = 10
            return i + 15
        'Test that a warning is issued if MD_EXTENSIONS is used'
        settings = read_settings(path=None, override={'PATH': INPUT_PATH, 'OUTPUT_PATH': self.temp_path, 'CACHE_PATH': self.temp_cache, 'MD_EXTENSIONS': {}})
        pelican = Pelican(settings=settings)
        mute(True)(pelican.run)()
        self.assertLogCountEqual(count=1, msg='MD_EXTENSIONS is deprecated use MARKDOWN instead.', level=logging.WARNING)

    def test_parse_errors(self):
        if False:
            return 10
        settings = read_settings(path=None, override={'PATH': os.path.abspath(os.path.join(CURRENT_DIR, 'parse_error')), 'OUTPUT_PATH': self.temp_path, 'CACHE_PATH': self.temp_cache})
        pelican = Pelican(settings=settings)
        mute(True)(pelican.run)()
        self.assertLogCountEqual(count=1, msg='Could not process .*parse_error.rst', level=logging.ERROR)

    def test_module_load(self):
        if False:
            return 10
        'Test loading via python -m pelican --help displays the help'
        output = subprocess.check_output([sys.executable, '-m', 'pelican', '--help']).decode('ascii', 'replace')
        assert 'usage:' in output