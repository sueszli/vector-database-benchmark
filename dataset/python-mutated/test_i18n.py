"""
Tests for ``ckan.lib.i18n``.
"""
import codecs
import json
import os.path
import shutil
import tempfile
from unittest import mock
import pytest
from ckan.lib import i18n
from ckan import plugins
from ckan.lib.plugins import DefaultTranslation
HERE = os.path.abspath(os.path.dirname(__file__))
I18N_DIR = os.path.join(HERE, '_i18n_build_js_translations')
I18N_DUMMY_DIR = os.path.join(HERE, '_i18n_dummy_es')
I18N_TEMP_DIR = tempfile.mkdtemp()

class JSTranslationsTestPlugin(plugins.SingletonPlugin, DefaultTranslation):
    """
    CKAN plugin for testing JavaScript translations from extensions.

    Registered in ``setup.py`` as ``test_js_translations_plugin``.
    """
    plugins.implements(plugins.ITranslation)

    def i18n_directory(self):
        if False:
            for i in range(10):
                print('nop')
        return I18N_DIR

    def i18n_domain(self):
        if False:
            for i in range(10):
                print('nop')
        return 'ckanext-test_js_translations'

@pytest.fixture
def temp_i18n_dir():
    if False:
        return 10
    yield
    shutil.rmtree(I18N_TEMP_DIR, ignore_errors=True)

@pytest.mark.ckan_config('ckan.plugins', 'test_js_translations_plugin')
@pytest.mark.usefixtures('with_plugins', 'temp_i18n_dir')
class TestBuildJSTranslations(object):
    """
    Tests for ``ckan.lib.i18n.build_js_translations``.
    """
    temp_dir = I18N_TEMP_DIR

    def build_js_translations(self):
        if False:
            return 10
        '\n        Build JS translations in temporary directory.\n        '
        with mock.patch('ckan.lib.i18n.get_js_translations_dir', return_value=self.temp_dir):
            return i18n.build_js_translations()

    def test_output_is_valid(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the generated JS files are valid.\n        '

        def check_file(path):
            if False:
                for i in range(10):
                    print('nop')
            with codecs.open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert data[''].get('domain', None) == 'ckan'
        self.build_js_translations()
        files = os.listdir(self.temp_dir)
        assert set(i18n.get_locales()).difference(['en']) == set((os.path.splitext(fn)[0] for fn in files))
        for filename in files:
            check_file(os.path.join(self.temp_dir, filename))

    def test_regenerate_only_if_necessary(self):
        if False:
            while True:
                i = 10
        '\n        Test that translation files are only generated when necessary.\n        '
        self.build_js_translations()
        mtimes = {}
        for filename in os.listdir(self.temp_dir):
            fullname = os.path.join(self.temp_dir, filename)
            mtimes[filename] = os.path.getmtime(fullname)
        (removed_filename, outdated_filename) = sorted(mtimes.keys())[:2]
        mtimes.pop(removed_filename)
        outdated_mtime = mtimes.pop(outdated_filename)
        os.remove(os.path.join(self.temp_dir, removed_filename))
        os.utime(os.path.join(self.temp_dir, outdated_filename), (0, 0))
        self.build_js_translations()
        assert os.path.isfile(os.path.join(self.temp_dir, removed_filename))
        fullname = os.path.join(self.temp_dir, outdated_filename)
        assert os.path.getmtime(fullname) >= outdated_mtime
        for filename in os.listdir(self.temp_dir):
            if filename in [removed_filename, outdated_filename]:
                continue
            fullname = os.path.join(self.temp_dir, filename)
            new_mtime = os.path.getmtime(fullname)
            assert new_mtime == mtimes[filename]

    def test_translations_from_extensions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that translations from extensions are taken into account.\n        '
        self.build_js_translations()
        filename = os.path.join(self.temp_dir, 'de.js')
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            de = json.load(f)
        assert 'Loading...' in de
        assert de['Loading...'] == [None, 'foo']
        assert 'Test JS Translations 1' in de
        assert de['Test JS Translations 1'] == [None, 'bar']
        assert 'Test JS Translations 2' not in de

@pytest.mark.ckan_config('ckan.plugins', 'test_blueprint_plugin')
@pytest.mark.usefixtures('with_plugins')
class TestI18nFlask(object):

    def test_translation_works(self, app):
        if False:
            while True:
                i = 10
        resp = app.get('/view_translated')
        assert resp.data == b'Dataset'
        resp = app.get('/es/view_translated')
        assert resp.data == b'Conjunto de datos'

    @pytest.mark.ckan_config('ckan.i18n_directory', I18N_DUMMY_DIR)
    def test_config_i18n_directory(self, app):
        if False:
            print('Hello World!')
        resp = app.get('/view_translated')
        assert resp.data == b'Dataset'
        resp = app.get('/es/view_translated')
        assert resp.data == b'Foo baz 123'