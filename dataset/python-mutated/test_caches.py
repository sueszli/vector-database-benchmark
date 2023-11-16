import pathlib
from django.core.checks import Warning
from django.core.checks.caches import E001, check_cache_location_not_exposed, check_default_cache_is_configured, check_file_based_cache_is_absolute
from django.test import SimpleTestCase
from django.test.utils import override_settings

class CheckCacheSettingsAppDirsTest(SimpleTestCase):
    VALID_CACHES_CONFIGURATION = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
    INVALID_CACHES_CONFIGURATION = {'other': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}

    @override_settings(CACHES=VALID_CACHES_CONFIGURATION)
    def test_default_cache_included(self):
        if False:
            return 10
        "\n        Don't error if 'default' is present in CACHES setting.\n        "
        self.assertEqual(check_default_cache_is_configured(None), [])

    @override_settings(CACHES=INVALID_CACHES_CONFIGURATION)
    def test_default_cache_not_included(self):
        if False:
            return 10
        "\n        Error if 'default' not present in CACHES setting.\n        "
        self.assertEqual(check_default_cache_is_configured(None), [E001])

class CheckCacheLocationTest(SimpleTestCase):
    warning_message = "Your 'default' cache configuration might expose your cache or lead to corruption of your data because its LOCATION %s %s."

    @staticmethod
    def get_settings(setting, cache_path, setting_path):
        if False:
            while True:
                i = 10
        return {'CACHES': {'default': {'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache', 'LOCATION': cache_path}}, setting: [setting_path] if setting == 'STATICFILES_DIRS' else setting_path}

    def test_cache_path_matches_media_static_setting(self):
        if False:
            return 10
        root = pathlib.Path.cwd()
        for setting in ('MEDIA_ROOT', 'STATIC_ROOT', 'STATICFILES_DIRS'):
            settings = self.get_settings(setting, root, root)
            with self.subTest(setting=setting), self.settings(**settings):
                msg = self.warning_message % ('matches', setting)
                self.assertEqual(check_cache_location_not_exposed(None), [Warning(msg, id='caches.W002')])

    def test_cache_path_inside_media_static_setting(self):
        if False:
            return 10
        root = pathlib.Path.cwd()
        for setting in ('MEDIA_ROOT', 'STATIC_ROOT', 'STATICFILES_DIRS'):
            settings = self.get_settings(setting, root / 'cache', root)
            with self.subTest(setting=setting), self.settings(**settings):
                msg = self.warning_message % ('is inside', setting)
                self.assertEqual(check_cache_location_not_exposed(None), [Warning(msg, id='caches.W002')])

    def test_cache_path_contains_media_static_setting(self):
        if False:
            while True:
                i = 10
        root = pathlib.Path.cwd()
        for setting in ('MEDIA_ROOT', 'STATIC_ROOT', 'STATICFILES_DIRS'):
            settings = self.get_settings(setting, root, root / 'other')
            with self.subTest(setting=setting), self.settings(**settings):
                msg = self.warning_message % ('contains', setting)
                self.assertEqual(check_cache_location_not_exposed(None), [Warning(msg, id='caches.W002')])

    def test_cache_path_not_conflict(self):
        if False:
            while True:
                i = 10
        root = pathlib.Path.cwd()
        for setting in ('MEDIA_ROOT', 'STATIC_ROOT', 'STATICFILES_DIRS'):
            settings = self.get_settings(setting, root / 'cache', root / 'other')
            with self.subTest(setting=setting), self.settings(**settings):
                self.assertEqual(check_cache_location_not_exposed(None), [])

    def test_staticfiles_dirs_prefix(self):
        if False:
            print('Hello World!')
        root = pathlib.Path.cwd()
        tests = [(root, root, 'matches'), (root / 'cache', root, 'is inside'), (root, root / 'other', 'contains')]
        for (cache_path, setting_path, msg) in tests:
            settings = self.get_settings('STATICFILES_DIRS', cache_path, ('prefix', setting_path))
            with self.subTest(path=setting_path), self.settings(**settings):
                msg = self.warning_message % (msg, 'STATICFILES_DIRS')
                self.assertEqual(check_cache_location_not_exposed(None), [Warning(msg, id='caches.W002')])

    def test_staticfiles_dirs_prefix_not_conflict(self):
        if False:
            while True:
                i = 10
        root = pathlib.Path.cwd()
        settings = self.get_settings('STATICFILES_DIRS', root / 'cache', ('prefix', root / 'other'))
        with self.settings(**settings):
            self.assertEqual(check_cache_location_not_exposed(None), [])

class CheckCacheAbsolutePath(SimpleTestCase):

    def test_absolute_path(self):
        if False:
            print('Hello World!')
        with self.settings(CACHES={'default': {'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache', 'LOCATION': pathlib.Path.cwd() / 'cache'}}):
            self.assertEqual(check_file_based_cache_is_absolute(None), [])

    def test_relative_path(self):
        if False:
            while True:
                i = 10
        with self.settings(CACHES={'default': {'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache', 'LOCATION': 'cache'}}):
            self.assertEqual(check_file_based_cache_is_absolute(None), [Warning("Your 'default' cache LOCATION path is relative. Use an absolute path instead.", id='caches.W003')])