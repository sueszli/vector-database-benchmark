"""
A subset of the tests in tests/servers/tests exercising
django.contrib.staticfiles.testing.StaticLiveServerTestCase instead of
django.test.LiveServerTestCase.
"""
import os
from urllib.request import urlopen
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.core.exceptions import ImproperlyConfigured
from django.test import modify_settings, override_settings
TEST_ROOT = os.path.dirname(__file__)
TEST_SETTINGS = {'MEDIA_URL': 'media/', 'STATIC_URL': 'static/', 'MEDIA_ROOT': os.path.join(TEST_ROOT, 'project', 'site_media', 'media'), 'STATIC_ROOT': os.path.join(TEST_ROOT, 'project', 'site_media', 'static')}

class LiveServerBase(StaticLiveServerTestCase):
    available_apps = []

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.settings_override = override_settings(**TEST_SETTINGS)
        cls.settings_override.enable()
        cls.addClassCleanup(cls.settings_override.disable)
        super().setUpClass()

class StaticLiveServerChecks(LiveServerBase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        old_STATIC_URL = TEST_SETTINGS['STATIC_URL']
        TEST_SETTINGS['STATIC_URL'] = None
        try:
            cls.raises_exception()
        finally:
            TEST_SETTINGS['STATIC_URL'] = old_STATIC_URL

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        pass

    @classmethod
    def raises_exception(cls):
        if False:
            return 10
        try:
            super().setUpClass()
        except ImproperlyConfigured:
            pass
        else:
            raise Exception('setUpClass() should have raised an exception.')

    def test_test_test(self):
        if False:
            print('Hello World!')
        pass

class StaticLiveServerView(LiveServerBase):

    def urlopen(self, url):
        if False:
            print('Hello World!')
        return urlopen(self.live_server_url + url)

    @modify_settings(INSTALLED_APPS={'append': 'staticfiles_tests.apps.test'})
    def test_collectstatic_emulation(self):
        if False:
            return 10
        "\n        StaticLiveServerTestCase use of staticfiles' serve() allows it\n        to discover app's static assets without having to collectstatic first.\n        "
        with self.urlopen('/static/test/file.txt') as f:
            self.assertEqual(f.read().rstrip(b'\r\n'), b'In static directory.')