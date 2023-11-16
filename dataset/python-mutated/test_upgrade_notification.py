from django.test import RequestFactory, TestCase, override_settings
from wagtail.admin.views.home import UpgradeNotificationPanel
from wagtail.test.utils import WagtailTestUtils

class TestUpgradeNotificationPanel(WagtailTestUtils, TestCase):
    DATA_ATTRIBUTE_UPGRADE_CHECK = 'data-w-upgrade'
    DATA_ATTRIBUTE_UPGRADE_CHECK_LTS = 'data-w-upgrade-lts-only'

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        cls.panel = UpgradeNotificationPanel()
        cls.request_factory = RequestFactory()
        cls.user = cls.create_user(username='tester')
        cls.superuser = cls.create_superuser(username='supertester')
        cls.request = cls.request_factory.get('/')

    def test_get_upgrade_check_setting_default(self):
        if False:
            return 10
        self.assertTrue(self.panel.get_upgrade_check_setting())

    @override_settings(WAGTAIL_ENABLE_UPDATE_CHECK=False)
    def test_get_upgrade_check_setting_false(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.panel.get_upgrade_check_setting())

    @override_settings(WAGTAIL_ENABLE_UPDATE_CHECK='LTS')
    def test_get_upgrade_check_setting_LTS(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.panel.get_upgrade_check_setting(), 'LTS')

    @override_settings(WAGTAIL_ENABLE_UPDATE_CHECK='lts')
    def test_get_upgrade_check_setting_lts(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.panel.get_upgrade_check_setting(), 'lts')

    def test_upgrade_check_lts_only_default(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.panel.upgrade_check_lts_only())

    @override_settings(WAGTAIL_ENABLE_UPDATE_CHECK=False)
    def test_upgrade_check_lts_only_setting_true(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.panel.upgrade_check_lts_only())

    @override_settings(WAGTAIL_ENABLE_UPDATE_CHECK='LTS')
    def test_upgrade_check_lts_only_setting_LTS(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.panel.upgrade_check_lts_only())

    @override_settings(WAGTAIL_ENABLE_UPDATE_CHECK='lts')
    def test_upgrade_check_lts_only_setting_lts(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.panel.upgrade_check_lts_only())

    def test_render_html_normal_user(self):
        if False:
            while True:
                i = 10
        self.request.user = self.user
        parent_context = {'request': self.request}
        result = self.panel.render_html(parent_context)
        self.assertEqual(result, '')

    def test_render_html_superuser(self):
        if False:
            while True:
                i = 10
        self.request.user = self.superuser
        parent_context = {'request': self.request}
        result = self.panel.render_html(parent_context)
        self.assertIn(self.DATA_ATTRIBUTE_UPGRADE_CHECK, result)
        self.assertNotIn(self.DATA_ATTRIBUTE_UPGRADE_CHECK_LTS, result)

    @override_settings(WAGTAIL_ENABLE_UPDATE_CHECK=False)
    def test_render_html_setting_false(self):
        if False:
            while True:
                i = 10
        self.request.user = self.superuser
        parent_context = {'request': self.request}
        result = self.panel.render_html(parent_context)
        self.assertEqual(result, '')

    @override_settings(WAGTAIL_ENABLE_UPDATE_CHECK='LTS')
    def test_render_html_setting_LTS(self):
        if False:
            for i in range(10):
                print('nop')
        self.request.user = self.superuser
        parent_context = {'request': self.request}
        result = self.panel.render_html(parent_context)
        self.assertIn(self.DATA_ATTRIBUTE_UPGRADE_CHECK, result)
        self.assertIn(self.DATA_ATTRIBUTE_UPGRADE_CHECK_LTS, result)

    @override_settings(WAGTAIL_ENABLE_UPDATE_CHECK='lts')
    def test_render_html_setting_lts(self):
        if False:
            while True:
                i = 10
        self.request.user = self.superuser
        parent_context = {'request': self.request}
        result = self.panel.render_html(parent_context)
        self.assertIn(self.DATA_ATTRIBUTE_UPGRADE_CHECK, result)
        self.assertIn(self.DATA_ATTRIBUTE_UPGRADE_CHECK_LTS, result)