from .base_tests import SupersetTestCase
from .conftest import with_feature_flags

class TestDynamicPlugins(SupersetTestCase):

    @with_feature_flags(DYNAMIC_PLUGINS=False)
    def test_dynamic_plugins_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dynamic Plugins: Responds not found when disabled\n        '
        self.login(username='admin')
        uri = '/dynamic-plugins/api'
        rv = self.client.get(uri)
        self.assertEqual(rv.status_code, 404)

    @with_feature_flags(DYNAMIC_PLUGINS=True)
    def test_dynamic_plugins_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dynamic Plugins: Responds successfully when enabled\n        '
        self.login(username='admin')
        uri = '/dynamic-plugins/api'
        rv = self.client.get(uri)
        self.assertEqual(rv.status_code, 200)