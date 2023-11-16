from redash.models import ApiKey
from tests import BaseTestCase

class TestApiKeyGetByObject(BaseTestCase):

    def test_returns_none_if_not_exists(self):
        if False:
            return 10
        dashboard = self.factory.create_dashboard()
        self.assertIsNone(ApiKey.get_by_object(dashboard))

    def test_returns_only_active_key(self):
        if False:
            i = 10
            return i + 15
        dashboard = self.factory.create_dashboard()
        api_key = self.factory.create_api_key(object=dashboard, active=False)
        self.assertIsNone(ApiKey.get_by_object(dashboard))
        api_key = self.factory.create_api_key(object=dashboard)
        self.assertEqual(api_key, ApiKey.get_by_object(dashboard))