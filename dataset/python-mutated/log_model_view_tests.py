from unittest.mock import patch
from superset.views.log.views import LogModelView
from .base_tests import SupersetTestCase

class TestLogModelView(SupersetTestCase):

    def test_disabled(self):
        if False:
            return 10
        with patch.object(LogModelView, 'is_enabled', return_value=False):
            self.login('admin')
            uri = '/logmodelview/list/'
            rv = self.client.get(uri)
            self.assert404(rv)

    def test_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.object(LogModelView, 'is_enabled', return_value=True):
            self.login('admin')
            uri = '/logmodelview/list/'
            rv = self.client.get(uri)
            self.assert200(rv)