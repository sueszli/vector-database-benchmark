"""
    :codeauthor: :email:`Amel Ajdinovic <amel@stackpointcloud.com>`
"""
import pytest
from tests.integration.cloud.helpers.cloud_test_base import TIMEOUT, CloudTest
try:
    from oneandone.client import OneAndOneService
    HAS_ONEANDONE = True
except ImportError:
    HAS_ONEANDONE = False

@pytest.mark.skipif(HAS_ONEANDONE is False, reason='salt-cloud requires >= 1and1 1.2.0')
class OneAndOneTest(CloudTest):
    """
    Integration tests for the 1and1 cloud provider
    """
    PROVIDER = 'oneandone'
    REQUIRED_PROVIDER_CONFIG_ITEMS = ('api_token',)

    def test_list_images(self):
        if False:
            return 10
        '\n        Tests the return of running the --list-images command for 1and1\n        '
        image_list = self.run_cloud('--list-images {}'.format(self.PROVIDER_NAME))
        self.assertIn('coreOSimage', [i.strip() for i in image_list])

    def test_instance(self):
        if False:
            return 10
        '\n        Test creating an instance on 1and1\n        '
        ret_str = self.run_cloud('-p oneandone-test {}'.format(self.instance_name), timeout=TIMEOUT)
        self.assertInstanceExists(ret_str)
        self.assertDestroyInstance()