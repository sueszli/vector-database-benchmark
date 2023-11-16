"""
    :codeauthor: Ethan Devenport <ethand@stackpointcloud.com>
"""
import pytest
from tests.integration.cloud.helpers.cloud_test_base import TIMEOUT, CloudTest
try:
    from profitbricks.client import ProfitBricksService
    HAS_PROFITBRICKS = True
except ImportError:
    HAS_PROFITBRICKS = False

@pytest.mark.skipif(HAS_PROFITBRICKS is False, reason='salt-cloud requires >= profitbricks 4.1.0')
class ProfitBricksTest(CloudTest):
    """
    Integration tests for the ProfitBricks cloud provider
    """
    PROVIDER = 'profitbricks'
    REQUIRED_PROVIDER_CONFIG_ITEMS = ('username', 'password', 'datacenter_id')

    def setUp(self):
        if False:
            return 10
        super().setUp()
        username = self.provider_config.get('username')
        password = self.provider_config.get('password')
        self.skipTest('Conf items are missing that must be provided to run these tests:  username, password\nCheck tests/integration/files/conf/cloud.providers.d/{}.conf'.format(self.PROVIDER))

    def test_list_images(self):
        if False:
            return 10
        '\n        Tests the return of running the --list-images command for ProfitBricks\n        '
        list_images = self.run_cloud('--list-images {}'.format(self.PROVIDER))
        self.assertIn('Ubuntu-16.04-LTS-server-2017-10-01', [i.strip() for i in list_images])

    def test_list_image_alias(self):
        if False:
            while True:
                i = 10
        '\n        Tests the return of running the -f list_images\n        command for ProfitBricks\n        '
        cmd = '-f list_images {}'.format(self.PROVIDER)
        list_images = self.run_cloud(cmd)
        self.assertIn('- ubuntu:latest', [i.strip() for i in list_images])

    def test_list_sizes(self):
        if False:
            print('Hello World!')
        '\n        Tests the return of running the --list_sizes command for ProfitBricks\n        '
        list_sizes = self.run_cloud('--list-sizes {}'.format(self.PROVIDER))
        self.assertIn('Micro Instance:', [i.strip() for i in list_sizes])

    def test_list_datacenters(self):
        if False:
            while True:
                i = 10
        '\n        Tests the return of running the -f list_datacenters\n        command for ProfitBricks\n        '
        cmd = '-f list_datacenters {}'.format(self.PROVIDER)
        list_datacenters = self.run_cloud(cmd)
        self.assertIn(self.provider_config['datacenter_id'], [i.strip() for i in list_datacenters])

    def test_list_nodes(self):
        if False:
            return 10
        '\n        Tests the return of running the -f list_nodes command for ProfitBricks\n        '
        list_nodes = self.run_cloud('-f list_nodes {}'.format(self.PROVIDER))
        self.assertIn('state:', [i.strip() for i in list_nodes])
        self.assertIn('name:', [i.strip() for i in list_nodes])

    def test_list_nodes_full(self):
        if False:
            return 10
        '\n        Tests the return of running the -f list_nodes_full\n        command for ProfitBricks\n        '
        cmd = '-f list_nodes_full {}'.format(self.PROVIDER)
        list_nodes = self.run_cloud(cmd)
        self.assertIn('state:', [i.strip() for i in list_nodes])
        self.assertIn('name:', [i.strip() for i in list_nodes])

    def test_list_location(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests the return of running the --list-locations\n        command for ProfitBricks\n        '
        cmd = '--list-locations {}'.format(self.PROVIDER)
        list_locations = self.run_cloud(cmd)
        self.assertIn('de/fkb', [i.strip() for i in list_locations])
        self.assertIn('de/fra', [i.strip() for i in list_locations])
        self.assertIn('us/las', [i.strip() for i in list_locations])
        self.assertIn('us/ewr', [i.strip() for i in list_locations])

    def test_instance(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test creating an instance on ProfitBricks\n        '
        ret_str = self.run_cloud('-p profitbricks-test {}'.format(self.instance_name), timeout=TIMEOUT)
        self.assertInstanceExists(ret_str)
        self.assertDestroyInstance()