"""
    :codeauthor: Li Kexian <doyenli@tencent.com>
"""
import os
import pytest
from saltfactories.utils import random_string
from salt.config import cloud_providers_config
from tests.support.case import ShellCase
from tests.support.runtests import RUNTIME_VARS
INSTANCE_NAME = random_string('CLOUD-TEST-', lowercase=False)
PROVIDER_NAME = 'tencentcloud'

@pytest.mark.expensive_test
class TencentCloudTest(ShellCase):
    """
    Integration tests for the Tencent Cloud cloud provider in Salt-Cloud
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        Sets up the test requirements\n        '
        super().setUp()
        profile_str = 'tencentcloud-config'
        providers = self.run_cloud('--list-providers')
        if profile_str + ':' not in providers:
            self.skipTest('Configuration file for {0} was not found. Check {0}.conf files in tests/integration/files/conf/cloud.*.d/ to run these tests.'.format(PROVIDER_NAME))
        config = cloud_providers_config(os.path.join(RUNTIME_VARS.FILES, 'conf', 'cloud.providers.d', PROVIDER_NAME + '.conf'))
        tid = config[profile_str][PROVIDER_NAME]['id']
        key = config[profile_str][PROVIDER_NAME]['key']
        if tid == '' or key == '':
            self.skipTest('An api id and key must be provided to run these tests. Check tests/integration/files/conf/cloud.providers.d/{}.conf'.format(PROVIDER_NAME))

    def test_instance(self):
        if False:
            print('Hello World!')
        '\n        Test creating an instance on Tencent Cloud\n        '
        try:
            self.assertIn(INSTANCE_NAME, [i.strip() for i in self.run_cloud('-p tencentcloud-test {}'.format(INSTANCE_NAME), timeout=500)])
        except AssertionError:
            self.run_cloud('-d {} --assume-yes'.format(INSTANCE_NAME), timeout=500)
            raise
        self.assertIn(INSTANCE_NAME + ':', [i.strip() for i in self.run_cloud('-d {} --assume-yes'.format(INSTANCE_NAME), timeout=500)])

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clean up after tests\n        '
        query = self.run_cloud('--query')
        ret_str = '        {}:'.format(INSTANCE_NAME)
        if ret_str in query:
            self.run_cloud('-d {} --assume-yes'.format(INSTANCE_NAME), timeout=500)