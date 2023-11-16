"""
.. module: security_monkey.tests.vpc.test_dhcp
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.tests.watchers import SecurityMonkeyWatcherTestCase
from security_monkey.watchers.vpc.dhcp import DHCP
from security_monkey import AWS_DEFAULT_REGION
import boto3
from moto import mock_sts, mock_ec2
from freezegun import freeze_time

class DHCPTestCase(SecurityMonkeyWatcherTestCase):

    @freeze_time('2016-07-18 12:00:00')
    @mock_sts
    @mock_ec2
    def test_slurp(self):
        if False:
            while True:
                i = 10
        ec2 = boto3.resource('ec2', region_name=AWS_DEFAULT_REGION)
        ec2.create_dhcp_options(DhcpConfigurations=[{'Key': 'domain-name', 'Values': ['example.com']}, {'Key': 'domain-name-servers', 'Values': ['10.0.10.2']}])
        watcher = DHCP(accounts=[self.account.name])
        (item_list, exception_map) = watcher.slurp()
        self.assertIs(expr1=len(item_list), expr2=1, msg='Watcher should have 1 item but has {}'.format(len(item_list)))