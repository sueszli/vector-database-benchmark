"""
.. module: security_monkey.tests.watchers.vpc.test_peering
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.tests.watchers import SecurityMonkeyWatcherTestCase
from security_monkey.watchers.vpc.route_table import RouteTable
import boto
from moto import mock_sts, mock_ec2_deprecated
from freezegun import freeze_time

class RouteTableWatcherTestCase(SecurityMonkeyWatcherTestCase):

    @freeze_time('2016-07-18 12:00:00')
    @mock_sts
    @mock_ec2_deprecated
    def test_slurp(self):
        if False:
            while True:
                i = 10
        conn = boto.connect_vpc('the_key', 'the secret')
        vpc = conn.create_vpc('10.0.0.0/16')
        watcher = RouteTable(accounts=[self.account.name])
        (item_list, exception_map) = watcher.slurp()
        vpc_ids = {routetable.config['vpc_id'] for routetable in item_list}
        self.assertIn(vpc.id, vpc_ids)