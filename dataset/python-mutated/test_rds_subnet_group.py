"""
.. module: security_monkey.tests.watchers.rds.test_rds_subnet_group
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.tests.watchers import SecurityMonkeyWatcherTestCase
from security_monkey.watchers.rds.rds_subnet_group import RDSSubnetGroup
from security_monkey import AWS_DEFAULT_REGION
import boto
from moto import mock_sts, mock_rds_deprecated, mock_ec2_deprecated
from freezegun import freeze_time

class RDSSubnetGroupWatcherTestCase(SecurityMonkeyWatcherTestCase):

    @freeze_time('2016-07-18 12:00:00')
    @mock_sts
    @mock_rds_deprecated
    @mock_ec2_deprecated
    def test_slurp(self):
        if False:
            i = 10
            return i + 15
        vpc_conn = boto.connect_vpc(AWS_DEFAULT_REGION)
        vpc = vpc_conn.create_vpc('10.0.0.0/16')
        subnet = vpc_conn.create_subnet(vpc.id, '10.0.0.0/24')
        subnet_ids = [subnet.id]
        conn = boto.rds.connect_to_region(AWS_DEFAULT_REGION)
        conn.create_db_subnet_group('db_subnet', 'my db subnet', subnet_ids)
        watcher = RDSSubnetGroup(accounts=[self.account.name])
        (item_list, exception_map) = watcher.slurp()
        self.assertIs(expr1=len(item_list), expr2=1, msg='Watcher should have 1 item but has {}'.format(len(item_list)))