"""
.. module: security_monkey.tests.watchers.rds.test_rds_security_group
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.tests.watchers import SecurityMonkeyWatcherTestCase
from security_monkey.watchers.rds.rds_security_group import RDSSecurityGroup
from security_monkey import AWS_DEFAULT_REGION
import boto
from moto import mock_sts, mock_rds_deprecated
from freezegun import freeze_time

class RDSecurityGroupWatcherTestCase(SecurityMonkeyWatcherTestCase):

    @freeze_time('2016-07-18 12:00:00')
    @mock_sts
    @mock_rds_deprecated
    def test_slurp(self):
        if False:
            while True:
                i = 10
        conn = boto.rds.connect_to_region(AWS_DEFAULT_REGION)
        conn.create_dbsecurity_group('db_sg1', 'DB Security Group')
        watcher = RDSSecurityGroup(accounts=[self.account.name])
        (item_list, exception_map) = watcher.slurp()
        self.assertIs(expr1=len(item_list), expr2=1, msg='Watcher should have 1 item but has {}'.format(len(item_list)))