"""
.. module: security_monkey.tests.watchers.ec2.test_ec2_image
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.tests.watchers import SecurityMonkeyWatcherTestCase
from security_monkey.watchers.ec2.ec2_image import EC2Image
from security_monkey import AWS_DEFAULT_REGION
import boto3
from moto import mock_sts, mock_ec2
from freezegun import freeze_time

class EC2ImageWatcherTestCase(SecurityMonkeyWatcherTestCase):

    @freeze_time('2016-07-18 12:00:00')
    @mock_sts
    @mock_ec2
    def test_slurp(self):
        if False:
            print('Hello World!')

        def get_method(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if kwargs['region'] == 'us-east-1':
                return {'Arn': 'somearn', 'ImageId': 'ami-1234abcd'}
            return {}

        def list_method(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            if kwargs['region'] == 'us-east-1':
                return [{'Arn': 'somearn', 'ImageId': 'ami-1234abcd'}]
            return []
        EC2Image.get_method = lambda *args, **kwargs: get_method(*args, **kwargs)
        EC2Image.list_method = lambda *args, **kwargs: list_method(*args, **kwargs)
        watcher = EC2Image(accounts=[self.account.name])
        (item_list, exception_map) = watcher.slurp()
        self.assertIs(expr1=len(item_list), expr2=1, msg='Watcher should have 1 item but has {}'.format(len(item_list)))