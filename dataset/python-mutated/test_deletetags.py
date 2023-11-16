from boto3.ec2.deletetags import delete_tags
from tests import mock, unittest

class TestDeleteTags(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.client = mock.Mock()
        self.resource = mock.Mock()
        self.resource.meta.client = self.client
        self.instance_id = 'instance_id'
        self.resource.id = self.instance_id

    def test_delete_tags(self):
        if False:
            for i in range(10):
                print('nop')
        tags = {'Tags': [{'Key': 'key1', 'Value': 'value1'}, {'Key': 'key2', 'Value': 'value2'}, {'Key': 'key3', 'Value': 'value3'}]}
        delete_tags(self.resource, **tags)
        kwargs = tags
        kwargs['Resources'] = [self.instance_id]
        self.client.delete_tags.assert_called_with(**kwargs)