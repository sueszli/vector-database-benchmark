import boto3.session
from boto3.ec2 import createtags
from tests import mock, unittest

class TestCreateTags(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.client = mock.Mock()
        self.resource = mock.Mock()
        self.resource.meta.client = self.client
        self.ref_tags = ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6']
        self.resource.Tag.side_effect = self.ref_tags

    def test_create_tags(self):
        if False:
            i = 10
            return i + 15
        ref_kwargs = {'Resources': ['foo', 'bar'], 'Tags': [{'Key': 'key1', 'Value': 'value1'}, {'Key': 'key2', 'Value': 'value2'}, {'Key': 'key3', 'Value': 'value3'}]}
        result_tags = createtags.create_tags(self.resource, **ref_kwargs)
        self.client.create_tags.assert_called_with(**ref_kwargs)
        assert self.resource.Tag.call_args_list == [mock.call('foo', 'key1', 'value1'), mock.call('foo', 'key2', 'value2'), mock.call('foo', 'key3', 'value3'), mock.call('bar', 'key1', 'value1'), mock.call('bar', 'key2', 'value2'), mock.call('bar', 'key3', 'value3')]
        assert result_tags == self.ref_tags

class TestCreateTagsInjection(unittest.TestCase):

    def test_create_tags_injected_to_resource(self):
        if False:
            return 10
        session = boto3.session.Session(region_name='us-west-2')
        with mock.patch('boto3.ec2.createtags.create_tags') as mock_method:
            resource = session.resource('ec2')
            assert hasattr(resource, 'create_tags')
            assert resource.create_tags is mock_method