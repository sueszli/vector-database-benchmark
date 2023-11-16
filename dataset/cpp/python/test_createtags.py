# Copyright 2015 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# https://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import boto3.session
from boto3.ec2 import createtags
from tests import mock, unittest


class TestCreateTags(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()
        self.resource = mock.Mock()
        self.resource.meta.client = self.client
        self.ref_tags = ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6']
        self.resource.Tag.side_effect = self.ref_tags

    def test_create_tags(self):
        ref_kwargs = {
            'Resources': ['foo', 'bar'],
            'Tags': [
                {'Key': 'key1', 'Value': 'value1'},
                {'Key': 'key2', 'Value': 'value2'},
                {'Key': 'key3', 'Value': 'value3'},
            ],
        }

        result_tags = createtags.create_tags(self.resource, **ref_kwargs)

        # Ensure the client method was called properly.
        self.client.create_tags.assert_called_with(**ref_kwargs)

        # Ensure the calls to the Tag reference were correct.
        assert self.resource.Tag.call_args_list == [
            mock.call('foo', 'key1', 'value1'),
            mock.call('foo', 'key2', 'value2'),
            mock.call('foo', 'key3', 'value3'),
            mock.call('bar', 'key1', 'value1'),
            mock.call('bar', 'key2', 'value2'),
            mock.call('bar', 'key3', 'value3'),
        ]

        # Ensure the return values are as expected.
        assert result_tags == self.ref_tags


class TestCreateTagsInjection(unittest.TestCase):
    def test_create_tags_injected_to_resource(self):
        session = boto3.session.Session(region_name='us-west-2')
        with mock.patch('boto3.ec2.createtags.create_tags') as mock_method:
            resource = session.resource('ec2')
            assert hasattr(resource, 'create_tags')
            assert resource.create_tags is mock_method
