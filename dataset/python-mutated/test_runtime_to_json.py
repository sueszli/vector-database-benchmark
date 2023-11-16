import unittest
from typing import Mapping, Optional, Sequence
import pulumi
from pulumi.runtime import to_json

class ToJSONTests(unittest.TestCase):

    def test_to_json_basic_types(self):
        if False:
            while True:
                i = 10
        self.assertEqual('{}', to_json({}))
        self.assertEqual('[]', to_json([]))
        self.assertEqual('"hello"', to_json('hello'))
        self.assertEqual('42', to_json(42))
        self.assertEqual('{"hello": 42}', to_json({'hello': 42}))
        self.assertEqual('[1, 2, 3]', to_json([1, 2, 3]))
        self.assertEqual('["a", "b", "c"]', to_json(['a', 'b', 'c']))
        self.assertEqual('{"hello": [1, 2, 3]}', to_json({'hello': [1, 2, 3]}))
        self.assertEqual('[{"hello": 42}]', to_json([{'hello': 42}]))

    def test_to_json_basic_input_type(self):
        if False:
            for i in range(10):
                print('nop')

        @pulumi.input_type
        class ProviderAssumeRoleArgs:
            role_arn: Optional[pulumi.Input[str]] = pulumi.property('roleArn')
            tags: Optional[pulumi.Input[Mapping[str, pulumi.Input[str]]]]
        assume_role = ProviderAssumeRoleArgs(role_arn='some-arn', tags={'hello': 'world'})
        self.assertEqual('{"roleArn": "some-arn", "tags": {"hello": "world"}}', to_json(assume_role))

    def test_to_json_nested_input_type(self):
        if False:
            while True:
                i = 10

        @pulumi.input_type
        class ProviderFeaturesNetworkArgs:
            relaxed_locking: Optional[pulumi.Input[bool]] = pulumi.property('relaxedLocking')

        @pulumi.input_type
        class ProviderFeaturesArgs:
            network: Optional[pulumi.Input[ProviderFeaturesNetworkArgs]]
        features = ProviderFeaturesArgs(network=ProviderFeaturesNetworkArgs(relaxed_locking=False))
        self.assertEqual('{"network": {"relaxedLocking": false}}', to_json(features))