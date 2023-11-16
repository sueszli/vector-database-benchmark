import json
from botocore.compat import OrderedDict
from unittest import TestCase
from samcli.yamlhelper import yaml_parse, yaml_dump

class TestYaml(TestCase):
    yaml_with_tags = '\n    Resource:\n        Key1: !Ref Something\n        Key2: !GetAtt Another.Arn\n        Key3: !FooBar [!Baz YetAnother, "hello"]\n        Key4: !SomeTag {"a": "1"}\n        Key5: !GetAtt OneMore.Outputs.Arn\n        Key6: !Condition OtherCondition\n        Key7: "012345678"\n    '
    parsed_yaml_dict = {'Resource': {'Key1': {'Ref': 'Something'}, 'Key2': {'Fn::GetAtt': ['Another', 'Arn']}, 'Key3': {'Fn::FooBar': [{'Fn::Baz': 'YetAnother'}, 'hello']}, 'Key4': {'Fn::SomeTag': {'a': '1'}}, 'Key5': {'Fn::GetAtt': ['OneMore', 'Outputs.Arn']}, 'Key6': {'Condition': 'OtherCondition'}, 'Key7': '012345678'}}

    def test_yaml_with_tags(self):
        if False:
            while True:
                i = 10
        output = yaml_parse(self.yaml_with_tags)
        self.assertEqual(self.parsed_yaml_dict, output)
        formatted_str = yaml_dump(output)
        output_again = yaml_parse(formatted_str)
        self.assertEqual(output, output_again)

    def test_yaml_dumps(self):
        if False:
            for i in range(10):
                print('nop')
        input_yaml_dict = {'Resource': {'Key7': '012345678'}}
        expected_output = "Resource:\n  Key7: '012345678'\n"
        output = yaml_dump(input_yaml_dict)
        self.assertEqual(output, expected_output)

    def test_yaml_getatt(self):
        if False:
            while True:
                i = 10
        yaml_input = '\n        Resource:\n            Key: !GetAtt ["a", "b"]\n        '
        output = {'Resource': {'Key': {'Fn::GetAtt': ['a', 'b']}}}
        actual_output = yaml_parse(yaml_input)
        self.assertEqual(actual_output, output)

    def test_parse_json_with_tabs(self):
        if False:
            return 10
        template = '{\n\t"foo": "bar"\n}'
        output = yaml_parse(template)
        self.assertEqual(output, {'foo': 'bar'})

    def test_parse_json_preserve_elements_order(self):
        if False:
            while True:
                i = 10
        input_template = '\n        {\n            "B_Resource": {\n                "Key2": {\n                    "Name": "name2"\n                },\n                "Key1": {\n                    "Name": "name1"\n                }\n            },\n            "A_Resource": {\n                "Key2": {\n                    "Name": "name2"\n                },\n                "Key1": {\n                    "Name": "name1"\n                }\n            }\n        }\n        '
        expected_dict = OrderedDict([('B_Resource', OrderedDict([('Key2', {'Name': 'name2'}), ('Key1', {'Name': 'name1'})])), ('A_Resource', OrderedDict([('Key2', {'Name': 'name2'}), ('Key1', {'Name': 'name1'})]))])
        output_dict = yaml_parse(input_template)
        self.assertEqual(expected_dict, output_dict)

    def test_parse_yaml_preserve_elements_order(self):
        if False:
            i = 10
            return i + 15
        input_template = 'B_Resource:\n  Key2:\n    Name: name2\n  Key1:\n    Name: name1\nA_Resource:\n  Key2:\n    Name: name2\n  Key1:\n    Name: name1\n'
        output_dict = yaml_parse(input_template)
        expected_dict = OrderedDict([('B_Resource', OrderedDict([('Key2', {'Name': 'name2'}), ('Key1', {'Name': 'name1'})])), ('A_Resource', OrderedDict([('Key2', {'Name': 'name2'}), ('Key1', {'Name': 'name1'})]))])
        self.assertEqual(expected_dict, output_dict)
        output_template = yaml_dump(output_dict)
        self.assertEqual(input_template, output_template)

    def test_yaml_merge_tag(self):
        if False:
            return 10
        test_yaml = '\n        base: &base\n            property: value\n        test:\n            <<: *base\n        '
        output = yaml_parse(test_yaml)
        self.assertTrue(isinstance(output, OrderedDict))
        self.assertEqual(output.get('test').get('property'), 'value')

    def test_unroll_yaml_anchors(self):
        if False:
            return 10
        properties = {'Foo': 'bar', 'Spam': 'eggs'}
        template = {'Resources': {'Resource1': {'Properties': properties}, 'Resource2': {'Properties': properties}}}
        expected = 'Resources:\n  Resource1:\n    Properties:\n      Foo: bar\n      Spam: eggs\n  Resource2:\n    Properties:\n      Foo: bar\n      Spam: eggs\n'
        actual = yaml_dump(template)
        self.assertEqual(actual, expected)

    def test_unquoted_template_format_version_to_json(self):
        if False:
            while True:
                i = 10
        input_template = 'AWSTemplateFormatVersion: 2010-09-09\nTransform: AWS::Serverless-2016-10-31\nResources:\n  HelloWorldFunction:\n    Type: AWS::Serverless::Function\n    Properties:\n      Handler: main.handler\n      Runtime: python3.7\n      CodeUri: .\n      Timeout: 600\n'
        output = yaml_parse(input_template)
        self.assertIsInstance(output['AWSTemplateFormatVersion'], str)
        self.assertEqual(output['AWSTemplateFormatVersion'], '2010-09-09')
        json.dumps(output)