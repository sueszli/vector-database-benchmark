import os
from unittest import TestCase
from unittest.mock import Mock
from parameterized import parameterized
import samcli.yamlhelper as yamlhelper
from samcli.lib.translate.sam_template_validator import SamTemplateValidator
from samcli.commands.validate.lib.exceptions import InvalidSamDocumentException
TEMPLATE_DIR = 'tests/functional/commands/validate/lib/models'

class TestValidate(TestCase):
    VALID_TEST_TEMPLATES = [os.path.join(TEMPLATE_DIR, filename) for filename in os.listdir(TEMPLATE_DIR)]

    def test_valid_template(self):
        if False:
            i = 10
            return i + 15
        template = {'AWSTemplateFormatVersion': '2010-09-09', 'Transform': 'AWS::Serverless-2016-10-31', 'Resources': {'ServerlessFunction': {'Type': 'AWS::Serverless::Function', 'Properties': {'Handler': 'index.handler', 'CodeUri': 's3://fake-bucket/lambda-code.zip', 'Runtime': 'nodejs6.10', 'Timeout': 60}}}}
        managed_policy_mock = Mock()
        managed_policy_mock.load.return_value = {'PolicyName': 'FakePolicy'}
        validator = SamTemplateValidator(template, managed_policy_mock, region='us-east-1')
        validator.get_translated_template_if_valid()

    def test_invalid_template(self):
        if False:
            return 10
        template = {'AWSTemplateFormatVersion': '2010-09-09', 'Transform': 'AWS::Serverless-2016-10-31', 'Resources': {'ServerlessFunction': {'Type': 'AWS::Serverless::Function', 'Properties': {'Handler': 'index.handler', 'CodeUri': 's3://lambda-code.zip', 'Timeout': 60}}}}
        managed_policy_mock = Mock()
        managed_policy_mock.load.return_value = {'PolicyName': 'FakePolicy'}
        validator = SamTemplateValidator(template, managed_policy_mock, region='us-east-1')
        with self.assertRaises(InvalidSamDocumentException):
            validator.get_translated_template_if_valid()

    def test_valid_template_with_local_code_for_function(self):
        if False:
            print('Hello World!')
        template = {'AWSTemplateFormatVersion': '2010-09-09', 'Transform': 'AWS::Serverless-2016-10-31', 'Resources': {'ServerlessFunction': {'Type': 'AWS::Serverless::Function', 'Properties': {'Handler': 'index.handler', 'CodeUri': './', 'Runtime': 'nodejs6.10', 'Timeout': 60}}}}
        managed_policy_mock = Mock()
        managed_policy_mock.load.return_value = {'PolicyName': 'FakePolicy'}
        validator = SamTemplateValidator(template, managed_policy_mock, region='us-east-1')
        validator.get_translated_template_if_valid()

    def test_valid_template_with_local_code_for_layer_version(self):
        if False:
            for i in range(10):
                print('nop')
        template = {'AWSTemplateFormatVersion': '2010-09-09', 'Transform': 'AWS::Serverless-2016-10-31', 'Resources': {'ServerlessLayerVersion': {'Type': 'AWS::Serverless::LayerVersion', 'Properties': {'ContentUri': './'}}}}
        managed_policy_mock = Mock()
        managed_policy_mock.load.return_value = {'PolicyName': 'FakePolicy'}
        validator = SamTemplateValidator(template, managed_policy_mock, region='us-east-1')
        validator.get_translated_template_if_valid()

    def test_valid_template_with_local_code_for_api(self):
        if False:
            while True:
                i = 10
        template = {'AWSTemplateFormatVersion': '2010-09-09', 'Transform': 'AWS::Serverless-2016-10-31', 'Resources': {'ServerlessApi': {'Type': 'AWS::Serverless::Api', 'Properties': {'StageName': 'Prod', 'DefinitionUri': './'}}}}
        managed_policy_mock = Mock()
        managed_policy_mock.load.return_value = {'PolicyName': 'FakePolicy'}
        validator = SamTemplateValidator(template, managed_policy_mock, region='us-east-1')
        validator.get_translated_template_if_valid()

    def test_valid_template_with_DefinitionBody_for_api(self):
        if False:
            return 10
        template = {'AWSTemplateFormatVersion': '2010-09-09', 'Transform': 'AWS::Serverless-2016-10-31', 'Resources': {'ServerlessApi': {'Type': 'AWS::Serverless::Api', 'Properties': {'StageName': 'Prod', 'DefinitionBody': {'swagger': '2.0'}}}}}
        managed_policy_mock = Mock()
        managed_policy_mock.load.return_value = {'PolicyName': 'FakePolicy'}
        validator = SamTemplateValidator(template, managed_policy_mock, region='us-east-1')
        validator.get_translated_template_if_valid()

    def test_valid_template_with_s3_object_passed(self):
        if False:
            i = 10
            return i + 15
        template = {'AWSTemplateFormatVersion': '2010-09-09', 'Transform': 'AWS::Serverless-2016-10-31', 'Resources': {'ServerlessApi': {'Type': 'AWS::Serverless::Api', 'Properties': {'StageName': 'Prod', 'DefinitionUri': {'Bucket': 'mybucket-name', 'Key': 'swagger', 'Version': 121212}}}, 'ServerlessFunction': {'Type': 'AWS::Serverless::Function', 'Properties': {'Handler': 'index.handler', 'CodeUri': {'Bucket': 'mybucket-name', 'Key': 'code.zip', 'Version': 121212}, 'Runtime': 'nodejs6.10', 'Timeout': 60}}}}
        managed_policy_mock = Mock()
        managed_policy_mock.load.return_value = {'PolicyName': 'FakePolicy'}
        validator = SamTemplateValidator(template, managed_policy_mock, region='us-east-1')
        validator.get_translated_template_if_valid()
        self.assertEqual(validator.sam_template.get('Resources').get('ServerlessApi').get('Properties').get('DefinitionUri'), {'Bucket': 'mybucket-name', 'Key': 'swagger', 'Version': 121212})
        self.assertEqual(validator.sam_template.get('Resources').get('ServerlessFunction').get('Properties').get('CodeUri'), {'Bucket': 'mybucket-name', 'Key': 'code.zip', 'Version': 121212})

    @parameterized.expand(VALID_TEST_TEMPLATES)
    def test_valid_api_request_model_template(self, template_path):
        if False:
            while True:
                i = 10
        with open(template_path) as f:
            template = yamlhelper.yaml_parse(f.read())
        managed_policy_mock = Mock()
        managed_policy_mock.load.return_value = {'PolicyName': 'FakePolicy'}
        validator = SamTemplateValidator(template, managed_policy_mock, region='us-east-1')
        validator.get_translated_template_if_valid()