from unittest import TestCase
from samcli.lib.warnings.sam_cli_warning import TemplateWarningsChecker, CodeDeployWarning, CodeDeployConditionWarning
from samcli.yamlhelper import yaml_parse
from parameterized import parameterized, param
import os
FAULTY_TEMPLATE = "\nResources:\n  Function:\n    Type: AWS::Serverless::Function\n    Properties:\n      DeploymentPreference:\n        Events: ''\n\n  preTrafficHook:\n    Type: AWS::Serverless::Function\n    Properties:\n      DeploymentPreference:\n        Enabled: false\n"
NO_PROPERTY_TEMPLATE = '\nResources:\n  Function:\n    Type: AWS::Serverless::Function\n\n  preTrafficHook:\n    Type: AWS::Serverless::Function\n'
ALL_DISABLED_TEMPLATE = '\nResources:\n  Function:\n    Type: AWS::Serverless::Function\n    Properties:\n      DeploymentPreference:\n        Enabled: false\n\n  preTrafficHook:\n    Type: AWS::Serverless::Function\n    Properties:\n      DeploymentPreference:\n        Enabled: false\n'
ALL_ENABLED_TEMPLATE = "\nResources:\n  Function:\n    Type: AWS::Serverless::Function\n    Properties:\n      DeploymentPreference:\n        Event: 'some-event'\n\n  preTrafficHook:\n    Type: AWS::Serverless::Function\n    Properties:\n      DeploymentPreference:\n        Event: 'some-event'\n"
NO_DEPLOYMENT_PREFERENCES = '\nResources:\n  Function:\n    Type: AWS::Serverless::Function\n    Properties:\n        Random: Property\n\n  preTrafficHook:\n    Type: AWS::Serverless::Function\n    Properties:\n        Random: Property\n'
NO_TYPE_RESOURCE = '\nResources:\n  Fn::Transform:\n    Name: AWS::Include\n    Parameters:\n      Location: ./deploy/queries.yaml\n'

class TestWarnings(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.msg = 'message'
        os.environ['SAM_CLI_TELEMETRY'] = '0'

    @parameterized.expand([param(FAULTY_TEMPLATE, CodeDeployWarning.WARNING_MESSAGE, True), param(ALL_DISABLED_TEMPLATE, CodeDeployWarning.WARNING_MESSAGE, False), param(ALL_ENABLED_TEMPLATE, CodeDeployWarning.WARNING_MESSAGE, False), param(NO_DEPLOYMENT_PREFERENCES, CodeDeployWarning.WARNING_MESSAGE, False), param(None, CodeDeployWarning.WARNING_MESSAGE, False)])
    def test_warning_check(self, template_txt, expected_warning_msg, message_present):
        if False:
            print('Hello World!')
        if template_txt:
            template_dict = yaml_parse(template_txt)
        else:
            template_dict = None
        current_warning_checker = TemplateWarningsChecker()
        actual_warning_msg = current_warning_checker.check_template_for_warning(CodeDeployWarning.__name__, template_dict)
        if not message_present:
            self.assertIsNone(actual_warning_msg)
        else:
            self.assertEqual(expected_warning_msg, actual_warning_msg)

    def test_warning_check_invalid_warning_name(self):
        if False:
            return 10
        template_dict = yaml_parse(ALL_ENABLED_TEMPLATE)
        current_warning_checker = TemplateWarningsChecker()
        actual_warning_msg = current_warning_checker.check_template_for_warning('SomeRandomName', template_dict)
        self.assertIsNone(actual_warning_msg)

class TestCodeDeployWarning(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.msg = 'message'
        os.environ['SAM_CLI_TELEMETRY'] = '0'

    @parameterized.expand([param(FAULTY_TEMPLATE, True), param(ALL_DISABLED_TEMPLATE, False), param(ALL_ENABLED_TEMPLATE, False), param(NO_PROPERTY_TEMPLATE, False), param(NO_TYPE_RESOURCE, False)])
    def test_code_deploy_warning(self, template, expected):
        if False:
            while True:
                i = 10
        code_deploy_warning = CodeDeployWarning()
        (is_warning, message) = code_deploy_warning.check(yaml_parse(template))
        self.assertEqual(expected, is_warning)
FUNCTION_WITH_CONDITION = "\nResources:\n  TestFunction:\n    Condition: value_dont_matter\n    Type: 'AWS::Serverless::Function'\n    Properties:\n      DeploymentPreference:\n        Type: Linear10PercentEvery2Minutes\n"
FUNCTION_WITHOUT_CONDITOIN = "\nResources:\n  TestFunction:\n    Type: 'AWS::Serverless::Function'\n    Properties:\n      DeploymentPreference:\n        Type: Linear10PercentEvery2Minutes\n"
FUNCTION_WITH_CONDITION_NO_DEPLOYMENT_PREFERENCES = "\nResources:\n  TestFunction:\n    Condition: value_dont_matter\n    Type: 'AWS::Serverless::Function'\n    Properties:\n      Handler: index.handler\n"

class TestCodeDeployWarningCondition(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.msg = 'message'
        os.environ['SAM_CLI_TELEMETRY'] = '0'

    @parameterized.expand([param(FUNCTION_WITH_CONDITION, True), param(FUNCTION_WITHOUT_CONDITOIN, False), param(FUNCTION_WITH_CONDITION_NO_DEPLOYMENT_PREFERENCES, False)])
    def test_code_deploy_warning_condition(self, template, expected):
        if False:
            i = 10
            return i + 15
        code_deploy_warning_condition = CodeDeployConditionWarning()
        (is_warning, _) = code_deploy_warning_condition.check(yaml_parse(template))
        self.assertEqual(expected, is_warning)