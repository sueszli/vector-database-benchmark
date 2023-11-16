from __future__ import absolute_import
try:
    import simplejson as json
except ImportError:
    import json
import six
import mock
from st2common.exceptions.apivalidation import ValueValidationException
from st2common.models.api.action import ActionAPI
from st2common.bootstrap import runnersregistrar as runners_registrar
import st2common.validators.api.action as action_validator
from st2tests import DbTestCase
from st2tests.fixtures.packs import executions as fixture
__all__ = ['TestActionAPIValidator']

class TestActionAPIValidator(DbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(TestActionAPIValidator, cls).setUpClass()
        runners_registrar.register_runners()

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    def test_validate_runner_type_happy_case(self):
        if False:
            for i in range(10):
                print('nop')
        action_api_dict = fixture.ARTIFACTS['actions']['local']
        action_api = ActionAPI(**action_api_dict)
        try:
            action_validator.validate_action(action_api)
        except:
            self.fail('Exception validating action: %s' % json.dumps(action_api_dict))

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    def test_validate_runner_type_invalid_runner(self):
        if False:
            print('Hello World!')
        action_api_dict = fixture.ARTIFACTS['actions']['action-with-invalid-runner']
        action_api = ActionAPI(**action_api_dict)
        try:
            action_validator.validate_action(action_api)
            self.fail('Action validation should not have passed. %s' % json.dumps(action_api_dict))
        except ValueValidationException:
            pass

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    def test_validate_override_immutable_runner_param(self):
        if False:
            print('Hello World!')
        action_api_dict = fixture.ARTIFACTS['actions']['remote-override-runner-immutable']
        action_api = ActionAPI(**action_api_dict)
        try:
            action_validator.validate_action(action_api)
            self.fail('Action validation should not have passed. %s' % json.dumps(action_api_dict))
        except ValueValidationException as e:
            self.assertIn('Cannot override in action.', six.text_type(e))

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    def test_validate_action_param_immutable(self):
        if False:
            return 10
        action_api_dict = fixture.ARTIFACTS['actions']['action-immutable-param-no-default']
        action_api = ActionAPI(**action_api_dict)
        try:
            action_validator.validate_action(action_api)
            self.fail('Action validation should not have passed. %s' % json.dumps(action_api_dict))
        except ValueValidationException as e:
            self.assertIn('requires a default value.', six.text_type(e))

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    def test_validate_action_param_immutable_no_default(self):
        if False:
            for i in range(10):
                print('nop')
        action_api_dict = fixture.ARTIFACTS['actions']['action-immutable-runner-param-no-default']
        action_api = ActionAPI(**action_api_dict)
        try:
            action_validator.validate_action(action_api)
        except ValueValidationException as e:
            print(e)
            self.fail('Action validation should have passed. %s' % json.dumps(action_api_dict))

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    def test_validate_action_param_position_values_unique(self):
        if False:
            i = 10
            return i + 15
        action_api_dict = fixture.ARTIFACTS['actions']['action-with-non-unique-positions']
        action_api = ActionAPI(**action_api_dict)
        try:
            action_validator.validate_action(action_api)
            self.fail('Action validation should have failed ' + 'because position values are not unique.' % json.dumps(action_api_dict))
        except ValueValidationException as e:
            self.assertIn('have same position', six.text_type(e))

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    def test_validate_action_param_position_values_contiguous(self):
        if False:
            print('Hello World!')
        action_api_dict = fixture.ARTIFACTS['actions']['action-with-non-contiguous-positions']
        action_api = ActionAPI(**action_api_dict)
        try:
            action_validator.validate_action(action_api)
            self.fail('Action validation should have failed ' + 'because position values are not contiguous.' % json.dumps(action_api_dict))
        except ValueValidationException as e:
            self.assertIn('are not contiguous', six.text_type(e))