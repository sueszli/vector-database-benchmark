import os
import time
import unittest
from unittest.mock import patch
from metaflow.exception import MetaflowException
import metaflow.metaflow_config
from metaflow.plugins.secrets.secrets_decorator import SecretSpec, validate_env_vars_across_secrets, validate_env_vars_vs_existing_env, validate_env_vars, get_secrets_backend_provider

class TestSecretsDecorator(unittest.TestCase):

    @patch('metaflow.metaflow_config.DEFAULT_SECRETS_BACKEND_TYPE', None)
    def test_missing_default_secrets_backend_type(self):
        if False:
            return 10
        self.assertIsNone(metaflow.metaflow_config.DEFAULT_SECRETS_BACKEND_TYPE)
        with self.assertRaises(MetaflowException):
            SecretSpec.secret_spec_from_str('secret_id', None)

    @patch('metaflow.metaflow_config.DEFAULT_SECRETS_BACKEND_TYPE', 'some-default-backend-type')
    def test_constructors(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual({'options': {}, 'secret_id': 'the_id', 'secrets_backend_type': 'explicit-type', 'role': None}, SecretSpec.secret_spec_from_str('explicit-type.the_id', None).to_json())
        self.assertEqual({'options': {}, 'secret_id': 'the_id', 'secrets_backend_type': 'some-default-backend-type', 'role': None}, SecretSpec.secret_spec_from_str('the_id', None).to_json())
        self.assertEqual({'options': {}, 'secret_id': 'the_id', 'secrets_backend_type': 'explicit-type', 'role': None}, SecretSpec.secret_spec_from_dict({'type': 'explicit-type', 'id': 'the_id'}, None).to_json())
        self.assertEqual({'options': {'a': 'b'}, 'secret_id': 'the_id', 'secrets_backend_type': 'some-default-backend-type', 'role': None}, SecretSpec.secret_spec_from_dict({'id': 'the_id', 'options': {'a': 'b'}}, None).to_json())
        self.assertDictEqual({'secret_id': 'the_id', 'secrets_backend_type': 'some-default-backend-type', 'role': 'source-level-role', 'options': {}}, SecretSpec.secret_spec_from_dict({'id': 'the_id', 'role': 'source-level-role'}, 'decorator-level-role').to_json())
        self.assertDictEqual({'secret_id': 'the_id', 'secrets_backend_type': 'some-default-backend-type', 'role': 'decorator-level-role', 'options': {}}, SecretSpec.secret_spec_from_dict({'id': 'the_id'}, role='decorator-level-role').to_json())
        with self.assertRaises(MetaflowException):
            SecretSpec.secret_spec_from_dict({'type': 42, 'id': 'the_id'}, None)
        with self.assertRaises(MetaflowException):
            SecretSpec.secret_spec_from_dict({'id': 42}, None)
        with self.assertRaises(MetaflowException):
            SecretSpec.secret_spec_from_dict({'id': 'the_id', 'options': []}, None)
        with self.assertRaises(MetaflowException):
            SecretSpec.secret_spec_from_dict({'id': 'the_id', 'role': 42}, None)

    def test_secrets_provider_resolution(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(MetaflowException):
            get_secrets_backend_provider(str(time.time()))

class TestEnvVarValidations(unittest.TestCase):

    def test_validate_env_vars_across_secrets(self):
        if False:
            i = 10
            return i + 15
        all_secrets_env_vars = [(SecretSpec.secret_spec_from_str('t.1', None), {'A': 'a', 'B': 'b'}), (SecretSpec.secret_spec_from_str('t.2', None), {'B': 'b', 'C': 'c'})]
        with self.assertRaises(MetaflowException):
            validate_env_vars_across_secrets(all_secrets_env_vars)

    def test_validate_env_vars_vs_existing_env(self):
        if False:
            while True:
                i = 10
        (existing_os_env_k, existing_os_env_v) = next(iter(os.environ.items()))
        all_secrets_env_vars = [(SecretSpec.secret_spec_from_str('t.1', None), {'A': 'a', existing_os_env_k: existing_os_env_v})]
        with self.assertRaises(MetaflowException):
            validate_env_vars_vs_existing_env(all_secrets_env_vars)

    def test_validate_env_vars(self):
        if False:
            while True:
                i = 10
        env_vars = {'TYPICAL_KEY_1': 'TYPICAL_VALUE_1', '_typical_key_2': 'typical_value_2'}
        validate_env_vars(env_vars)
        mistyped_keys = [1, tuple(), b'old_school']
        for k in mistyped_keys:
            with self.assertRaises(MetaflowException):
                validate_env_vars({k: 'v'})
        mistyped_values = [1, {}, b'old_school']
        for (i, v) in enumerate(mistyped_values):
            with self.assertRaises(MetaflowException):
                validate_env_vars({f'K{i}': v})
        weird_keys = ['1_', 'hello world', 'hey_arnold!', 'I_♥_NY', 'door-', 'METAFLOW_SOMETHING_OR_OTHER']
        for k in weird_keys:
            with self.assertRaises(MetaflowException):
                validate_env_vars({k: 'v'})
if __name__ == '__main__':
    unittest.main()