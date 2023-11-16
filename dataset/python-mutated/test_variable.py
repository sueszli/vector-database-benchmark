from __future__ import annotations
import logging
import os
from unittest import mock
import pytest
from cryptography.fernet import Fernet
from airflow import settings
from airflow.models import Variable, crypto, variable
from airflow.secrets.cache import SecretCache
from airflow.secrets.metastore import MetastoreBackend
from tests.test_utils import db
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test

class TestVariable:

    @pytest.fixture(autouse=True)
    def setup_test_cases(self):
        if False:
            return 10
        crypto._fernet = None
        db.clear_db_variables()
        SecretCache.reset()
        with conf_vars({('secrets', 'use_cache'): 'true'}):
            SecretCache.init()
        with mock.patch('airflow.models.variable.mask_secret', autospec=True) as m:
            self.mask_secret = m
            yield
        db.clear_db_variables()
        crypto._fernet = None

    @conf_vars({('core', 'fernet_key'): ''})
    def test_variable_no_encryption(self):
        if False:
            return 10
        '\n        Test variables without encryption\n        '
        Variable.set('key', 'value')
        session = settings.Session()
        test_var = session.query(Variable).filter(Variable.key == 'key').one()
        assert not test_var.is_encrypted
        assert test_var.val == 'value'
        self.mask_secret.assert_called_once_with('value', 'key')

    @conf_vars({('core', 'fernet_key'): Fernet.generate_key().decode()})
    def test_variable_with_encryption(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test variables with encryption\n        '
        Variable.set('key', 'value')
        session = settings.Session()
        test_var = session.query(Variable).filter(Variable.key == 'key').one()
        assert test_var.is_encrypted
        assert test_var.val == 'value'

    @pytest.mark.parametrize('test_value', ['value', ''])
    def test_var_with_encryption_rotate_fernet_key(self, test_value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests rotating encrypted variables.\n        '
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        with conf_vars({('core', 'fernet_key'): key1.decode()}):
            Variable.set('key', test_value)
            session = settings.Session()
            test_var = session.query(Variable).filter(Variable.key == 'key').one()
            assert test_var.is_encrypted
            assert test_var.val == test_value
            assert Fernet(key1).decrypt(test_var._val.encode()) == test_value.encode()
        with conf_vars({('core', 'fernet_key'): f'{key2.decode()},{key1.decode()}'}):
            crypto._fernet = None
            assert test_var.val == test_value
            test_var.rotate_fernet_key()
            assert test_var.is_encrypted
            assert test_var.val == test_value
            assert Fernet(key2).decrypt(test_var._val.encode()) == test_value.encode()

    def test_variable_set_get_round_trip(self):
        if False:
            for i in range(10):
                print('nop')
        Variable.set('tested_var_set_id', 'Monday morning breakfast')
        assert 'Monday morning breakfast' == Variable.get('tested_var_set_id')

    def test_variable_set_with_env_variable(self, caplog):
        if False:
            return 10
        caplog.set_level(logging.WARNING, logger=variable.log.name)
        Variable.set('key', 'db-value')
        with mock.patch.dict('os.environ', AIRFLOW_VAR_KEY='env-value'):
            Variable.set('key', 'new-db-value')
            assert 'env-value' == Variable.get('key')
        SecretCache.invalidate_variable('key')
        assert 'new-db-value' == Variable.get('key')
        assert caplog.messages[0] == 'The variable key is defined in the EnvironmentVariablesBackend secrets backend, which takes precedence over reading from the database. The value in the database will be updated, but to read it you have to delete the conflicting variable from EnvironmentVariablesBackend'

    @mock.patch('airflow.models.variable.ensure_secrets_loaded')
    def test_variable_set_with_extra_secret_backend(self, mock_ensure_secrets, caplog):
        if False:
            return 10
        caplog.set_level(logging.WARNING, logger=variable.log.name)
        mock_backend = mock.Mock()
        mock_backend.get_variable.return_value = 'secret_val'
        mock_backend.__class__.__name__ = 'MockSecretsBackend'
        mock_ensure_secrets.return_value = [mock_backend, MetastoreBackend]
        Variable.set('key', 'new-db-value')
        assert Variable.get('key') == 'secret_val'
        assert caplog.messages[0] == 'The variable key is defined in the MockSecretsBackend secrets backend, which takes precedence over reading from the database. The value in the database will be updated, but to read it you have to delete the conflicting variable from MockSecretsBackend'

    def test_variable_set_get_round_trip_json(self):
        if False:
            for i in range(10):
                print('nop')
        value = {'a': 17, 'b': 47}
        Variable.set('tested_var_set_id', value, serialize_json=True)
        assert value == Variable.get('tested_var_set_id', deserialize_json=True)

    def test_variable_update(self):
        if False:
            print('Hello World!')
        Variable.set('test_key', 'value1')
        assert 'value1' == Variable.get('test_key')
        Variable.update('test_key', 'value2')
        assert 'value2' == Variable.get('test_key')

    def test_variable_update_fails_on_non_metastore_variable(self):
        if False:
            while True:
                i = 10
        with mock.patch.dict('os.environ', AIRFLOW_VAR_KEY='env-value'):
            with pytest.raises(AttributeError):
                Variable.update('key', 'new-value')

    def test_variable_update_preserves_description(self):
        if False:
            print('Hello World!')
        Variable.set('key', 'value', description='a test variable')
        assert Variable.get('key') == 'value'
        Variable.update('key', 'value2')
        session = settings.Session()
        test_var = session.query(Variable).filter(Variable.key == 'key').one()
        assert test_var.val == 'value2'
        assert test_var.description == 'a test variable'

    def test_set_variable_sets_description(self):
        if False:
            for i in range(10):
                print('nop')
        Variable.set('key', 'value', description='a test variable')
        session = settings.Session()
        test_var = session.query(Variable).filter(Variable.key == 'key').one()
        assert test_var.description == 'a test variable'
        assert test_var.val == 'value'

    def test_variable_set_existing_value_to_blank(self):
        if False:
            for i in range(10):
                print('nop')
        test_value = 'Some value'
        test_key = 'test_key'
        Variable.set(test_key, test_value)
        Variable.set(test_key, '')
        assert '' == Variable.get('test_key')

    def test_get_non_existing_var_should_return_default(self):
        if False:
            print('Hello World!')
        default_value = 'some default val'
        assert default_value == Variable.get('thisIdDoesNotExist', default_var=default_value)

    def test_get_non_existing_var_should_raise_key_error(self):
        if False:
            while True:
                i = 10
        with pytest.raises(KeyError):
            Variable.get('thisIdDoesNotExist')

    def test_update_non_existing_var_should_raise_key_error(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(KeyError):
            Variable.update('thisIdDoesNotExist', 'value')

    def test_get_non_existing_var_with_none_default_should_return_none(self):
        if False:
            for i in range(10):
                print('nop')
        assert Variable.get('thisIdDoesNotExist', default_var=None) is None

    def test_get_non_existing_var_should_not_deserialize_json_default(self):
        if False:
            while True:
                i = 10
        default_value = '}{ this is a non JSON default }{'
        assert default_value == Variable.get('thisIdDoesNotExist', default_var=default_value, deserialize_json=True)

    def test_variable_setdefault_round_trip(self):
        if False:
            for i in range(10):
                print('nop')
        key = 'tested_var_setdefault_1_id'
        value = 'Monday morning breakfast in Paris'
        Variable.setdefault(key, value)
        assert value == Variable.get(key)

    def test_variable_setdefault_round_trip_json(self):
        if False:
            i = 10
            return i + 15
        key = 'tested_var_setdefault_2_id'
        value = {'city': 'Paris', 'Happiness': True}
        Variable.setdefault(key, value, deserialize_json=True)
        assert value == Variable.get(key, deserialize_json=True)

    def test_variable_setdefault_existing_json(self):
        if False:
            print('Hello World!')
        key = 'tested_var_setdefault_2_id'
        value = {'city': 'Paris', 'Happiness': True}
        Variable.set(key, value, serialize_json=True)
        val = Variable.setdefault(key, value, deserialize_json=True)
        assert value == val
        assert value == Variable.get(key, deserialize_json=True)

    def test_variable_delete(self):
        if False:
            return 10
        key = 'tested_var_delete'
        value = 'to be deleted'
        Variable.delete(key)
        with pytest.raises(KeyError):
            Variable.get(key)
        Variable.set(key, value)
        assert value == Variable.get(key)
        Variable.delete(key)
        with pytest.raises(KeyError):
            Variable.get(key)

    def test_masking_from_db(self):
        if False:
            print('Hello World!')
        'Test secrets are masked when loaded directly from the DB'
        session = settings.Session()
        try:
            var = Variable(key=f'password-{os.getpid()}', val='s3cr3t')
            session.add(var)
            session.flush()
            session.expunge(var)
            self.mask_secret.reset_mock()
            session.get(Variable, var.id)
            assert self.mask_secret.mock_calls == [mock.call('s3cr3t', var.key)]
        finally:
            session.rollback()

    @mock.patch('airflow.models.variable.ensure_secrets_loaded')
    def test_caching_caches(self, mock_ensure_secrets: mock.Mock):
        if False:
            for i in range(10):
                print('nop')
        mock_backend = mock.Mock()
        mock_backend.get_variable.return_value = 'secret_val'
        mock_backend.__class__.__name__ = 'MockSecretsBackend'
        mock_ensure_secrets.return_value = [mock_backend, MetastoreBackend]
        key = "doesn't matter"
        first = Variable.get(key)
        second = Variable.get(key)
        mock_backend.get_variable.assert_called_once()
        assert first == second

    def test_cache_invalidation_on_set(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch.dict('os.environ', AIRFLOW_VAR_KEY='from_env'):
            a = Variable.get('key')
        with mock.patch.dict('os.environ', AIRFLOW_VAR_KEY='from_env_two'):
            b = Variable.get('key')
        assert a == b
        Variable.set('key', 'new_value')
        c = Variable.get('key')
        assert c != b

@pytest.mark.parametrize('variable_value, deserialize_json, expected_masked_values', [('s3cr3t', False, ['s3cr3t']), ('{"api_key": "s3cr3t"}', True, ['s3cr3t']), ('{"api_key": "s3cr3t", "normal_key": "normal_value"}', True, ['s3cr3t']), ('{"api_key": "s3cr3t", "another_secret": "123456"}', True, ['s3cr3t', '123456'])])
def test_masking_only_secret_values(variable_value, deserialize_json, expected_masked_values):
    if False:
        return 10
    from airflow.utils.log.secrets_masker import _secrets_masker
    SecretCache.reset()
    session = settings.Session()
    try:
        var = Variable(key=f'password-{os.getpid()}', val=variable_value)
        session.add(var)
        session.flush()
        session.expunge(var)
        _secrets_masker().patterns = set()
        Variable.get(var.key, deserialize_json=deserialize_json)
        for expected_masked_value in expected_masked_values:
            assert expected_masked_value in _secrets_masker().patterns
    finally:
        session.rollback()
        db.clear_db_variables()