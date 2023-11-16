import pytest
from unittest.mock import patch, MagicMock
import json
from openai.error import AuthenticationError
from core.model_providers.providers.base import CredentialsValidateFailedError
from core.model_providers.providers.openai_provider import OpenAIProvider
from models.provider import ProviderType, Provider
PROVIDER_NAME = 'openai'
MODEL_PROVIDER_CLASS = OpenAIProvider
VALIDATE_CREDENTIAL_KEY = 'openai_api_key'

def moderation_side_effect(*args, **kwargs):
    if False:
        return 10
    if kwargs['api_key'] == 'valid_key':
        mock_instance = MagicMock()
        mock_instance.request = MagicMock()
        return (mock_instance, {})
    else:
        raise AuthenticationError('Invalid credentials')

def encrypt_side_effect(tenant_id, encrypt_key):
    if False:
        while True:
            i = 10
    return f'encrypted_{encrypt_key}'

def decrypt_side_effect(tenant_id, encrypted_key):
    if False:
        i = 10
        return i + 15
    return encrypted_key.replace('encrypted_', '')

@patch('openai.ChatCompletion.create', side_effect=moderation_side_effect)
def test_is_provider_credentials_valid_or_raise_valid(mock_create):
    if False:
        return 10
    credentials = {VALIDATE_CREDENTIAL_KEY: 'valid_key'}
    assert MODEL_PROVIDER_CLASS.is_provider_credentials_valid_or_raise(credentials) is None

@patch('openai.ChatCompletion.create', side_effect=moderation_side_effect)
def test_is_provider_credentials_valid_or_raise_invalid(mock_create):
    if False:
        i = 10
        return i + 15
    with pytest.raises(CredentialsValidateFailedError):
        MODEL_PROVIDER_CLASS.is_provider_credentials_valid_or_raise({})
    with pytest.raises(CredentialsValidateFailedError):
        MODEL_PROVIDER_CLASS.is_provider_credentials_valid_or_raise({VALIDATE_CREDENTIAL_KEY: 'invalid_key'})

@patch('core.helper.encrypter.encrypt_token', side_effect=encrypt_side_effect)
def test_encrypt_credentials(mock_encrypt):
    if False:
        return 10
    api_key = 'valid_key'
    result = MODEL_PROVIDER_CLASS.encrypt_provider_credentials('tenant_id', {VALIDATE_CREDENTIAL_KEY: api_key})
    mock_encrypt.assert_called_with('tenant_id', api_key)
    assert result[VALIDATE_CREDENTIAL_KEY] == f'encrypted_{api_key}'

@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_get_credentials_custom(mock_decrypt):
    if False:
        print('Hello World!')
    provider = Provider(id='provider_id', tenant_id='tenant_id', provider_name=PROVIDER_NAME, provider_type=ProviderType.CUSTOM.value, encrypted_config=json.dumps({VALIDATE_CREDENTIAL_KEY: 'encrypted_valid_key'}), is_valid=True)
    model_provider = MODEL_PROVIDER_CLASS(provider=provider)
    result = model_provider.get_provider_credentials()
    assert result[VALIDATE_CREDENTIAL_KEY] == 'valid_key'

@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_get_credentials_custom_str(mock_decrypt):
    if False:
        i = 10
        return i + 15
    '\n    Only the OpenAI provider needs to be compatible with the previous case where the encrypted_config was stored as a plain string.\n\n    :param mock_decrypt:\n    :return:\n    '
    provider = Provider(id='provider_id', tenant_id='tenant_id', provider_name=PROVIDER_NAME, provider_type=ProviderType.CUSTOM.value, encrypted_config='encrypted_valid_key', is_valid=True)
    model_provider = MODEL_PROVIDER_CLASS(provider=provider)
    result = model_provider.get_provider_credentials()
    assert result[VALIDATE_CREDENTIAL_KEY] == 'valid_key'

@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_get_credentials_obfuscated(mock_decrypt):
    if False:
        for i in range(10):
            print('nop')
    openai_api_key = 'valid_key'
    provider = Provider(id='provider_id', tenant_id='tenant_id', provider_name=PROVIDER_NAME, provider_type=ProviderType.CUSTOM.value, encrypted_config=json.dumps({VALIDATE_CREDENTIAL_KEY: f'encrypted_{openai_api_key}'}), is_valid=True)
    model_provider = MODEL_PROVIDER_CLASS(provider=provider)
    result = model_provider.get_provider_credentials(obfuscated=True)
    middle_token = result[VALIDATE_CREDENTIAL_KEY][6:-2]
    assert len(middle_token) == max(len(openai_api_key) - 8, 0)
    assert all((char == '*' for char in middle_token))

@patch('core.model_providers.providers.hosted.hosted_model_providers.openai')
def test_get_credentials_hosted(mock_hosted):
    if False:
        for i in range(10):
            print('nop')
    provider = Provider(id='provider_id', tenant_id='tenant_id', provider_name=PROVIDER_NAME, provider_type=ProviderType.SYSTEM.value, encrypted_config='', is_valid=True)
    model_provider = MODEL_PROVIDER_CLASS(provider=provider)
    mock_hosted.api_key = 'hosted_key'
    result = model_provider.get_provider_credentials()
    assert result[VALIDATE_CREDENTIAL_KEY] == 'hosted_key'