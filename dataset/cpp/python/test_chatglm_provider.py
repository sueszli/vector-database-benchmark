import pytest
from unittest.mock import patch
import json

from langchain.schema import LLMResult, Generation, AIMessage, ChatResult, ChatGeneration

from core.model_providers.providers.base import CredentialsValidateFailedError
from core.model_providers.providers.chatglm_provider import ChatGLMProvider
from core.model_providers.providers.spark_provider import SparkProvider
from models.provider import ProviderType, Provider


PROVIDER_NAME = 'chatglm'
MODEL_PROVIDER_CLASS = ChatGLMProvider
VALIDATE_CREDENTIAL = {
    'api_base': 'valid_api_base',
}


def encrypt_side_effect(tenant_id, encrypt_key):
    return f'encrypted_{encrypt_key}'


def decrypt_side_effect(tenant_id, encrypted_key):
    return encrypted_key.replace('encrypted_', '')


def test_is_provider_credentials_valid_or_raise_valid(mocker):
    mocker.patch('langchain.llms.chatglm.ChatGLM._call',
                 return_value="abc")

    MODEL_PROVIDER_CLASS.is_provider_credentials_valid_or_raise(VALIDATE_CREDENTIAL)


def test_is_provider_credentials_valid_or_raise_invalid():
    # raise CredentialsValidateFailedError if api_key is not in credentials
    with pytest.raises(CredentialsValidateFailedError):
        MODEL_PROVIDER_CLASS.is_provider_credentials_valid_or_raise({})

    credential = VALIDATE_CREDENTIAL.copy()
    credential['api_base'] = 'invalid_api_base'

    # raise CredentialsValidateFailedError if api_key is invalid
    with pytest.raises(CredentialsValidateFailedError):
        MODEL_PROVIDER_CLASS.is_provider_credentials_valid_or_raise(credential)


@patch('core.helper.encrypter.encrypt_token', side_effect=encrypt_side_effect)
def test_encrypt_credentials(mock_encrypt):
    result = MODEL_PROVIDER_CLASS.encrypt_provider_credentials('tenant_id', VALIDATE_CREDENTIAL.copy())
    assert result['api_base'] == f'encrypted_{VALIDATE_CREDENTIAL["api_base"]}'


@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_get_credentials_custom(mock_decrypt):
    encrypted_credential = VALIDATE_CREDENTIAL.copy()
    encrypted_credential['api_base'] = 'encrypted_' + encrypted_credential['api_base']

    provider = Provider(
        id='provider_id',
        tenant_id='tenant_id',
        provider_name=PROVIDER_NAME,
        provider_type=ProviderType.CUSTOM.value,
        encrypted_config=json.dumps(encrypted_credential),
        is_valid=True,
    )
    model_provider = MODEL_PROVIDER_CLASS(provider=provider)
    result = model_provider.get_provider_credentials()
    assert result['api_base'] == 'valid_api_base'


@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_get_credentials_obfuscated(mock_decrypt):
    encrypted_credential = VALIDATE_CREDENTIAL.copy()
    encrypted_credential['api_base'] = 'encrypted_' + encrypted_credential['api_base']

    provider = Provider(
        id='provider_id',
        tenant_id='tenant_id',
        provider_name=PROVIDER_NAME,
        provider_type=ProviderType.CUSTOM.value,
        encrypted_config=json.dumps(encrypted_credential),
        is_valid=True,
    )
    model_provider = MODEL_PROVIDER_CLASS(provider=provider)
    result = model_provider.get_provider_credentials(obfuscated=True)
    middle_token = result['api_base'][6:-2]
    assert len(middle_token) == max(len(VALIDATE_CREDENTIAL['api_base']) - 8, 0)
    assert all(char == '*' for char in middle_token)
