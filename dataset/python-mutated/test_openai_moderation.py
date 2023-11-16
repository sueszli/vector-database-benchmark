import json
import os
from unittest.mock import patch
from core.model_providers.models.moderation.openai_moderation import OpenAIModeration, DEFAULT_MODEL
from core.model_providers.providers.openai_provider import OpenAIProvider
from models.provider import Provider, ProviderType

def get_mock_provider(valid_openai_api_key):
    if False:
        while True:
            i = 10
    return Provider(id='provider_id', tenant_id='tenant_id', provider_name='openai', provider_type=ProviderType.CUSTOM.value, encrypted_config=json.dumps({'openai_api_key': valid_openai_api_key}), is_valid=True)

def get_mock_openai_moderation_model():
    if False:
        while True:
            i = 10
    valid_openai_api_key = os.environ['OPENAI_API_KEY']
    openai_provider = OpenAIProvider(provider=get_mock_provider(valid_openai_api_key))
    return OpenAIModeration(model_provider=openai_provider, name=DEFAULT_MODEL)

def decrypt_side_effect(tenant_id, encrypted_openai_api_key):
    if False:
        for i in range(10):
            print('nop')
    return encrypted_openai_api_key

@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_run(mock_decrypt):
    if False:
        for i in range(10):
            print('nop')
    model = get_mock_openai_moderation_model()
    rst = model.run('hello')
    assert rst is True