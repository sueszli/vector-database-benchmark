import json
import os
from unittest.mock import patch, MagicMock
from core.model_providers.models.entity.message import PromptMessage, MessageType
from core.model_providers.models.entity.model_params import ModelKwargs, ModelType
from core.model_providers.models.llm.openllm_model import OpenLLMModel
from core.model_providers.providers.openllm_provider import OpenLLMProvider
from models.provider import Provider, ProviderType, ProviderModel

def get_mock_provider():
    if False:
        return 10
    return Provider(id='provider_id', tenant_id='tenant_id', provider_name='openllm', provider_type=ProviderType.CUSTOM.value, encrypted_config='', is_valid=True)

def get_mock_model(model_name, mocker):
    if False:
        i = 10
        return i + 15
    model_kwargs = ModelKwargs(max_tokens=10, temperature=0.01)
    server_url = os.environ['OPENLLM_SERVER_URL']
    model_provider = OpenLLMProvider(provider=get_mock_provider())
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = ProviderModel(provider_name='openllm', model_name=model_name, model_type=ModelType.TEXT_GENERATION.value, encrypted_config=json.dumps({'server_url': server_url}), is_valid=True)
    mocker.patch('extensions.ext_database.db.session.query', return_value=mock_query)
    return OpenLLMModel(model_provider=model_provider, name=model_name, model_kwargs=model_kwargs)

def decrypt_side_effect(tenant_id, encrypted_api_key):
    if False:
        return 10
    return encrypted_api_key

@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_get_num_tokens(mock_decrypt, mocker):
    if False:
        for i in range(10):
            print('nop')
    model = get_mock_model('facebook/opt-125m', mocker)
    rst = model.get_num_tokens([PromptMessage(type=MessageType.USER, content='Who is your manufacturer?')])
    assert rst == 5

@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_run(mock_decrypt, mocker):
    if False:
        while True:
            i = 10
    mocker.patch('core.model_providers.providers.base.BaseModelProvider.update_last_used', return_value=None)
    model = get_mock_model('facebook/opt-125m', mocker)
    messages = [PromptMessage(content='Human: who are you? \nAnswer: ')]
    rst = model.run(messages)
    assert len(rst.content) > 0