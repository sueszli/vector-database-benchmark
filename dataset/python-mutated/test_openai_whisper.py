import json
import os
from unittest.mock import patch
from core.model_providers.models.speech2text.openai_whisper import OpenAIWhisper
from core.model_providers.providers.openai_provider import OpenAIProvider
from models.provider import Provider, ProviderType

def get_mock_provider(valid_openai_api_key):
    if False:
        i = 10
        return i + 15
    return Provider(id='provider_id', tenant_id='tenant_id', provider_name='openai', provider_type=ProviderType.CUSTOM.value, encrypted_config=json.dumps({'openai_api_key': valid_openai_api_key}), is_valid=True)

def get_mock_openai_whisper_model():
    if False:
        while True:
            i = 10
    model_name = 'whisper-1'
    valid_openai_api_key = os.environ['OPENAI_API_KEY']
    openai_provider = OpenAIProvider(provider=get_mock_provider(valid_openai_api_key))
    return OpenAIWhisper(model_provider=openai_provider, name=model_name)

def decrypt_side_effect(tenant_id, encrypted_openai_api_key):
    if False:
        print('Hello World!')
    return encrypted_openai_api_key

@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_run(mock_decrypt):
    if False:
        print('Hello World!')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.join(current_dir, 'audio.mp3')
    model = get_mock_openai_whisper_model()
    with open(audio_file_path, 'rb') as audio_file:
        rst = model.run(audio_file)
    assert isinstance(rst, dict)
    assert rst['text'] == '1, 2, 3, 4, 5, 6, 7, 8, 9, 10'