import os
import pytest
from examples.getting_started import getting_started
from haystack.schema import Answer, Document

@pytest.mark.parametrize('provider', ['cohere', 'huggingface', 'openai'])
def test_getting_started(provider):
    if False:
        while True:
            i = 10
    if provider == 'anthropic':
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    elif provider == 'cohere':
        api_key = os.environ.get('COHERE_API_KEY', '')
    elif provider == 'huggingface':
        api_key = os.environ.get('HUGGINGFACE_API_KEY', '')
    elif provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY', '')
    if api_key:
        result = getting_started(provider=provider, API_KEY=api_key)
        assert isinstance(result, dict)
        assert type(result['answers'][0]) == Answer
        assert type(result['documents'][0]) == Document