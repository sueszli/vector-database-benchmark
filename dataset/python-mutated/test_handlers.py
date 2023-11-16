from unittest.mock import patch
import pytest
from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler

@pytest.mark.unit
def test_prompt_handler_positive():
    if False:
        return 10
    mock_tokens = ['I', 'am', 'a', 'tokenized', 'prompt']
    mock_prompt = 'I am a tokenized prompt'
    with patch('haystack.nodes.prompt.invocation_layer.handlers.AutoTokenizer.from_pretrained', autospec=True) as mock_tokenizer:
        tokenizer_instance = mock_tokenizer.return_value
        tokenizer_instance.tokenize.return_value = mock_tokens
        tokenizer_instance.convert_tokens_to_string.return_value = mock_prompt
        prompt_handler = DefaultPromptHandler('model_path', 10, 3)
        result = prompt_handler(mock_prompt)
    assert result == {'resized_prompt': mock_prompt, 'prompt_length': 5, 'new_prompt_length': 5, 'model_max_length': 10, 'max_length': 3}

@pytest.mark.unit
def test_prompt_handler_negative():
    if False:
        return 10
    mock_tokens = ['I', 'am', 'a', 'tokenized', 'prompt', 'of', 'length', 'eight']
    mock_prompt = 'I am a tokenized prompt of length'
    with patch('haystack.nodes.prompt.invocation_layer.handlers.AutoTokenizer.from_pretrained', autospec=True) as mock_tokenizer:
        tokenizer_instance = mock_tokenizer.return_value
        tokenizer_instance.tokenize.return_value = mock_tokens
        tokenizer_instance.convert_tokens_to_string.return_value = mock_prompt
        prompt_handler = DefaultPromptHandler('model_path', 10, 3)
        result = prompt_handler(mock_prompt)
    assert result == {'resized_prompt': mock_prompt, 'prompt_length': 8, 'new_prompt_length': 7, 'model_max_length': 10, 'max_length': 3}

@pytest.mark.unit
@patch('haystack.nodes.prompt.invocation_layer.handlers.AutoTokenizer.from_pretrained')
def test_prompt_handler_model_max_length_set_in_tokenizer(mock_tokenizer):
    if False:
        while True:
            i = 10
    prompt_handler = DefaultPromptHandler(model_name_or_path='model_path', model_max_length=10, max_length=3)
    assert prompt_handler.tokenizer.model_max_length == 10

@pytest.mark.integration
def test_prompt_handler_basics():
    if False:
        for i in range(10):
            print('nop')
    handler = DefaultPromptHandler(model_name_or_path='gpt2', model_max_length=20, max_length=10)
    assert callable(handler)
    handler = DefaultPromptHandler(model_name_or_path='gpt2', model_max_length=20)
    assert handler.max_length == 100
    assert handler.tokenizer.model_max_length == 20

@pytest.mark.integration
def test_gpt2_prompt_handler():
    if False:
        while True:
            i = 10
    handler = DefaultPromptHandler(model_name_or_path='gpt2', model_max_length=20, max_length=10)
    assert handler('This is a test') == {'prompt_length': 4, 'resized_prompt': 'This is a test', 'max_length': 10, 'model_max_length': 20, 'new_prompt_length': 4}
    assert handler('This is a prompt that will be resized because it is longer than allowed') == {'prompt_length': 15, 'resized_prompt': 'This is a prompt that will be resized because', 'max_length': 10, 'model_max_length': 20, 'new_prompt_length': 10}

@pytest.mark.integration
def test_flan_prompt_handler_no_resize():
    if False:
        return 10
    handler = DefaultPromptHandler(model_name_or_path='google/flan-t5-xxl', model_max_length=20, max_length=10)
    assert handler('This is a test') == {'prompt_length': 5, 'resized_prompt': 'This is a test', 'max_length': 10, 'model_max_length': 20, 'new_prompt_length': 5}

@pytest.mark.integration
def test_flan_prompt_handler_resize():
    if False:
        return 10
    handler = DefaultPromptHandler(model_name_or_path='google/flan-t5-xxl', model_max_length=20, max_length=10)
    assert handler('This is a prompt that will be resized because it is longer than allowed') == {'prompt_length': 17, 'resized_prompt': 'This is a prompt that will be re', 'max_length': 10, 'model_max_length': 20, 'new_prompt_length': 10}

@pytest.mark.integration
def test_flan_prompt_handler_empty_string():
    if False:
        print('Hello World!')
    handler = DefaultPromptHandler(model_name_or_path='google/flan-t5-xxl', model_max_length=20, max_length=10)
    assert handler('') == {'prompt_length': 0, 'resized_prompt': '', 'max_length': 10, 'model_max_length': 20, 'new_prompt_length': 0}

@pytest.mark.integration
def test_flan_prompt_handler_none():
    if False:
        i = 10
        return i + 15
    handler = DefaultPromptHandler(model_name_or_path='google/flan-t5-xxl', model_max_length=20, max_length=10)
    assert handler(None) == {'prompt_length': 0, 'resized_prompt': None, 'max_length': 10, 'model_max_length': 20, 'new_prompt_length': 0}