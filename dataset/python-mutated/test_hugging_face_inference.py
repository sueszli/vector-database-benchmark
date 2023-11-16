import logging
import unittest
from unittest.mock import patch, MagicMock
import pytest
from haystack.nodes.prompt.invocation_layer.handlers import DefaultTokenStreamingHandler, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer import HFInferenceEndpointInvocationLayer

@pytest.fixture
def mock_get_task():
    if False:
        while True:
            i = 10
    with patch('haystack.nodes.prompt.invocation_layer.hugging_face_inference.get_task') as mock_get_task:
        mock_get_task.return_value = 'text2text-generation'
        yield mock_get_task

@pytest.fixture
def mock_get_task_invalid():
    if False:
        for i in range(10):
            print('nop')
    with patch('haystack.nodes.prompt.invocation_layer.hugging_face_inference.get_task') as mock_get_task:
        mock_get_task.return_value = 'some-nonexistent-type'
        yield mock_get_task

@pytest.mark.unit
def test_default_constructor(mock_auto_tokenizer):
    if False:
        print('Hello World!')
    '\n    Test that the default constructor sets the correct values\n    '
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='some_fake_key')
    assert layer.api_key == 'some_fake_key'
    assert layer.max_length == 100
    assert layer.model_input_kwargs == {}

@pytest.mark.unit
def test_constructor_with_model_kwargs(mock_auto_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that model_kwargs are correctly set in the constructor\n    and that model_kwargs_rejected are correctly filtered out\n    '
    model_kwargs = {'temperature': 0.7, 'do_sample': True, 'stream': True}
    model_kwargs_rejected = {'fake_param': 0.7, 'another_fake_param': 1}
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='some_fake_key', **model_kwargs, **model_kwargs_rejected)
    assert 'temperature' in layer.model_input_kwargs
    assert 'do_sample' in layer.model_input_kwargs
    assert 'stream' in layer.model_input_kwargs
    assert 'fake_param' not in layer.model_input_kwargs
    assert 'another_fake_param' not in layer.model_input_kwargs

@pytest.mark.unit
def test_set_model_max_length(mock_auto_tokenizer):
    if False:
        while True:
            i = 10
    '\n    Test that model max length is set correctly\n    '
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='some_fake_key', model_max_length=2048)
    assert layer.prompt_handler.model_max_length == 2048

@pytest.mark.unit
def test_url(mock_auto_tokenizer):
    if False:
        print('Hello World!')
    '\n    Test that the url is correctly set in the constructor\n    '
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='some_fake_key')
    assert layer.url == 'https://api-inference.huggingface.co/models/google/flan-t5-xxl'
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='https://23445.us-east-1.aws.endpoints.huggingface.cloud', api_key='some_fake_key')
    assert layer.url == 'https://23445.us-east-1.aws.endpoints.huggingface.cloud'

@pytest.mark.unit
def test_invoke_with_no_kwargs(mock_auto_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that invoke raises an error if no prompt is provided\n    '
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='some_fake_key')
    with pytest.raises(ValueError) as e:
        layer.invoke()
        assert e.match('No prompt provided.')

@pytest.mark.unit
def test_invoke_with_stop_words(mock_auto_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test stop words are correctly passed to HTTP POST request\n    '
    stop_words = ['but', 'not', 'bye']
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='fake_key')
    with unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post') as mock_post:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt='Tell me hello', stop_words=stop_words)
    assert mock_post.called
    (_, called_kwargs) = mock_post.call_args
    assert 'stop' in called_kwargs['data']['parameters']
    assert called_kwargs['data']['parameters']['stop'] == stop_words

@pytest.mark.unit
@pytest.mark.parametrize('stream', [True, False])
def test_streaming_stream_param_in_constructor(mock_auto_tokenizer, stream):
    if False:
        i = 10
        return i + 15
    '\n    Test stream parameter is correctly passed to HTTP POST request via constructor\n    '
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='fake_key', stream=stream)
    with unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post') as mock_post:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt='Tell me hello')
    assert mock_post.called
    (_, called_kwargs) = mock_post.call_args
    assert 'stream' in called_kwargs
    assert called_kwargs['stream'] == stream

@pytest.mark.unit
@pytest.mark.parametrize('stream', [True, False])
def test_streaming_stream_param_in_method(mock_auto_tokenizer, stream):
    if False:
        return 10
    '\n    Test stream parameter is correctly passed to HTTP POST request via method\n    '
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='fake_key')
    with unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post') as mock_post:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt='Tell me hello', stream=stream)
    assert mock_post.called
    (_, called_kwargs) = mock_post.call_args
    assert 'stream' in called_kwargs
    assert called_kwargs['stream'] == stream

@pytest.mark.unit
def test_streaming_stream_handler_param_in_constructor(mock_auto_tokenizer):
    if False:
        i = 10
        return i + 15
    '\n    Test stream_handler parameter is correctly passed to HTTP POST request via constructor\n    '
    stream_handler = DefaultTokenStreamingHandler()
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='fake_key', stream_handler=stream_handler)
    with unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post') as mock_post, unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._process_streaming_response') as mock_post_stream:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt='Tell me hello')
    assert mock_post.called
    (_, called_kwargs) = mock_post.call_args
    assert 'stream' in called_kwargs
    assert called_kwargs['stream']
    (called_args, _) = mock_post_stream.call_args
    assert isinstance(called_args[1], TokenStreamingHandler)

@pytest.mark.unit
def test_streaming_no_stream_handler_param_in_constructor(mock_auto_tokenizer):
    if False:
        return 10
    '\n    Test stream_handler parameter is correctly passed to HTTP POST request via constructor\n    '
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='fake_key')
    with unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post') as mock_post:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt='Tell me hello')
    assert mock_post.called
    (_, called_kwargs) = mock_post.call_args
    assert 'stream' in called_kwargs
    assert not called_kwargs['stream']

@pytest.mark.unit
def test_streaming_stream_handler_param_in_method(mock_auto_tokenizer):
    if False:
        while True:
            i = 10
    '\n    Test stream_handler parameter is correctly passed to HTTP POST request via method\n    '
    stream_handler = DefaultTokenStreamingHandler()
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='fake_key')
    with unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post') as mock_post, unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._process_streaming_response') as mock_post_stream:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt='Tell me hello', stream_handler=stream_handler)
    assert mock_post.called
    (called_args, called_kwargs) = mock_post.call_args
    assert 'stream' in called_kwargs
    assert called_kwargs['stream']
    (called_args, called_kwargs) = mock_post_stream.call_args
    assert isinstance(called_args[1], TokenStreamingHandler)

@pytest.mark.unit
def test_streaming_no_stream_handler_param_in_method(mock_auto_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test stream_handler parameter is correctly passed to HTTP POST request via method\n    '
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path='google/flan-t5-xxl', api_key='fake_key')
    with unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post') as mock_post:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt='Tell me hello', stream_handler=None)
    assert mock_post.called
    (_, called_kwargs) = mock_post.call_args
    assert 'stream' in called_kwargs
    assert not called_kwargs['stream']

@pytest.mark.integration
@pytest.mark.parametrize('model_name_or_path', ['google/flan-t5-xxl', 'OpenAssistant/oasst-sft-1-pythia-12b', 'bigscience/bloomz'])
def test_ensure_token_limit_no_resize(model_name_or_path):
    if False:
        for i in range(10):
            print('nop')
    handler = HFInferenceEndpointInvocationLayer('fake_api_key', model_name_or_path, max_length=100)
    prompt = 'This is a test prompt.'
    resized_prompt = handler._ensure_token_limit(prompt)
    assert resized_prompt == prompt

@pytest.mark.integration
@pytest.mark.parametrize('model_name_or_path', ['google/flan-t5-xxl', 'OpenAssistant/oasst-sft-1-pythia-12b', 'bigscience/bloomz'])
def test_ensure_token_limit_resize(caplog, model_name_or_path):
    if False:
        while True:
            i = 10
    handler = HFInferenceEndpointInvocationLayer('fake_api_key', model_name_or_path, max_length=5, model_max_length=10)
    prompt = 'This is a test prompt that will be resized because model_max_length is 10 and max_length is 5.'
    with caplog.at_level(logging.WARN):
        resized_prompt = handler._ensure_token_limit(prompt)
        assert 'The prompt has been truncated' in caplog.text
    assert resized_prompt != prompt
    assert 'This is a test' in resized_prompt and 'because model_max_length is 10 and max_length is 5' not in resized_prompt

@pytest.mark.unit
def test_oasst_prompt_preprocessing(mock_auto_tokenizer):
    if False:
        i = 10
        return i + 15
    model_name = 'OpenAssistant/oasst-sft-1-pythia-12b'
    layer = HFInferenceEndpointInvocationLayer('fake_api_key', model_name)
    with unittest.mock.patch('haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post') as mock_post:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        result = layer.invoke(prompt='Tell me hello')
    assert result == ['Hello']
    assert mock_post.called
    (_, called_kwargs) = mock_post.call_args
    assert called_kwargs['data']['inputs'] == '<|prompter|>Tell me hello<|endoftext|><|assistant|>'

@pytest.mark.unit
def test_invalid_key():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='must be a valid Hugging Face token'):
        HFInferenceEndpointInvocationLayer('', 'irrelevant_model_name')

@pytest.mark.unit
def test_invalid_model():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='cannot be None or empty string'):
        HFInferenceEndpointInvocationLayer('fake_api', '')

@pytest.mark.unit
def test_supports(mock_get_task):
    if False:
        i = 10
        return i + 15
    '\n    Test that supports returns True correctly for HFInferenceEndpointInvocationLayer\n    '
    assert HFInferenceEndpointInvocationLayer.supports('google/flan-t5-xxl', api_key='fake_key')
    assert HFInferenceEndpointInvocationLayer.supports('https://<your-unique-deployment-id>.us-east-1.aws.endpoints.huggingface.cloud', api_key='fake_key')

@pytest.mark.unit
def test_supports_not(mock_get_task_invalid):
    if False:
        return 10
    assert not HFInferenceEndpointInvocationLayer.supports('fake_model', api_key='fake_key')
    assert not HFInferenceEndpointInvocationLayer.supports('google/flan-t5-xxl')
    assert not HFInferenceEndpointInvocationLayer.supports('google/flan-t5-xxl', api_key='')
    assert not HFInferenceEndpointInvocationLayer.supports('google/flan-t5-xxl', api_key=None)
    mock_get_task.side_effect = RuntimeError
    assert not HFInferenceEndpointInvocationLayer.supports('google/flan-t5-xxl', api_key='fake_key')