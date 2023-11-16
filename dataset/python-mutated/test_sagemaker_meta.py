import os
from unittest.mock import patch, MagicMock, Mock
import pytest
from haystack.lazy_imports import LazyImport
from haystack.errors import SageMakerConfigurationError
from haystack.nodes.prompt.invocation_layer import SageMakerMetaInvocationLayer
with LazyImport() as boto3_import:
    from botocore.exceptions import BotoCoreError

@pytest.fixture
def mock_boto3_session():
    if False:
        i = 10
        return i + 15
    with patch('boto3.Session') as mock_client:
        yield mock_client

@pytest.fixture
def mock_prompt_handler():
    if False:
        print('Hello World!')
    with patch('haystack.nodes.prompt.invocation_layer.handlers.DefaultPromptHandler') as mock_prompt_handler:
        yield mock_prompt_handler

@pytest.mark.unit
def test_default_constructor(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the default constructor sets the correct values\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_fake_model', max_length=99, aws_access_key_id='some_fake_id', aws_secret_access_key='some_fake_key', aws_session_token='some_fake_token', aws_profile_name='some_fake_profile', aws_region_name='fake_region')
    assert layer.max_length == 99
    assert layer.model_name_or_path == 'some_fake_model'
    assert layer.prompt_handler is not None
    assert layer.prompt_handler.model_max_length == 4096
    mock_boto3_session.assert_called_once()
    mock_boto3_session.assert_called_with(aws_access_key_id='some_fake_id', aws_secret_access_key='some_fake_key', aws_session_token='some_fake_token', profile_name='some_fake_profile', region_name='fake_region')

@pytest.mark.unit
def test_constructor_prompt_handler_initialized(mock_auto_tokenizer, mock_boto3_session):
    if False:
        return 10
    '\n    Test that the constructor sets the prompt_handler correctly, with the correct model_max_length for llama-2\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_fake_model', prompt_handler=mock_prompt_handler)
    assert layer.prompt_handler is not None
    assert layer.prompt_handler.model_max_length == 4096

@pytest.mark.unit
def test_constructor_with_model_kwargs(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    '\n    Test that model_kwargs are correctly set in the constructor\n    '
    model_kwargs = {'temperature': 0.7}
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_fake_model', **model_kwargs)
    assert 'temperature' in layer.model_input_kwargs
    assert layer.model_input_kwargs['temperature'] == 0.7

@pytest.mark.unit
def test_constructor_with_empty_model_name():
    if False:
        i = 10
        return i + 15
    '\n    Test that the constructor raises an error when the model_name_or_path is empty\n    '
    with pytest.raises(ValueError, match='cannot be None or empty string'):
        SageMakerMetaInvocationLayer(model_name_or_path='')

@pytest.mark.unit
def test_invoke_with_no_kwargs(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test invoke raises an error if no prompt is provided\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_fake_model')
    with pytest.raises(ValueError, match='No valid prompt provided.'):
        layer.invoke()

@pytest.mark.unit
def test_invoke_with_stop_words(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    "\n    SageMakerMetaInvocationLayer does not support stop words. Tests that they'll be ignored\n    "
    stop_words = ['but', 'not', 'bye']
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_model')
    with patch('haystack.nodes.prompt.invocation_layer.SageMakerMetaInvocationLayer._post') as mock_post:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt='Tell me hello', stop_words=stop_words)
    assert mock_post.called
    (_, call_kwargs) = mock_post.call_args
    assert 'stop_words' not in call_kwargs['params']

@pytest.mark.unit
def test_short_prompt_is_not_truncated(mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that a short prompt is not truncated\n    '
    mock_prompt_text = 'I am a tokenized prompt'
    mock_prompt_tokens = mock_prompt_text.split()
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = mock_prompt_tokens
    max_length_generated_text = 3
    total_model_max_length = 10
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        layer = SageMakerMetaInvocationLayer('some_fake_endpoint', max_length=max_length_generated_text, model_max_length=total_model_max_length)
        prompt_after_resize = layer._ensure_token_limit(mock_prompt_text)
    assert prompt_after_resize == mock_prompt_text

@pytest.mark.unit
def test_long_prompt_is_truncated(mock_boto3_session):
    if False:
        print('Hello World!')
    '\n    Test that a long prompt is truncated\n    '
    long_prompt_text = 'I am a tokenized prompt of length eight'
    long_prompt_tokens = long_prompt_text.split()
    truncated_prompt_text = 'I am a tokenized prompt of length'
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = long_prompt_tokens
    mock_tokenizer.convert_tokens_to_string.return_value = truncated_prompt_text
    max_length_generated_text = 3
    total_model_max_length = 10
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        layer = SageMakerMetaInvocationLayer('some_fake_endpoint', max_length=max_length_generated_text, model_max_length=total_model_max_length)
        prompt_after_resize = layer._ensure_token_limit(long_prompt_text)
    assert prompt_after_resize == truncated_prompt_text

@pytest.mark.unit
def test_streaming_init_kwarg(mock_auto_tokenizer, mock_boto3_session):
    if False:
        print('Hello World!')
    '\n    Test stream parameter passed as init kwarg raises an error on layer invocation\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='irrelevant', stream=True)
    with pytest.raises(SageMakerConfigurationError, match='SageMaker model response streaming is not supported yet'):
        layer.invoke(prompt='Tell me hello')

@pytest.mark.unit
def test_streaming_invoke_kwarg(mock_auto_tokenizer, mock_boto3_session):
    if False:
        i = 10
        return i + 15
    '\n    Test stream parameter passed as invoke kwarg raises an error on layer invocation\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='irrelevant')
    with pytest.raises(SageMakerConfigurationError, match='SageMaker model response streaming is not supported yet'):
        layer.invoke(prompt='Tell me hello', stream=True)

@pytest.mark.unit
def test_streaming_handler_init_kwarg(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    '\n    Test stream_handler parameter passed as init kwarg raises an error on layer invocation\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='irrelevant', stream_handler=Mock())
    with pytest.raises(SageMakerConfigurationError, match='SageMaker model response streaming is not supported yet'):
        layer.invoke(prompt='Tell me hello')

@pytest.mark.unit
def test_streaming_handler_invoke_kwarg(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test stream_handler parameter passed as invoke kwarg raises an error on layer invocation\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='irrelevant')
    with pytest.raises(SageMakerConfigurationError, match='SageMaker model response streaming is not supported yet'):
        layer.invoke(prompt='Tell me hello', stream_handler=Mock())

@pytest.mark.unit
def test_supports_for_valid_aws_configuration():
    if False:
        i = 10
        return i + 15
    '\n    Test that the SageMakerMetaInvocationLayer identifies a valid SageMaker Inference endpoint via the supports() method\n    '
    mock_client = MagicMock()
    mock_client.describe_endpoint.return_value = {'EndpointStatus': 'InService'}
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client
    with patch('haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session', return_value=mock_session):
        supported = SageMakerMetaInvocationLayer.supports(model_name_or_path='some_sagemaker_deployed_model', aws_profile_name='some_real_profile', aws_custom_attributes={'accept_eula': True})
    (args, kwargs) = mock_client.describe_endpoint.call_args
    assert kwargs['EndpointName'] == 'some_sagemaker_deployed_model'
    (args, kwargs) = mock_session.client.call_args
    assert args[0] == 'sagemaker-runtime'
    assert supported

@pytest.mark.unit
def test_supports_not_on_invalid_aws_profile_name():
    if False:
        return 10
    '\n    Test that the SageMakerMetaInvocationLayer raises SageMakerConfigurationError when the profile name is invalid\n    '
    with patch('boto3.Session') as mock_boto3_session:
        mock_boto3_session.side_effect = BotoCoreError()
        with pytest.raises(SageMakerConfigurationError, match='Failed to initialize the session'):
            SageMakerMetaInvocationLayer.supports(model_name_or_path='some_fake_model', aws_profile_name='some_fake_profile', aws_custom_attributes={'accept_eula': True})

@pytest.mark.unit
def test_supports_not_on_missing_eula():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the SageMakerMetaInvocationLayer is not supported when the EULA is missing\n    '
    mock_client = MagicMock()
    mock_client.describe_endpoint.return_value = {'EndpointStatus': 'InService'}
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client
    with patch('haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session', return_value=mock_session):
        supported = SageMakerMetaInvocationLayer.supports(model_name_or_path='some_sagemaker_deployed_model', aws_profile_name='some_real_profile')
    assert not supported

@pytest.mark.unit
def test_supports_not_on_eula_not_accepted():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the SageMakerMetaInvocationLayer is not supported when the EULA is not accepted\n    '
    mock_client = MagicMock()
    mock_client.describe_endpoint.return_value = {'EndpointStatus': 'InService'}
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client
    with patch('haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session', return_value=mock_session):
        supported = SageMakerMetaInvocationLayer.supports(model_name_or_path='some_sagemaker_deployed_model', aws_profile_name='some_real_profile', aws_custom_attributes={'accept_eula': False})
    assert not supported

@pytest.mark.unit
def test_format_custom_attributes_with_non_empty_dict():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the SageMakerMetaInvocationLayer correctly formats the custom attributes, attributes specified\n    '
    attributes = {'key1': 'value1', 'key2': 'value2'}
    expected_output = 'key1=value1;key2=value2'
    assert SageMakerMetaInvocationLayer.format_custom_attributes(attributes) == expected_output

@pytest.mark.unit
def test_format_custom_attributes_with_empty_dict():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the SageMakerMetaInvocationLayer correctly formats the custom attributes, attributes not specified\n    '
    attributes = {}
    expected_output = ''
    assert SageMakerMetaInvocationLayer.format_custom_attributes(attributes) == expected_output

@pytest.mark.unit
def test_format_custom_attributes_with_none():
    if False:
        while True:
            i = 10
    '\n    Test that the SageMakerMetaInvocationLayer correctly formats the custom attributes, attributes are None\n    '
    attributes = None
    expected_output = ''
    assert SageMakerMetaInvocationLayer.format_custom_attributes(attributes) == expected_output

@pytest.mark.unit
def test_format_custom_attributes_with_bool_value():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the SageMakerMetaInvocationLayer correctly formats the custom attributes, attributes are bool\n    '
    attributes = {'key1': True, 'key2': False}
    expected_output = 'key1=true;key2=false'
    assert SageMakerMetaInvocationLayer.format_custom_attributes(attributes) == expected_output

@pytest.mark.unit
def test_format_custom_attributes_with_single_bool_value():
    if False:
        print('Hello World!')
    '\n    Test that the SageMakerMetaInvocationLayer correctly formats the custom attributes, attributes are single bool\n    '
    attributes = {'key1': True}
    expected_output = 'key1=true'
    assert SageMakerMetaInvocationLayer.format_custom_attributes(attributes) == expected_output

@pytest.mark.unit
def test_format_custom_attributes_with_int_value():
    if False:
        i = 10
        return i + 15
    '\n    Test that the SageMakerMetaInvocationLayer correctly formats the custom attributes, attributes are ints\n    '
    attributes = {'key1': 1, 'key2': 2}
    expected_output = 'key1=1;key2=2'
    assert SageMakerMetaInvocationLayer.format_custom_attributes(attributes) == expected_output

@pytest.mark.unit
def test_invoke_chat_format(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the SageMakerMetaInvocationLayer accepts a chat in the correct format\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_fake_model')
    prompt = [[{'role': 'user', 'content': 'Hello'}]]
    expected_response = [[{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hello there'}]]
    with patch('haystack.nodes.prompt.invocation_layer.sagemaker_meta.SageMakerMetaInvocationLayer._post') as mock_post:
        mock_post.return_value = expected_response
        layer.invoke(prompt=prompt)

@pytest.mark.unit
def test_invoke_invalid_chat_format(mock_auto_tokenizer, mock_boto3_session):
    if False:
        print('Hello World!')
    '\n    Test that the SageMakerMetaInvocationLayer raises an exception when the chat is in the wrong format\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_fake_model')
    prompt = [{'roe': 'user', 'cotent': 'Hello'}]
    expected_response = [[{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hello there'}]]
    with patch('haystack.nodes.prompt.invocation_layer.sagemaker_meta.SageMakerMetaInvocationLayer._post') as mock_post:
        mock_post.return_value = expected_response
        with pytest.raises(ValueError, match='The prompt format is different than what the model expects'):
            layer.invoke(prompt=prompt)

@pytest.mark.unit
def test_invoke_prompt_string(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the SageMakerMetaInvocationLayer accepts a prompt in the correct string format\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_fake_model')
    with patch('haystack.nodes.prompt.invocation_layer.sagemaker_meta.SageMakerMetaInvocationLayer._post') as mock_post:
        mock_post.return_value = ['Hello there']
        layer.invoke(prompt='Hello')

@pytest.mark.unit
def test_invoke_empty_prompt(mock_auto_tokenizer, mock_boto3_session):
    if False:
        print('Hello World!')
    '\n    Test that the SageMakerMetaInvocationLayer raises an exception when the prompt is empty string\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_fake_model')
    with pytest.raises(ValueError):
        layer.invoke(prompt='')

@pytest.mark.unit
def test_invoke_improper_prompt_type(mock_auto_tokenizer, mock_boto3_session):
    if False:
        i = 10
        return i + 15
    '\n    Test that the SageMakerMetaInvocationLayer raises an exception when the prompt is int instead of str\n    '
    layer = SageMakerMetaInvocationLayer(model_name_or_path='some_fake_model')
    prompt = 123
    with pytest.raises(ValueError):
        layer.invoke(prompt=prompt)

@pytest.mark.skipif(not os.environ.get('TEST_SAGEMAKER_MODEL_ENDPOINT', None), reason='Skipping because SageMaker not configured')
@pytest.mark.integration
def test_supports_triggered_for_valid_sagemaker_endpoint():
    if False:
        while True:
            i = 10
    '\n    Test that the SageMakerMetaInvocationLayer identifies a valid SageMaker Inference endpoint via the supports() method\n    '
    model_name_or_path = os.environ.get('TEST_SAGEMAKER_MODEL_ENDPOINT')
    assert SageMakerMetaInvocationLayer.supports(model_name_or_path=model_name_or_path)

@pytest.mark.skipif(not os.environ.get('TEST_SAGEMAKER_MODEL_ENDPOINT', None), reason='Skipping because SageMaker not configured')
@pytest.mark.integration
def test_supports_not_triggered_for_invalid_iam_profile():
    if False:
        print('Hello World!')
    '\n    Test that the SageMakerMetaInvocationLayer identifies an invalid SageMaker Inference endpoint\n    (in this case because of an invalid IAM AWS Profile via the supports() method)\n    '
    assert not SageMakerMetaInvocationLayer.supports(model_name_or_path='fake_endpoint')
    assert not SageMakerMetaInvocationLayer.supports(model_name_or_path='fake_endpoint', aws_profile_name='invalid-profile')