import os
from unittest.mock import patch, MagicMock, Mock
import pytest
from haystack.lazy_imports import LazyImport
from haystack.errors import SageMakerConfigurationError
from haystack.nodes.prompt.invocation_layer import SageMakerHFInferenceInvocationLayer
with LazyImport() as boto3_import:
    from botocore.exceptions import BotoCoreError

@pytest.fixture
def mock_boto3_session():
    if False:
        return 10
    with patch('boto3.Session') as mock_client:
        yield mock_client

@pytest.fixture
def mock_prompt_handler():
    if False:
        return 10
    with patch('haystack.nodes.prompt.invocation_layer.handlers.DefaultPromptHandler') as mock_prompt_handler:
        yield mock_prompt_handler

@pytest.mark.unit
def test_default_constructor(mock_auto_tokenizer, mock_boto3_session):
    if False:
        return 10
    '\n    Test that the default constructor sets the correct values\n    '
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='some_fake_model', max_length=99, aws_access_key_id='some_fake_id', aws_secret_access_key='some_fake_key', aws_session_token='some_fake_token', aws_profile_name='some_fake_profile', aws_region_name='fake_region')
    assert layer.max_length == 99
    assert layer.model_name_or_path == 'some_fake_model'
    mock_boto3_session.assert_called_once()
    mock_boto3_session.assert_called_with(aws_access_key_id='some_fake_id', aws_secret_access_key='some_fake_key', aws_session_token='some_fake_token', profile_name='some_fake_profile', region_name='fake_region')

@pytest.mark.unit
def test_constructor_with_model_kwargs(mock_auto_tokenizer, mock_boto3_session):
    if False:
        i = 10
        return i + 15
    '\n    Test that model_kwargs are correctly set in the constructor\n    and that model_kwargs_rejected are correctly filtered out\n    '
    model_kwargs = {'temperature': 0.7, 'do_sample': True, 'stream': True}
    model_kwargs_rejected = {'fake_param': 0.7, 'another_fake_param': 1}
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='some_fake_model', **model_kwargs, **model_kwargs_rejected)
    assert layer.model_input_kwargs['temperature'] == 0.7
    assert 'do_sample' in layer.model_input_kwargs
    assert 'fake_param' not in layer.model_input_kwargs
    assert 'another_fake_param' not in layer.model_input_kwargs

@pytest.mark.unit
def test_invoke_with_no_kwargs(mock_auto_tokenizer, mock_boto3_session):
    if False:
        i = 10
        return i + 15
    '\n    Test that invoke raises an error if no prompt is provided\n    '
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='some_fake_model')
    with pytest.raises(ValueError, match='No prompt provided.'):
        layer.invoke()

@pytest.mark.unit
def test_invoke_with_stop_words(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    '\n    Test stop words are correctly passed to HTTP POST request\n    '
    stop_words = ['but', 'not', 'bye']
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='some_model', api_key='fake_key')
    with patch('haystack.nodes.prompt.invocation_layer.SageMakerHFInferenceInvocationLayer._post') as mock_post:
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt='Tell me hello', stop_words=stop_words)
    assert mock_post.called
    (_, call_kwargs) = mock_post.call_args
    assert call_kwargs['params']['stopping_criteria'] == stop_words

@pytest.mark.unit
def test_short_prompt_is_not_truncated(mock_boto3_session):
    if False:
        while True:
            i = 10
    mock_prompt_text = 'I am a tokenized prompt'
    mock_prompt_tokens = mock_prompt_text.split()
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = mock_prompt_tokens
    max_length_generated_text = 3
    total_model_max_length = 10
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        layer = SageMakerHFInferenceInvocationLayer('some_fake_endpoint', max_length=max_length_generated_text, model_max_length=total_model_max_length)
        prompt_after_resize = layer._ensure_token_limit(mock_prompt_text)
    assert prompt_after_resize == mock_prompt_text

@pytest.mark.unit
def test_long_prompt_is_truncated(mock_boto3_session):
    if False:
        print('Hello World!')
    long_prompt_text = 'I am a tokenized prompt of length eight'
    long_prompt_tokens = long_prompt_text.split()
    truncated_prompt_text = 'I am a tokenized prompt of length'
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = long_prompt_tokens
    mock_tokenizer.convert_tokens_to_string.return_value = truncated_prompt_text
    max_length_generated_text = 3
    total_model_max_length = 10
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        layer = SageMakerHFInferenceInvocationLayer('some_fake_endpoint', max_length=max_length_generated_text, model_max_length=total_model_max_length)
        prompt_after_resize = layer._ensure_token_limit(long_prompt_text)
    assert prompt_after_resize == truncated_prompt_text

@pytest.mark.unit
def test_empty_model_name():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError, match='cannot be None or empty string'):
        SageMakerHFInferenceInvocationLayer(model_name_or_path='')

@pytest.mark.unit
def test_streaming_init_kwarg(mock_auto_tokenizer, mock_boto3_session):
    if False:
        i = 10
        return i + 15
    '\n    Test stream parameter passed as init kwarg is correctly logged as not supported\n    '
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant', stream=True)
    with pytest.raises(SageMakerConfigurationError, match='SageMaker model response streaming is not supported yet'):
        layer.invoke(prompt='Tell me hello')

@pytest.mark.unit
def test_streaming_invoke_kwarg(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test stream parameter passed as invoke kwarg is correctly logged as not supported\n    '
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    with pytest.raises(SageMakerConfigurationError, match='SageMaker model response streaming is not supported yet'):
        layer.invoke(prompt='Tell me hello', stream=True)

@pytest.mark.unit
def test_streaming_handler_init_kwarg(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    '\n    Test stream_handler parameter passed as init kwarg is correctly logged as not supported\n    '
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant', stream_handler=Mock())
    with pytest.raises(SageMakerConfigurationError, match='SageMaker model response streaming is not supported yet'):
        layer.invoke(prompt='Tell me hello')

@pytest.mark.unit
def test_streaming_handler_invoke_kwarg(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    '\n    Test stream_handler parameter passed as invoke kwarg is correctly logged as not supported\n    '
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    with pytest.raises(SageMakerConfigurationError, match='SageMaker model response streaming is not supported yet'):
        layer.invoke(prompt='Tell me hello', stream_handler=Mock())

@pytest.mark.unit
def test_supports_for_valid_aws_configuration():
    if False:
        while True:
            i = 10
    '\n    Test that the SageMakerHFInferenceInvocationLayer identifies a valid SageMaker Inference endpoint\n    via the supports() method\n    '
    mock_client = MagicMock()
    mock_client.describe_endpoint.return_value = {'EndpointStatus': 'InService'}
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client
    with patch('haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session', return_value=mock_session):
        supported = SageMakerHFInferenceInvocationLayer.supports(model_name_or_path='some_sagemaker_deployed_model', aws_profile_name='some_real_profile')
    (args, kwargs) = mock_client.describe_endpoint.call_args
    assert kwargs['EndpointName'] == 'some_sagemaker_deployed_model'
    (args, kwargs) = mock_session.client.call_args
    assert args[0] == 'sagemaker-runtime'
    assert supported

@pytest.mark.unit
def test_supports_not_on_invalid_aws_profile_name():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the SageMakerHFInferenceInvocationLayer raises SageMakerConfigurationError when the profile name is invalid\n    '
    with patch('boto3.Session') as mock_boto3_session:
        mock_boto3_session.side_effect = BotoCoreError()
        with pytest.raises(SageMakerConfigurationError, match='Failed to initialize the session'):
            SageMakerHFInferenceInvocationLayer.supports(model_name_or_path='some_fake_model', aws_profile_name='some_fake_profile')

@pytest.mark.skipif(not os.environ.get('TEST_SAGEMAKER_MODEL_ENDPOINT', None), reason='Skipping because SageMaker not configured')
@pytest.mark.integration
def test_supports_triggered_for_valid_sagemaker_endpoint():
    if False:
        print('Hello World!')
    '\n    Test that the SageMakerHFInferenceInvocationLayer identifies a valid SageMaker Inference endpoint via the supports() method\n    '
    model_name_or_path = os.environ.get('TEST_SAGEMAKER_MODEL_ENDPOINT')
    assert SageMakerHFInferenceInvocationLayer.supports(model_name_or_path=model_name_or_path)

@pytest.mark.skipif(not os.environ.get('TEST_SAGEMAKER_MODEL_ENDPOINT', None), reason='Skipping because SageMaker not configured')
@pytest.mark.integration
def test_supports_not_triggered_for_invalid_iam_profile():
    if False:
        print('Hello World!')
    '\n    Test that the SageMakerHFInferenceInvocationLayer identifies an invalid SageMaker Inference endpoint\n    (in this case because of an invalid IAM AWS Profile via the supports() method)\n    '
    assert not SageMakerHFInferenceInvocationLayer.supports(model_name_or_path='fake_endpoint')
    assert not SageMakerHFInferenceInvocationLayer.supports(model_name_or_path='fake_endpoint', aws_profile_name='invalid-profile')

@pytest.mark.unit
def test_dolly_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        return 10
    response = {'generated_texts': ['Berlin']}
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin']

@pytest.mark.unit
def test_dolly_multiple_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        return 10
    response = {'generated_texts': ['Berlin', 'More elaborate Berlin']}
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin', 'More elaborate Berlin']

@pytest.mark.unit
def test_flan_t5_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    response = {'generated_texts': ['berlin']}
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['berlin']

@pytest.mark.unit
def test_gpt_j_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    response = [[{'generated_text': 'Berlin'}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin']

@pytest.mark.unit
def test_gpt_j_multiple_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    response = [[{'generated_text': 'Berlin'}, {'generated_text': 'Berlin 2'}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin', 'Berlin 2']

@pytest.mark.unit
def test_mpt_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        i = 10
        return i + 15
    response = [[{'generated_text': 'Berlin'}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin']

@pytest.mark.unit
def test_mpt_multiple_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        print('Hello World!')
    response = [[{'generated_text': 'Berlin'}, {'generated_text': 'Berlin 2'}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin', 'Berlin 2']

@pytest.mark.unit
def test_open_llama_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        return 10
    response = {'generated_texts': ['Berlin']}
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin']

@pytest.mark.unit
def test_open_llama_multiple_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    response = {'generated_texts': ['Berlin', 'Berlin 2']}
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin', 'Berlin 2']

@pytest.mark.unit
def test_pajama_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    response = [[{'generated_text': ['Berlin']}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin']

@pytest.mark.unit
def test_pajama_multiple_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    response = [[{'generated_text': 'Berlin'}, {'generated_text': 'Berlin 2'}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin', 'Berlin 2']

@pytest.mark.unit
def test_flan_ul2_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
    response = {'generated_texts': ['Berlin']}
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin']

@pytest.mark.unit
def test_flan_ul2_multiple_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    response = {'generated_texts': ['Berlin', 'Berlin 2']}
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin', 'Berlin 2']

@pytest.mark.unit
def test_gpt_neo_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        print('Hello World!')
    response = [[{'generated_text': 'Berlin'}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin']

@pytest.mark.unit
def test_gpt_neo_multiple_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    response = [[{'generated_text': 'Berlin'}, {'generated_text': 'Berlin 2'}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin', 'Berlin 2']

@pytest.mark.unit
def test_bloomz_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        i = 10
        return i + 15
    response = [[{'generated_text': 'Berlin'}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin']

@pytest.mark.unit
def test_bloomz_multiple_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    response = [[{'generated_text': 'Berlin'}, {'generated_text': 'Berlin 2'}]]
    layer = SageMakerHFInferenceInvocationLayer(model_name_or_path='irrelevant')
    assert layer._extract_response(response) == ['Berlin', 'Berlin 2']