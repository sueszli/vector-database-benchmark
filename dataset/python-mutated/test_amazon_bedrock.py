from typing import Optional, Type
from unittest.mock import call, patch, MagicMock
import pytest
from haystack.lazy_imports import LazyImport
from haystack.errors import AmazonBedrockConfigurationError
from haystack.nodes.prompt.invocation_layer import AmazonBedrockInvocationLayer
from haystack.nodes.prompt.invocation_layer.amazon_bedrock import AI21LabsJurassic2Adapter, AnthropicClaudeAdapter, BedrockModelAdapter, CohereCommandAdapter, AmazonTitanAdapter, MetaLlama2ChatAdapter
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
        i = 10
        return i + 15
    with patch('haystack.nodes.prompt.invocation_layer.handlers.DefaultPromptHandler') as mock_prompt_handler:
        yield mock_prompt_handler

@pytest.mark.unit
def test_default_constructor(mock_auto_tokenizer, mock_boto3_session):
    if False:
        return 10
    '\n    Test that the default constructor sets the correct values\n    '
    layer = AmazonBedrockInvocationLayer(model_name_or_path='anthropic.claude-v2', max_length=99, aws_access_key_id='some_fake_id', aws_secret_access_key='some_fake_key', aws_session_token='some_fake_token', aws_profile_name='some_fake_profile', aws_region_name='fake_region')
    assert layer.max_length == 99
    assert layer.model_name_or_path == 'anthropic.claude-v2'
    assert layer.prompt_handler is not None
    assert layer.prompt_handler.model_max_length == 4096
    mock_boto3_session.assert_called_once()
    mock_boto3_session.assert_called_with(aws_access_key_id='some_fake_id', aws_secret_access_key='some_fake_key', aws_session_token='some_fake_token', profile_name='some_fake_profile', region_name='fake_region')

@pytest.mark.unit
def test_constructor_prompt_handler_initialized(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    '\n    Test that the constructor sets the prompt_handler correctly, with the correct model_max_length for llama-2\n    '
    layer = AmazonBedrockInvocationLayer(model_name_or_path='anthropic.claude-v2', prompt_handler=mock_prompt_handler)
    assert layer.prompt_handler is not None
    assert layer.prompt_handler.model_max_length == 4096

@pytest.mark.unit
def test_constructor_with_model_kwargs(mock_auto_tokenizer, mock_boto3_session):
    if False:
        i = 10
        return i + 15
    '\n    Test that model_kwargs are correctly set in the constructor\n    '
    model_kwargs = {'temperature': 0.7}
    layer = AmazonBedrockInvocationLayer(model_name_or_path='anthropic.claude-v2', **model_kwargs)
    assert 'temperature' in layer.model_adapter.model_kwargs
    assert layer.model_adapter.model_kwargs['temperature'] == 0.7

@pytest.mark.unit
def test_constructor_with_empty_model_name():
    if False:
        return 10
    '\n    Test that the constructor raises an error when the model_name_or_path is empty\n    '
    with pytest.raises(ValueError, match='cannot be None or empty string'):
        AmazonBedrockInvocationLayer(model_name_or_path='')

@pytest.mark.unit
def test_invoke_with_no_kwargs(mock_auto_tokenizer, mock_boto3_session):
    if False:
        while True:
            i = 10
    '\n    Test invoke raises an error if no prompt is provided\n    '
    layer = AmazonBedrockInvocationLayer(model_name_or_path='anthropic.claude-v2')
    with pytest.raises(ValueError, match='The model anthropic.claude-v2 requires a valid prompt.'):
        layer.invoke()

@pytest.mark.unit
def test_short_prompt_is_not_truncated(mock_boto3_session):
    if False:
        return 10
    '\n    Test that a short prompt is not truncated\n    '
    mock_prompt_text = 'I am a tokenized prompt'
    mock_prompt_tokens = mock_prompt_text.split()
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = mock_prompt_tokens
    max_length_generated_text = 3
    total_model_max_length = 10
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        layer = AmazonBedrockInvocationLayer('anthropic.claude-v2', max_length=max_length_generated_text, model_max_length=total_model_max_length)
        prompt_after_resize = layer._ensure_token_limit(mock_prompt_text)
    assert prompt_after_resize == mock_prompt_text

@pytest.mark.unit
def test_long_prompt_is_truncated(mock_boto3_session):
    if False:
        for i in range(10):
            print('nop')
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
        layer = AmazonBedrockInvocationLayer('anthropic.claude-v2', max_length=max_length_generated_text, model_max_length=total_model_max_length)
        prompt_after_resize = layer._ensure_token_limit(long_prompt_text)
    assert prompt_after_resize == truncated_prompt_text

@pytest.mark.unit
def test_supports_for_valid_aws_configuration():
    if False:
        print('Hello World!')
    mock_session = MagicMock()
    mock_session.client('bedrock').list_foundation_models.return_value = {'modelSummaries': [{'modelId': 'anthropic.claude-v2'}]}
    with patch('haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session', return_value=mock_session):
        supported = AmazonBedrockInvocationLayer.supports(model_name_or_path='anthropic.claude-v2', aws_profile_name='some_real_profile')
    (args, kwargs) = mock_session.client('bedrock').list_foundation_models.call_args
    assert kwargs['byOutputModality'] == 'TEXT'
    assert supported

@pytest.mark.unit
def test_supports_raises_on_invalid_aws_profile_name():
    if False:
        print('Hello World!')
    with patch('boto3.Session') as mock_boto3_session:
        mock_boto3_session.side_effect = BotoCoreError()
        with pytest.raises(AmazonBedrockConfigurationError, match='Failed to initialize the session'):
            AmazonBedrockInvocationLayer.supports(model_name_or_path='anthropic.claude-v2', aws_profile_name='some_fake_profile')

@pytest.mark.unit
def test_supports_for_invalid_bedrock_config():
    if False:
        while True:
            i = 10
    mock_session = MagicMock()
    mock_session.client.side_effect = BotoCoreError()
    with patch('haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session', return_value=mock_session), pytest.raises(AmazonBedrockConfigurationError, match='Could not connect to Amazon Bedrock.'):
        AmazonBedrockInvocationLayer.supports(model_name_or_path='anthropic.claude-v2', aws_profile_name='some_real_profile')

@pytest.mark.unit
def test_supports_for_invalid_bedrock_config_error_on_list_models():
    if False:
        for i in range(10):
            print('nop')
    mock_session = MagicMock()
    mock_session.client('bedrock').list_foundation_models.side_effect = BotoCoreError()
    with patch('haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session', return_value=mock_session), pytest.raises(AmazonBedrockConfigurationError, match='Could not connect to Amazon Bedrock.'):
        AmazonBedrockInvocationLayer.supports(model_name_or_path='anthropic.claude-v2', aws_profile_name='some_real_profile')

@pytest.mark.unit
def test_supports_for_no_aws_params():
    if False:
        for i in range(10):
            print('nop')
    supported = AmazonBedrockInvocationLayer.supports(model_name_or_path='anthropic.claude-v2')
    assert supported == False

@pytest.mark.unit
def test_supports_for_unknown_model():
    if False:
        i = 10
        return i + 15
    supported = AmazonBedrockInvocationLayer.supports(model_name_or_path='unknown_model', aws_profile_name='some_real_profile')
    assert supported == False

@pytest.mark.unit
def test_supports_with_stream_true_for_model_that_supports_streaming():
    if False:
        print('Hello World!')
    mock_session = MagicMock()
    mock_session.client('bedrock').list_foundation_models.return_value = {'modelSummaries': [{'modelId': 'anthropic.claude-v2', 'responseStreamingSupported': True}]}
    with patch('haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session', return_value=mock_session):
        supported = AmazonBedrockInvocationLayer.supports(model_name_or_path='anthropic.claude-v2', aws_profile_name='some_real_profile', stream=True)
        assert supported == True

@pytest.mark.unit
def test_supports_with_stream_true_for_model_that_does_not_support_streaming():
    if False:
        i = 10
        return i + 15
    mock_session = MagicMock()
    mock_session.client('bedrock').list_foundation_models.return_value = {'modelSummaries': [{'modelId': 'ai21.j2-mid-v1', 'responseStreamingSupported': False}]}
    with patch('haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session', return_value=mock_session), pytest.raises(AmazonBedrockConfigurationError, match="The model ai21.j2-mid-v1 doesn't support streaming."):
        AmazonBedrockInvocationLayer.supports(model_name_or_path='ai21.j2-mid-v1', aws_profile_name='some_real_profile', stream=True)

@pytest.mark.unit
@pytest.mark.parametrize('model_name_or_path, expected_model_adapter', [('anthropic.claude-v1', AnthropicClaudeAdapter), ('anthropic.claude-v2', AnthropicClaudeAdapter), ('anthropic.claude-instant-v1', AnthropicClaudeAdapter), ('anthropic.claude-super-v5', AnthropicClaudeAdapter), ('cohere.command-text-v14', CohereCommandAdapter), ('cohere.command-light-text-v14', CohereCommandAdapter), ('cohere.command-text-v21', CohereCommandAdapter), ('ai21.j2-mid-v1', AI21LabsJurassic2Adapter), ('ai21.j2-ultra-v1', AI21LabsJurassic2Adapter), ('ai21.j2-mega-v5', AI21LabsJurassic2Adapter), ('amazon.titan-text-lite-v1', AmazonTitanAdapter), ('amazon.titan-text-express-v1', AmazonTitanAdapter), ('amazon.titan-text-agile-v1', AmazonTitanAdapter), ('amazon.titan-text-lightning-v8', AmazonTitanAdapter), ('meta.llama2-13b-chat-v1', MetaLlama2ChatAdapter), ('meta.llama2-70b-chat-v1', MetaLlama2ChatAdapter), ('meta.llama2-130b-v5', MetaLlama2ChatAdapter), ('unknown_model', None)])
def test_get_model_adapter(model_name_or_path: str, expected_model_adapter: Optional[Type[BedrockModelAdapter]]):
    if False:
        while True:
            i = 10
    '\n    Test that the correct model adapter is returned for a given model_name_or_path\n    '
    model_adapter = AmazonBedrockInvocationLayer.get_model_adapter(model_name_or_path=model_name_or_path)
    assert model_adapter == expected_model_adapter

class TestAnthropicClaudeAdapter:

    def test_prepare_body_with_default_params(self) -> None:
        if False:
            print('Hello World!')
        layer = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': '\n\nHuman: Hello, how are you?\n\nAssistant:', 'max_tokens_to_sample': 99, 'stop_sequences': ['\n\nHuman:']}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        if False:
            i = 10
            return i + 15
        layer = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': '\n\nHuman: Hello, how are you?\n\nAssistant:', 'max_tokens_to_sample': 50, 'stop_sequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'top_p': 0.8, 'top_k': 5}
        body = layer.prepare_body(prompt, temperature=0.7, top_p=0.8, top_k=5, max_tokens_to_sample=50, stop_sequences=['CUSTOM_STOP'], unknown_arg='unknown_value')
        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        if False:
            print('Hello World!')
        layer = AnthropicClaudeAdapter(model_kwargs={'temperature': 0.7, 'top_p': 0.8, 'top_k': 5, 'max_tokens_to_sample': 50, 'stop_sequences': ['CUSTOM_STOP'], 'unknown_arg': 'unknown_value'}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': '\n\nHuman: Hello, how are you?\n\nAssistant:', 'max_tokens_to_sample': 50, 'stop_sequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'top_p': 0.8, 'top_k': 5}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        if False:
            while True:
                i = 10
        layer = AnthropicClaudeAdapter(model_kwargs={'temperature': 0.6, 'top_p': 0.7, 'top_k': 4, 'max_tokens_to_sample': 49, 'stop_sequences': ['CUSTOM_STOP_MODEL_KWARGS']}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': '\n\nHuman: Hello, how are you?\n\nAssistant:', 'max_tokens_to_sample': 50, 'stop_sequences': ['CUSTOM_STOP_MODEL_KWARGS'], 'temperature': 0.7, 'top_p': 0.8, 'top_k': 5}
        body = layer.prepare_body(prompt, temperature=0.7, top_p=0.8, top_k=5, max_tokens_to_sample=50)
        assert body == expected_body

    def test_get_responses(self) -> None:
        if False:
            i = 10
            return i + 15
        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        response_body = {'completion': 'This is a single response.'}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        if False:
            while True:
                i = 10
        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        response_body = {'completion': '\n\t This is a single response.'}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        if False:
            print('Hello World!')
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()
        stream_mock.__iter__.return_value = [{'chunk': {'bytes': b'{"completion": " This"}'}}, {'chunk': {'bytes': b'{"completion": " is"}'}}, {'chunk': {'bytes': b'{"completion": " a"}'}}, {'chunk': {'bytes': b'{"completion": " single"}'}}, {'chunk': {'bytes': b'{"completion": " response."}'}}]
        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received
        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        expected_responses = ['This is a single response.']
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses
        stream_handler_mock.assert_has_calls([call(' This', event_data={'completion': ' This'}), call(' is', event_data={'completion': ' is'}), call(' a', event_data={'completion': ' a'}), call(' single', event_data={'completion': ' single'}), call(' response.', event_data={'completion': ' response.'})])

    def test_get_stream_responses_empty(self) -> None:
        if False:
            i = 10
            return i + 15
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()
        stream_mock.__iter__.return_value = []
        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received
        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        expected_responses = ['']
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses
        stream_handler_mock.assert_not_called()

class TestCohereCommandAdapter:

    def test_prepare_body_with_default_params(self) -> None:
        if False:
            i = 10
            return i + 15
        layer = CohereCommandAdapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'max_tokens': 99}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        if False:
            print('Hello World!')
        layer = CohereCommandAdapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'max_tokens': 50, 'stop_sequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'p': 0.8, 'k': 5, 'return_likelihoods': 'GENERATION', 'stream': True, 'logit_bias': {'token_id': 10.0}, 'num_generations': 1, 'truncate': 'START'}
        body = layer.prepare_body(prompt, temperature=0.7, p=0.8, k=5, max_tokens=50, stop_sequences=['CUSTOM_STOP'], return_likelihoods='GENERATION', stream=True, logit_bias={'token_id': 10.0}, num_generations=1, truncate='START', unknown_arg='unknown_value')
        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        if False:
            return 10
        layer = CohereCommandAdapter(model_kwargs={'temperature': 0.7, 'p': 0.8, 'k': 5, 'max_tokens': 50, 'stop_sequences': ['CUSTOM_STOP'], 'return_likelihoods': 'GENERATION', 'stream': True, 'logit_bias': {'token_id': 10.0}, 'num_generations': 1, 'truncate': 'START', 'unknown_arg': 'unknown_value'}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'max_tokens': 50, 'stop_sequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'p': 0.8, 'k': 5, 'return_likelihoods': 'GENERATION', 'stream': True, 'logit_bias': {'token_id': 10.0}, 'num_generations': 1, 'truncate': 'START'}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        if False:
            print('Hello World!')
        layer = CohereCommandAdapter(model_kwargs={'temperature': 0.6, 'p': 0.7, 'k': 4, 'max_tokens': 49, 'stop_sequences': ['CUSTOM_STOP_MODEL_KWARGS'], 'return_likelihoods': 'ALL', 'stream': False, 'logit_bias': {'token_id': 9.0}, 'num_generations': 2, 'truncate': 'NONE'}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'max_tokens': 50, 'stop_sequences': ['CUSTOM_STOP_MODEL_KWARGS'], 'temperature': 0.7, 'p': 0.8, 'k': 5, 'return_likelihoods': 'GENERATION', 'stream': True, 'logit_bias': {'token_id': 10.0}, 'num_generations': 1, 'truncate': 'START'}
        body = layer.prepare_body(prompt, temperature=0.7, p=0.8, k=5, max_tokens=50, return_likelihoods='GENERATION', stream=True, logit_bias={'token_id': 10.0}, num_generations=1, truncate='START')
        assert body == expected_body

    def test_get_responses(self) -> None:
        if False:
            print('Hello World!')
        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        response_body = {'generations': [{'text': 'This is a single response.'}]}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        if False:
            while True:
                i = 10
        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        response_body = {'generations': [{'text': '\n\t This is a single response.'}]}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_multiple_responses(self) -> None:
        if False:
            return 10
        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        response_body = {'generations': [{'text': 'This is a single response.'}, {'text': 'This is a second response.'}]}
        expected_responses = ['This is a single response.', 'This is a second response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()
        stream_mock.__iter__.return_value = [{'chunk': {'bytes': b'{"text": " This"}'}}, {'chunk': {'bytes': b'{"text": " is"}'}}, {'chunk': {'bytes': b'{"text": " a"}'}}, {'chunk': {'bytes': b'{"text": " single"}'}}, {'chunk': {'bytes': b'{"text": " response."}'}}, {'chunk': {'bytes': b'{"finish_reason": "MAX_TOKENS", "is_finished": true}'}}]
        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received
        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        expected_responses = ['This is a single response.']
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses
        stream_handler_mock.assert_has_calls([call(' This', event_data={'text': ' This'}), call(' is', event_data={'text': ' is'}), call(' a', event_data={'text': ' a'}), call(' single', event_data={'text': ' single'}), call(' response.', event_data={'text': ' response.'}), call('', event_data={'finish_reason': 'MAX_TOKENS', 'is_finished': True})])

    def test_get_stream_responses_empty(self) -> None:
        if False:
            while True:
                i = 10
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()
        stream_mock.__iter__.return_value = []
        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received
        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        expected_responses = ['']
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses
        stream_handler_mock.assert_not_called()

class TestAI21LabsJurrasic2Adapter:

    def test_prepare_body_with_default_params(self) -> None:
        if False:
            return 10
        layer = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'maxTokens': 99}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        layer = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'maxTokens': 50, 'stopSequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'topP': 0.8, 'countPenalty': {'scale': 1.0}, 'presencePenalty': {'scale': 5.0}, 'frequencyPenalty': {'scale': 500.0}, 'numResults': 1}
        body = layer.prepare_body(prompt, maxTokens=50, stopSequences=['CUSTOM_STOP'], temperature=0.7, topP=0.8, countPenalty={'scale': 1.0}, presencePenalty={'scale': 5.0}, frequencyPenalty={'scale': 500.0}, numResults=1, unknown_arg='unknown_value')
        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        if False:
            while True:
                i = 10
        layer = AI21LabsJurassic2Adapter(model_kwargs={'maxTokens': 50, 'stopSequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'topP': 0.8, 'countPenalty': {'scale': 1.0}, 'presencePenalty': {'scale': 5.0}, 'frequencyPenalty': {'scale': 500.0}, 'numResults': 1, 'unknown_arg': 'unknown_value'}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'maxTokens': 50, 'stopSequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'topP': 0.8, 'countPenalty': {'scale': 1.0}, 'presencePenalty': {'scale': 5.0}, 'frequencyPenalty': {'scale': 500.0}, 'numResults': 1}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        if False:
            i = 10
            return i + 15
        layer = AI21LabsJurassic2Adapter(model_kwargs={'maxTokens': 49, 'stopSequences': ['CUSTOM_STOP_MODEL_KWARGS'], 'temperature': 0.6, 'topP': 0.7, 'countPenalty': {'scale': 0.9}, 'presencePenalty': {'scale': 4.0}, 'frequencyPenalty': {'scale': 499.0}, 'numResults': 2, 'unknown_arg': 'unknown_value'}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'maxTokens': 50, 'stopSequences': ['CUSTOM_STOP_MODEL_KWARGS'], 'temperature': 0.7, 'topP': 0.8, 'countPenalty': {'scale': 1.0}, 'presencePenalty': {'scale': 5.0}, 'frequencyPenalty': {'scale': 500.0}, 'numResults': 1}
        body = layer.prepare_body(prompt, temperature=0.7, topP=0.8, maxTokens=50, countPenalty={'scale': 1.0}, presencePenalty={'scale': 5.0}, frequencyPenalty={'scale': 500.0}, numResults=1)
        assert body == expected_body

    def test_get_responses(self) -> None:
        if False:
            return 10
        adapter = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        response_body = {'completions': [{'data': {'text': 'This is a single response.'}}]}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        adapter = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        response_body = {'completions': [{'data': {'text': '\n\t This is a single response.'}}]}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_multiple_responses(self) -> None:
        if False:
            print('Hello World!')
        adapter = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        response_body = {'completions': [{'data': {'text': 'This is a single response.'}}, {'data': {'text': 'This is a second response.'}}]}
        expected_responses = ['This is a single response.', 'This is a second response.']
        assert adapter.get_responses(response_body) == expected_responses

class TestAmazonTitanAdapter:

    def test_prepare_body_with_default_params(self) -> None:
        if False:
            i = 10
            return i + 15
        layer = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'inputText': 'Hello, how are you?', 'textGenerationConfig': {'maxTokenCount': 99}}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        if False:
            i = 10
            return i + 15
        layer = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'inputText': 'Hello, how are you?', 'textGenerationConfig': {'maxTokenCount': 50, 'stopSequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'topP': 0.8}}
        body = layer.prepare_body(prompt, maxTokenCount=50, stopSequences=['CUSTOM_STOP'], temperature=0.7, topP=0.8, unknown_arg='unknown_value')
        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        if False:
            while True:
                i = 10
        layer = AmazonTitanAdapter(model_kwargs={'maxTokenCount': 50, 'stopSequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'topP': 0.8, 'unknown_arg': 'unknown_value'}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'inputText': 'Hello, how are you?', 'textGenerationConfig': {'maxTokenCount': 50, 'stopSequences': ['CUSTOM_STOP'], 'temperature': 0.7, 'topP': 0.8}}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        if False:
            return 10
        layer = AmazonTitanAdapter(model_kwargs={'maxTokenCount': 49, 'stopSequences': ['CUSTOM_STOP_MODEL_KWARGS'], 'temperature': 0.6, 'topP': 0.7}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'inputText': 'Hello, how are you?', 'textGenerationConfig': {'maxTokenCount': 50, 'stopSequences': ['CUSTOM_STOP_MODEL_KWARGS'], 'temperature': 0.7, 'topP': 0.8}}
        body = layer.prepare_body(prompt, temperature=0.7, topP=0.8, maxTokenCount=50)
        assert body == expected_body

    def test_get_responses(self) -> None:
        if False:
            print('Hello World!')
        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        response_body = {'results': [{'outputText': 'This is a single response.'}]}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        if False:
            print('Hello World!')
        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        response_body = {'results': [{'outputText': '\n\t This is a single response.'}]}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_multiple_responses(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        response_body = {'results': [{'outputText': 'This is a single response.'}, {'outputText': 'This is a second response.'}]}
        expected_responses = ['This is a single response.', 'This is a second response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        if False:
            i = 10
            return i + 15
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()
        stream_mock.__iter__.return_value = [{'chunk': {'bytes': b'{"outputText": " This"}'}}, {'chunk': {'bytes': b'{"outputText": " is"}'}}, {'chunk': {'bytes': b'{"outputText": " a"}'}}, {'chunk': {'bytes': b'{"outputText": " single"}'}}, {'chunk': {'bytes': b'{"outputText": " response."}'}}]
        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received
        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        expected_responses = ['This is a single response.']
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses
        stream_handler_mock.assert_has_calls([call(' This', event_data={'outputText': ' This'}), call(' is', event_data={'outputText': ' is'}), call(' a', event_data={'outputText': ' a'}), call(' single', event_data={'outputText': ' single'}), call(' response.', event_data={'outputText': ' response.'})])

    def test_get_stream_responses_empty(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()
        stream_mock.__iter__.return_value = []
        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received
        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        expected_responses = ['']
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses
        stream_handler_mock.assert_not_called()

class TestMetaLlama2ChatAdapter:

    def test_prepare_body_with_default_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        layer = MetaLlama2ChatAdapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'max_gen_len': 99}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        if False:
            return 10
        layer = MetaLlama2ChatAdapter(model_kwargs={}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'max_gen_len': 50, 'temperature': 0.7, 'top_p': 0.8}
        body = layer.prepare_body(prompt, temperature=0.7, top_p=0.8, max_gen_len=50, unknown_arg='unknown_value')
        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        if False:
            print('Hello World!')
        layer = MetaLlama2ChatAdapter(model_kwargs={'temperature': 0.7, 'top_p': 0.8, 'max_gen_len': 50, 'unknown_arg': 'unknown_value'}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'max_gen_len': 50, 'temperature': 0.7, 'top_p': 0.8}
        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        if False:
            return 10
        layer = MetaLlama2ChatAdapter(model_kwargs={'temperature': 0.6, 'top_p': 0.7, 'top_k': 4, 'max_gen_len': 49}, max_length=99)
        prompt = 'Hello, how are you?'
        expected_body = {'prompt': 'Hello, how are you?', 'max_gen_len': 50, 'temperature': 0.7, 'top_p': 0.7}
        body = layer.prepare_body(prompt, temperature=0.7, max_gen_len=50)
        assert body == expected_body

    def test_get_responses(self) -> None:
        if False:
            while True:
                i = 10
        adapter = MetaLlama2ChatAdapter(model_kwargs={}, max_length=99)
        response_body = {'generation': 'This is a single response.'}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        if False:
            return 10
        adapter = MetaLlama2ChatAdapter(model_kwargs={}, max_length=99)
        response_body = {'generation': '\n\t This is a single response.'}
        expected_responses = ['This is a single response.']
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        if False:
            i = 10
            return i + 15
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()
        stream_mock.__iter__.return_value = [{'chunk': {'bytes': b'{"generation": " This"}'}}, {'chunk': {'bytes': b'{"generation": " is"}'}}, {'chunk': {'bytes': b'{"generation": " a"}'}}, {'chunk': {'bytes': b'{"generation": " single"}'}}, {'chunk': {'bytes': b'{"generation": " response."}'}}]
        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received
        adapter = MetaLlama2ChatAdapter(model_kwargs={}, max_length=99)
        expected_responses = ['This is a single response.']
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses
        stream_handler_mock.assert_has_calls([call(' This', event_data={'generation': ' This'}), call(' is', event_data={'generation': ' is'}), call(' a', event_data={'generation': ' a'}), call(' single', event_data={'generation': ' single'}), call(' response.', event_data={'generation': ' response.'})])

    def test_get_stream_responses_empty(self) -> None:
        if False:
            return 10
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()
        stream_mock.__iter__.return_value = []
        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received
        adapter = MetaLlama2ChatAdapter(model_kwargs={}, max_length=99)
        expected_responses = ['']
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses
        stream_handler_mock.assert_not_called()