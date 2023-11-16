import copy
from unittest.mock import patch
import pytest
from tenacity import wait_none
from haystack.errors import OpenAIError, OpenAIRateLimitError, OpenAIUnauthorizedError
from haystack.utils.openai_utils import openai_request, _openai_text_completion_tokenization_details, check_openai_policy_violation

@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt_default():
    if False:
        print('Hello World!')
    (tokenizer_name, max_tokens_limit) = _openai_text_completion_tokenization_details(model_name='not-recognized-name')
    assert tokenizer_name == 'gpt2'
    assert max_tokens_limit == 2049

@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt_davinci():
    if False:
        return 10
    (tokenizer_name, max_tokens_limit) = _openai_text_completion_tokenization_details(model_name='text-davinci-003')
    assert tokenizer_name == 'p50k_base'
    assert max_tokens_limit == 4097

@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt3_5_azure():
    if False:
        while True:
            i = 10
    (tokenizer_name, max_tokens_limit) = _openai_text_completion_tokenization_details(model_name='gpt-35-turbo')
    assert tokenizer_name == 'cl100k_base'
    assert max_tokens_limit == 4096

@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt3_5():
    if False:
        print('Hello World!')
    (tokenizer_name, max_tokens_limit) = _openai_text_completion_tokenization_details(model_name='gpt-3.5-turbo')
    assert tokenizer_name == 'cl100k_base'
    assert max_tokens_limit == 4096

@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt_4():
    if False:
        i = 10
        return i + 15
    (tokenizer_name, max_tokens_limit) = _openai_text_completion_tokenization_details(model_name='gpt-4')
    assert tokenizer_name == 'cl100k_base'
    assert max_tokens_limit == 8192

@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt_4_32k():
    if False:
        print('Hello World!')
    (tokenizer_name, max_tokens_limit) = _openai_text_completion_tokenization_details(model_name='gpt-4-32k')
    assert tokenizer_name == 'cl100k_base'
    assert max_tokens_limit == 32768

@pytest.mark.unit
@patch('haystack.utils.openai_utils.requests')
def test_openai_request_retries_generic_error(mock_requests):
    if False:
        print('Hello World!')
    mock_requests.request.return_value.status_code = 418
    with pytest.raises(OpenAIError):
        openai_request.retry_with(wait=wait_none())(url='some_url', headers={}, payload={}, read_response=False)
    assert mock_requests.request.call_count == 5

@pytest.mark.unit
@patch('haystack.utils.openai_utils.requests')
def test_openai_request_retries_on_rate_limit_error(mock_requests):
    if False:
        while True:
            i = 10
    mock_requests.request.return_value.status_code = 429
    with pytest.raises(OpenAIRateLimitError):
        openai_request.retry_with(wait=wait_none())(url='some_url', headers={}, payload={}, read_response=False)
    assert mock_requests.request.call_count == 5

@pytest.mark.unit
@patch('haystack.utils.openai_utils.requests')
def test_openai_request_does_not_retry_on_unauthorized_error(mock_requests):
    if False:
        while True:
            i = 10
    mock_requests.request.return_value.status_code = 401
    with pytest.raises(OpenAIUnauthorizedError):
        openai_request.retry_with(wait=wait_none())(url='some_url', headers={}, payload={}, read_response=False)
    assert mock_requests.request.call_count == 1

@pytest.mark.unit
@patch('haystack.utils.openai_utils.requests')
def test_openai_request_does_not_retry_on_success(mock_requests):
    if False:
        while True:
            i = 10
    mock_requests.request.return_value.status_code = 200
    openai_request.retry_with(wait=wait_none())(url='some_url', headers={}, payload={}, read_response=False)
    assert mock_requests.request.call_count == 1

@pytest.mark.unit
def test_check_openai_policy_violation():
    if False:
        while True:
            i = 10
    moderation_endpoint_mock_response_flagged = {'id': 'modr-7Ok9zndoeSn5ij654vuNCgFVomU4U', 'model': 'text-moderation-004', 'results': [{'flagged': True, 'categories': {'sexual': False, 'hate': False, 'violence': True, 'self-harm': True, 'sexual/minors': False, 'hate/threatening': False, 'violence/graphic': False}, 'category_scores': {'sexual': 2.6659495e-06, 'hate': 1.9359974e-05, 'violence': 0.95964026, 'self-harm': 0.9696306, 'sexual/minors': 4.1061935e-07, 'hate/threatening': 4.9856953e-07, 'violence/graphic': 0.2683866}}]}
    moderation_endpoint_mock_response_not_flagged = copy.deepcopy(moderation_endpoint_mock_response_flagged)
    moderation_endpoint_mock_response_not_flagged['results'][0]['flagged'] = False
    moderation_endpoint_mock_response_not_flagged['results'][0]['categories'].update({'violence': False, 'self-harm': False})
    with patch('haystack.utils.openai_utils.openai_request') as mock_openai_request:
        mock_openai_request.return_value = moderation_endpoint_mock_response_flagged
        assert check_openai_policy_violation(input='violent input', headers={}) == True
        mock_openai_request.return_value = moderation_endpoint_mock_response_not_flagged
        assert check_openai_policy_violation(input='ok input', headers={}) == False