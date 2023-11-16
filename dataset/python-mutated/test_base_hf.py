"""Unit tests for the base huggingface LLM class"""
import pytest
import requests
from pandasai.exceptions import LLMResponseHTTPError
from pandasai.llm.base import HuggingFaceLLM
from pandasai.prompts import AbstractPrompt

class TestBaseHfLLM:
    """Unit tests for the huggingface LLM class"""

    @pytest.fixture
    def api_response(self):
        if False:
            while True:
                i = 10
        return [{'generated_text': 'Some text'}]

    @pytest.fixture
    def api_response_401(self):
        if False:
            print('Hello World!')
        return {'error': 'Authorization header is correct, but the token seems invalid'}

    @pytest.fixture
    def prompt(self):
        if False:
            return 10

        class MockAbstractPrompt(AbstractPrompt):
            template: str = 'instruction'
        return MockAbstractPrompt()

    def test_type(self):
        if False:
            i = 10
            return i + 15
        assert HuggingFaceLLM(api_token='test_token').type == 'huggingface-llm'

    def test_api_url(self):
        if False:
            for i in range(10):
                print('nop')
        assert HuggingFaceLLM(api_token='test_token')._api_url == 'https://api-inference.huggingface.co/models/'

    def test_query(self, mocker, api_response):
        if False:
            while True:
                i = 10
        response_mock = mocker.Mock()
        response_mock.status_code = 200
        response_mock.json.return_value = api_response
        mocker.patch('requests.post', return_value=response_mock)
        llm = HuggingFaceLLM(api_token='test_token')
        payload = {'inputs': 'Some input text'}
        result = llm.query(payload)
        requests.post.assert_called_once_with(llm._api_url, headers={'Authorization': 'Bearer test_token'}, json=payload, timeout=60)
        assert result == api_response[0]['generated_text']

    def test_query_http_error_401(self, mocker, api_response_401):
        if False:
            return 10
        response_mock = mocker.Mock()
        response_mock.status_code = 401
        response_mock.json.return_value = api_response_401
        mocker.patch('requests.post', return_value=response_mock)
        llm = HuggingFaceLLM(api_token='test_token')
        payload = {'inputs': 'Some input text'}
        with pytest.raises(LLMResponseHTTPError) as exc:
            llm.query(payload)
        assert api_response_401.get('error') in str(exc.value)
        requests.post.assert_called_once_with(llm._api_url, headers={'Authorization': 'Bearer test_token'}, json=payload, timeout=60)

    def test_call(self, mocker, prompt):
        if False:
            while True:
                i = 10
        huggingface = HuggingFaceLLM(api_token='test_token')
        mocker.patch.object(huggingface, 'call', return_value='Generated text')
        result = huggingface.call(prompt, 'value', 'suffix')
        assert result == 'Generated text'

    def test_call_removes_original_prompt(self, mocker):
        if False:
            while True:
                i = 10
        huggingface = HuggingFaceLLM(api_token='test_token')

        class MockAbstractPrompt(AbstractPrompt):
            template: str = 'instruction '
        instruction = MockAbstractPrompt()
        suffix = 'suffix '
        mocker.patch.object(huggingface, 'query', return_value='instruction suffix generated text')
        result = huggingface.call(instruction, suffix)
        assert result == 'generated text'