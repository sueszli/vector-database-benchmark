import os
from unittest.mock import patch, Mock
from unittest import TestCase
import requests
import json
from superagi.llms.hugging_face import HuggingFace
from superagi.config.config import get_config
from superagi.llms.utils.huggingface_utils.tasks import Tasks, TaskParameters
from superagi.llms.utils.huggingface_utils.public_endpoints import ACCOUNT_VERIFICATION_URL

class TestHuggingFace(TestCase):

    @patch.object(requests, 'get')
    def test_verify_access_key(self, mock_get):
        if False:
            i = 10
            return i + 15
        api_key = 'test_api_key'
        model = 'test_model'
        end_point = 'test_end_point'
        hf_instance = HuggingFace(api_key, model=model, end_point=end_point)
        mock_get.return_value.status_code = 200
        result = hf_instance.verify_access_key()
        mock_get.assert_called_with(ACCOUNT_VERIFICATION_URL, headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
        assert result is True

    @patch.object(requests, 'post')
    def test_verify_end_point(self, mock_post):
        if False:
            for i in range(10):
                print('nop')
        api_key = 'test_api_key'
        model = 'test_model'
        end_point = 'test_end_point'
        hf_instance = HuggingFace(api_key, model=model, end_point=end_point)
        mock_post.return_value.json.return_value = {'valid_response': 'valid'}
        result = hf_instance.verify_end_point()
        mock_post.assert_called_with(end_point, headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}, data=json.dumps({'inputs': 'validating end_point'}))
        assert result == {'valid_response': 'valid'}