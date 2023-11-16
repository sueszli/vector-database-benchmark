import os
import requests
import json
from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.llms.base_llm import BaseLlm
from superagi.llms.utils.huggingface_utils.tasks import Tasks, TaskParameters
from superagi.llms.utils.huggingface_utils.public_endpoints import ACCOUNT_VERIFICATION_URL

class HuggingFace(BaseLlm):

    def __init__(self, api_key, model=None, end_point=None, task=Tasks.TEXT_GENERATION, **kwargs):
        if False:
            print('Hello World!')
        self.api_key = api_key
        self.model = model
        self.end_point = end_point
        self.task = task
        self.task_params = TaskParameters().get_params(self.task, **kwargs)
        self.headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}

    def get_source(self):
        if False:
            print('Hello World!')
        return 'hugging face'

    def get_api_key(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n            str: The API key.\n        '
        return self.api_key

    def get_model(self):
        if False:
            print('Hello World!')
        '\n        The API needs a POST request with the parameter "inputs".\n\n        Returns:\n            response from the endpoint\n        '
        return self.model

    def get_models(self):
        if False:
            return 10
        '\n        Returns:\n            str: The model.\n        '
        return self.model

    def verify_access_key(self):
        if False:
            i = 10
            return i + 15
        '\n        Verify the access key is valid.\n\n        Returns:\n            bool: True if the access key is valid, False otherwise.\n        '
        response = requests.get(ACCOUNT_VERIFICATION_URL, headers=self.headers)
        return response.status_code == 200

    def chat_completion(self, messages, max_tokens=get_config('MAX_MODEL_TOKEN_LIMIT')):
        if False:
            while True:
                i = 10
        '\n        Call the HuggingFace inference API.\n        Args:\n            messages (list): The messages.\n            max_tokens (int): The maximum number of tokens.\n        Returns:\n            dict: The response.\n        '
        try:
            if isinstance(messages, list):
                messages = messages[0]['content'] + '\nThe response in json schema:'
            params = self.task_params
            if self.task == Tasks.TEXT_GENERATION:
                params['max_new_tokens'] = max_tokens
            params['return_full_text'] = False
            payload = {'inputs': messages, 'parameters': self.task_params, 'options': {'use_cache': False, 'wait_for_model': True}}
            response = requests.post(self.end_point, headers=self.headers, data=json.dumps(payload))
            completion = json.loads(response.content.decode('utf-8'))
            logger.info(f'completion={completion!r}')
            if self.task == Tasks.TEXT_GENERATION:
                content = completion[0]['generated_text']
            else:
                content = completion[0]['answer']
            return {'response': completion, 'content': content}
        except Exception as exception:
            logger.error(f'HF Exception: {exception}')
            return {'error': 'ERROR_HUGGINGFACE', 'message': 'HuggingFace Inference exception', 'details': exception}

    def verify_end_point(self):
        if False:
            while True:
                i = 10
        data = json.dumps({'inputs': 'validating end_point'})
        response = requests.post(self.end_point, headers=self.headers, data=data)
        return response.json()