import openai
from openai import APIError, InvalidRequestError
from openai.error import RateLimitError, AuthenticationError
from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.llms.base_llm import BaseLlm

class OpenAi(BaseLlm):

    def __init__(self, api_key, model='gpt-4', temperature=0.6, max_tokens=get_config('MAX_MODEL_TOKEN_LIMIT'), top_p=1, frequency_penalty=0, presence_penalty=0, number_of_results=1):
        if False:
            return 10
        '\n        Args:\n            api_key (str): The OpenAI API key.\n            model (str): The model.\n            temperature (float): The temperature.\n            max_tokens (int): The maximum number of tokens.\n            top_p (float): The top p.\n            frequency_penalty (float): The frequency penalty.\n            presence_penalty (float): The presence penalty.\n            number_of_results (int): The number of results.\n        '
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.number_of_results = number_of_results
        self.api_key = api_key
        openai.api_key = api_key
        openai.api_base = get_config('OPENAI_API_BASE', 'https://api.openai.com/v1')

    def get_source(self):
        if False:
            for i in range(10):
                print('nop')
        return 'openai'

    def get_api_key(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            str: The API key.\n        '
        return self.api_key

    def get_model(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            str: The model.\n        '
        return self.model

    def chat_completion(self, messages, max_tokens=get_config('MAX_MODEL_TOKEN_LIMIT')):
        if False:
            return 10
        '\n        Call the OpenAI chat completion API.\n\n        Args:\n            messages (list): The messages.\n            max_tokens (int): The maximum number of tokens.\n\n        Returns:\n            dict: The response.\n        '
        try:
            response = openai.ChatCompletion.create(n=self.number_of_results, model=self.model, messages=messages, temperature=self.temperature, max_tokens=max_tokens, top_p=self.top_p, frequency_penalty=self.frequency_penalty, presence_penalty=self.presence_penalty)
            content = response.choices[0].message['content']
            return {'response': response, 'content': content}
        except AuthenticationError as auth_error:
            logger.info('OpenAi AuthenticationError:', auth_error)
            return {'error': 'ERROR_AUTHENTICATION', 'message': 'Authentication error please check the api keys: ' + str(auth_error)}
        except RateLimitError as api_error:
            logger.info('OpenAi RateLimitError:', api_error)
            return {'error': 'ERROR_RATE_LIMIT', 'message': 'Openai rate limit exceeded: ' + str(api_error)}
        except InvalidRequestError as invalid_request_error:
            logger.info('OpenAi InvalidRequestError:', invalid_request_error)
            return {'error': 'ERROR_INVALID_REQUEST', 'message': 'Openai invalid request error: ' + str(invalid_request_error)}
        except Exception as exception:
            logger.info('OpenAi Exception:', exception)
            return {'error': 'ERROR_OPENAI', 'message': 'Open ai exception: ' + str(exception)}

    def verify_access_key(self):
        if False:
            print('Hello World!')
        '\n        Verify the access key is valid.\n\n        Returns:\n            bool: True if the access key is valid, False otherwise.\n        '
        try:
            models = openai.Model.list()
            return True
        except Exception as exception:
            logger.info('OpenAi Exception:', exception)
            return False

    def get_models(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the models.\n\n        Returns:\n            list: The models.\n        '
        try:
            models = openai.Model.list()
            models = [model['id'] for model in models['data']]
            models_supported = ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4-32k']
            models = [model for model in models if model in models_supported]
            return models
        except Exception as exception:
            logger.info('OpenAi Exception:', exception)
            return []