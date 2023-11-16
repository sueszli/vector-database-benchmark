from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.llms.base_llm import BaseLlm
from superagi.helper.llm_loader import LLMLoader

class LocalLLM(BaseLlm):

    def __init__(self, temperature=0.6, max_tokens=get_config('MAX_MODEL_TOKEN_LIMIT'), top_p=1, frequency_penalty=0, presence_penalty=0, number_of_results=1, model=None, api_key='EMPTY', context_length=4096):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            model (str): The model.\n            temperature (float): The temperature.\n            max_tokens (int): The maximum number of tokens.\n            top_p (float): The top p.\n            frequency_penalty (float): The frequency penalty.\n            presence_penalty (float): The presence penalty.\n            number_of_results (int): The number of results.\n        '
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.number_of_results = number_of_results
        self.context_length = context_length
        llm_loader = LLMLoader(self.context_length)
        self.llm_model = llm_loader.model
        self.llm_grammar = llm_loader.grammar

    def chat_completion(self, messages, max_tokens=get_config('MAX_MODEL_TOKEN_LIMIT')):
        if False:
            while True:
                i = 10
        '\n        Call the chat completion.\n\n        Args:\n            messages (list): The messages.\n            max_tokens (int): The maximum number of tokens.\n\n        Returns:\n            dict: The response.\n        '
        try:
            if self.llm_model is None or self.llm_grammar is None:
                logger.error('Model not found.')
                return {'error': 'Model loading error', 'message': 'Model not found. Please check your model path and try again.'}
            else:
                response = self.llm_model.create_chat_completion(messages=messages, functions=None, function_call=None, temperature=self.temperature, top_p=self.top_p, max_tokens=int(max_tokens), presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty, grammar=self.llm_grammar)
                content = response['choices'][0]['message']['content']
                logger.info(content)
                return {'response': response, 'content': content}
        except Exception as exception:
            logger.info('Exception:', exception)
            return {'error': 'ERROR', 'message': 'Error: ' + str(exception)}

    def get_source(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the source.\n\n        Returns:\n            str: The source.\n        '
        return 'Local LLM'

    def get_api_key(self):
        if False:
            while True:
                i = 10
        '\n        Returns:\n            str: The API key.\n        '
        return self.api_key

    def get_model(self):
        if False:
            print('Hello World!')
        '\n        Returns:\n            str: The model.\n        '
        return self.model

    def get_models(self):
        if False:
            print('Hello World!')
        '\n        Returns:\n            list: The models.\n        '
        return self.model

    def verify_access_key(self, api_key):
        if False:
            return 10
        return True