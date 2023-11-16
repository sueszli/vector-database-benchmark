import google.generativeai as palm
from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.llms.base_llm import BaseLlm

class GooglePalm(BaseLlm):

    def __init__(self, api_key, model='models/chat-bison-001', temperature=0.6, candidate_count=1, top_k=40, top_p=0.95):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            api_key (str): The Google PALM API key.\n            model (str): The model.\n            temperature (float): The temperature.\n            candidate_count (int): The number of candidates.\n            top_k (int): The top k.\n            top_p (float): The top p.\n        '
        self.model = model
        self.temperature = temperature
        self.candidate_count = candidate_count
        self.top_k = top_k
        self.top_p = top_p
        self.api_key = api_key
        palm.configure(api_key=api_key)

    def get_source(self):
        if False:
            while True:
                i = 10
        return 'google palm'

    def get_api_key(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            str: The API key.\n        '
        return self.api_key

    def get_model(self):
        if False:
            while True:
                i = 10
        '\n        Returns:\n            str: The model.\n        '
        return self.model

    def chat_completion(self, messages, max_tokens=get_config('MAX_MODEL_TOKEN_LIMIT') or 800, examples=[], context=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Call the Google PALM chat API.\n\n        Args:\n            context (str): The context.\n            examples (list): The examples.\n            messages (list): The messages.\n\n        Returns:\n            dict: The response.\n        '
        prompt = '\n'.join(['`' + message['role'] + '`: ' + message['content'] + '' for message in messages])
        if len(messages) == 1:
            prompt = messages[0]['content']
        try:
            final_model = 'models/text-bison-001' if self.model == 'models/chat-bison-001' else self.model
            completion = palm.generate_text(model=final_model, temperature=self.temperature, candidate_count=self.candidate_count, top_k=self.top_k, top_p=self.top_p, prompt=prompt, max_output_tokens=int(max_tokens))
            return {'response': completion, 'content': completion.result}
        except Exception as exception:
            logger.info('Google palm Exception:', exception)
            return {'error': 'ERROR_GOOGLE_PALM', 'message': 'Google palm exception'}

    def verify_access_key(self):
        if False:
            return 10
        '\n        Verify the access key is valid.\n\n        Returns:\n            bool: True if the access key is valid, False otherwise.\n        '
        try:
            models = palm.list_models()
            return True
        except Exception as exception:
            logger.info('Google palm Exception:', exception)
            return False

    def get_models(self):
        if False:
            print('Hello World!')
        '\n        Get the models.\n\n        Returns:\n            list: The models.\n        '
        try:
            models_supported = ['chat-bison-001']
            return models_supported
        except Exception as exception:
            logger.info('Google palm Exception:', exception)
            return []