import json
import tiktoken
from lwe.core.config import Config
from lwe.core.logger import Logger
from lwe.core import util

class TokenManager:
    """Manage functions in a cache."""

    def __init__(self, config, provider, model_name, function_cache):
        if False:
            while True:
                i = 10
        'Initialize the function cache.'
        self.config = config or Config()
        self.log = Logger(self.__class__.__name__, self.config)
        self.provider = provider
        self.model_name = model_name
        self.function_cache = function_cache

    def get_token_encoding(self):
        if False:
            return 10
        '\n        Get token encoding for a model.\n\n        :raises NotImplementedError: If unsupported model\n        :raises Exception: If error getting encoding\n        :returns: Encoding object\n        :rtype: Encoding\n        '
        if self.model_name not in self.provider.available_models:
            raise NotImplementedError(f'Unsupported model: {self.model_name}')
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        except Exception as err:
            raise Exception(f'Unable to get token encoding for model {self.model_name}: {str(err)}') from err
        return encoding

    def get_num_tokens_from_messages(self, messages, encoding=None):
        if False:
            while True:
                i = 10
        '\n        Get number of tokens for a list of messages.\n\n        :param messages: List of messages\n        :type messages: list\n        :param encoding: Encoding to use, defaults to None to auto-detect\n        :type encoding: Encoding, optional\n        :returns: Number of tokens\n        :rtype: int\n        '
        if not encoding:
            encoding = self.get_token_encoding()
        num_tokens = 0
        messages = self.function_cache.add_message_functions(messages)
        messages = util.transform_messages_to_chat_messages(messages)
        for message in messages:
            num_tokens += 4
            for (key, value) in message.items():
                if isinstance(value, dict):
                    value = json.dumps(value, indent=2)
                num_tokens += len(encoding.encode(value))
                if key == 'name':
                    num_tokens += -1
        num_tokens += 2
        if len(self.function_cache.functions) > 0:
            functions = [self.function_cache.function_manager.get_function_config(function_name) for function_name in self.function_cache.functions]
            functions_string = json.dumps(functions, indent=2)
            num_tokens += len(encoding.encode(functions_string))
        return num_tokens