from typing import List
import tiktoken
from superagi.types.common import BaseMessage
from superagi.lib.logger import logger
from superagi.models.models import Models
from sqlalchemy.orm import Session

class TokenCounter:

    def __init__(self, session: Session=None, organisation_id: int=None):
        if False:
            for i in range(10):
                print('nop')
        self.session = session
        self.organisation_id = organisation_id

    def token_limit(self, model: str='gpt-3.5-turbo-0301') -> int:
        if False:
            i = 10
            return i + 15
        '\n        Function to return the token limit for a given model.\n\n        Args:\n            model (str): The model to return the token limit for.\n\n        Raises:\n            KeyError: If the model is not found.\n\n        Returns:\n            int: The token limit.\n        '
        try:
            model_token_limit_dict = Models.fetch_model_tokens(self.session, self.organisation_id)
            return model_token_limit_dict[model]
        except KeyError:
            logger.warning('Warning: model not found. Using cl100k_base encoding.')
            return 8092

    @staticmethod
    def count_message_tokens(messages: List[BaseMessage], model: str='gpt-3.5-turbo-0301') -> int:
        if False:
            i = 10
            return i + 15
        '\n        Function to count the number of tokens in a list of messages.\n\n        Args:\n            messages (List[BaseMessage]): The list of messages to count the tokens for.\n            model (str): The model to count the tokens for.\n\n        Raises:\n            KeyError: If the model is not found.\n\n        Returns:\n            int: The number of tokens in the messages.\n        '
        try:
            default_tokens_per_message = 4
            model_token_per_message_dict = {'gpt-3.5-turbo-0301': 4, 'gpt-4-0314': 3, 'gpt-3.5-turbo': 4, 'gpt-4': 3, 'gpt-3.5-turbo-16k': 4, 'gpt-4-32k': 3, 'gpt-4-32k-0314': 3, 'models/chat-bison-001': 4}
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning('Warning: model not found. Using cl100k_base encoding.')
            encoding = tiktoken.get_encoding('cl100k_base')
        if model in model_token_per_message_dict.keys():
            tokens_per_message = model_token_per_message_dict[model]
        else:
            tokens_per_message = default_tokens_per_message
        if tokens_per_message is None:
            raise NotImplementedError(f'num_tokens_from_messages() is not implemented for model {model}.\n See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.')
        num_tokens = 0
        for message in messages:
            if isinstance(message, str):
                message = {'content': message}
            num_tokens += tokens_per_message
            num_tokens += len(encoding.encode(message['content']))
        num_tokens += 3
        print('tokens', num_tokens)
        return num_tokens

    @staticmethod
    def count_text_tokens(message: str) -> int:
        if False:
            return 10
        '\n        Function to count the number of tokens in a text.\n\n        Args:\n            message (str): The text to count the tokens for.\n\n        Returns:\n            int: The number of tokens in the text.\n        '
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(message)) + 4
        return num_tokens