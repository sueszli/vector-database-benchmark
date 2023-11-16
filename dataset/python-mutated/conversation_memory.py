import collections
from typing import OrderedDict, List, Optional, Any, Dict
from haystack.agents.memory import Memory

class ConversationMemory(Memory):
    """
    A memory class that stores conversation history.
    """

    def __init__(self, input_key: str='input', output_key: str='output'):
        if False:
            while True:
                i = 10
        '\n        Initialize ConversationMemory with input and output keys.\n\n        :param input_key: The key to use for storing user input.\n        :param output_key: The key to use for storing model output.\n        '
        self.list: List[OrderedDict] = []
        self.input_key = input_key
        self.output_key = output_key

    def load(self, keys: Optional[List[str]]=None, **kwargs) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Load conversation history as a formatted string.\n\n        :param keys: Optional list of keys (ignored in this implementation).\n        :param kwargs: Optional keyword arguments\n            - window_size: integer specifying the number of most recent conversation snippets to load.\n        :return: A formatted string containing the conversation history.\n        '
        chat_transcript = ''
        window_size = kwargs.get('window_size', None)
        if window_size is not None:
            chat_list = self.list[-window_size:]
        else:
            chat_list = self.list
        for chat_snippet in chat_list:
            chat_transcript += f"Human: {chat_snippet['Human']}\n"
            chat_transcript += f"AI: {chat_snippet['AI']}\n"
        return chat_transcript

    def save(self, data: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        '\n        Save a conversation snippet to memory.\n\n        :param data: A dictionary containing the conversation snippet to save.\n        '
        chat_snippet = collections.OrderedDict()
        chat_snippet['Human'] = data[self.input_key]
        chat_snippet['AI'] = data[self.output_key]
        self.list.append(chat_snippet)

    def clear(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Clear the conversation history.\n        '
        self.list = []