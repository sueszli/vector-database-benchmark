import json
import re
import time
from uuid import uuid1
import datetime
from chatgpt_wrapper import OpenAIAPI
import loguru
import requests
from chatgpt_wrapper import ChatGPT
from pentestgpt.config.chatgpt_config import ChatGPTConfig
logger = loguru.logger

class ChatGPTBrowser:
    """
    The ChatGPT Wrapper based on browser (playwright).
    It keeps the same interface as ChatGPT.
    """

    def __init__(self, model=None):
        if False:
            print('Hello World!')
        config = ChatGPTConfig()
        if model is not None:
            config.set('chat.model', model)
        self.bot = ChatGPT(config)

    def get_authorization(self):
        if False:
            print('Hello World!')
        return

    def get_latest_message_id(self, conversation_id):
        if False:
            i = 10
            return i + 15
        return

    def get_conversation_history(self, limit=20, offset=0):
        if False:
            for i in range(10):
                print('nop')
        return self.bot.get_history(limit, offset)

    def send_new_message(self, message):
        if False:
            while True:
                i = 10
        response = self.bot.ask(message)
        latest_uuid = self.get_conversation_history(limit=1, offset=0).keys()[0]
        return (response, latest_uuid)

    def send_message(self, message, conversation_id):
        if False:
            print('Hello World!')
        return

    def extract_code_fragments(self, text):
        if False:
            i = 10
            return i + 15
        return re.findall('```(.*?)```', text, re.DOTALL)

    def delete_conversation(self, conversation_id=None):
        if False:
            return 10
        if conversation_id is not None:
            self.bot.delete_conversation(conversation_id)
if __name__ == '__main__':
    bot = OpenAIAPI()
    (success, response, message) = bot.ask('Hello, world!')
    if success:
        print(response)
    else:
        raise RuntimeError(message)