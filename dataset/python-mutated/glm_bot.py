"""ChatGLM bot"""
from __future__ import annotations
from typing import Any
import zhipuai
from rich import print
from xiaogpt.bot.base_bot import BaseBot, ChatHistoryMixin

class GLMBot(ChatHistoryMixin, BaseBot):
    default_options = {'model': 'chatglm_turbo'}

    def __init__(self, glm_key: str) -> None:
        if False:
            while True:
                i = 10
        self.history = []
        zhipuai.api_key = glm_key

    @classmethod
    def from_config(cls, config):
        if False:
            for i in range(10):
                print('nop')
        return cls(glm_key=config.glm_key)

    def ask(self, query, **options):
        if False:
            i = 10
            return i + 15
        ms = self.get_messages()
        kwargs = {**self.default_options, **options}
        kwargs['prompt'] = ms
        ms.append({'role': 'user', 'content': f'{query}'})
        try:
            r = zhipuai.model_api.sse_invoke(**kwargs)
        except Exception as e:
            print(str(e))
            return
        message = ''
        for i in r.events():
            message += str(i.data)
        self.add_message(query, message)
        print(message)
        return message

    def ask_stream(self, query: str, **options: Any):
        if False:
            while True:
                i = 10
        raise Exception('GLM do not support stream')