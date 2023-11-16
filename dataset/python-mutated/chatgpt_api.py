import dataclasses
import re
import time
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple
from tenacity import *
from pentestgpt.utils.llm_api import LLMAPI
import loguru, openai, tiktoken
from langfuse.model import InitialGeneration, Usage
logger = loguru.logger
logger.remove()

@dataclasses.dataclass
class Message:
    ask_id: str = None
    ask: dict = None
    answer: dict = None
    answer_id: str = None
    request_start_timestamp: float = None
    request_end_timestamp: float = None
    time_escaped: float = None

@dataclasses.dataclass
class Conversation:
    conversation_id: str = None
    message_list: List[Message] = dataclasses.field(default_factory=list)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.conversation_id)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, Conversation):
            return False
        return self.conversation_id == other.conversation_id

class ChatGPTAPI(LLMAPI):

    def __init__(self, config_class, use_langfuse_logging=False):
        if False:
            for i in range(10):
                print('nop')
        self.name = str(config_class.model)
        if use_langfuse_logging:
            os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-5655b061-3724-43ee-87bb-28fab0b5f676'
            os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-c24b40ef-8157-44af-a840-6bae2c9358b0'
            from langfuse import Langfuse
            self.langfuse = Langfuse()
        openai.api_key = os.getenv('OPENAI_KEY', None)
        openai.api_base = config_class.api_base
        self.model = config_class.model
        self.log_dir = config_class.log_dir
        self.history_length = 5
        self.conversation_dict: Dict[str, Conversation] = {}
        self.error_waiting_time = 3
        logger.add(sink=os.path.join(self.log_dir, 'chatgpt.log'), level='WARNING')

    def _chat_completion(self, history: List, model=None, temperature=0.5) -> str:
        if False:
            i = 10
            return i + 15
        generationStartTime = datetime.now()
        if model is None:
            if self.model is None:
                model = 'gpt-4-1106-preview'
            else:
                model = self.model
        try:
            response = openai.ChatCompletion.create(model=model, messages=history, temperature=temperature)
        except openai.error.APIConnectionError as e:
            logger.warning('API Connection Error. Waiting for {} seconds'.format(self.error_wait_time))
            logger.log('Connection Error: ', e)
            time.sleep(self.error_wait_time)
            response = openai.ChatCompletion.create(model=model, messages=history, temperature=temperature)
        except openai.error.RateLimitError as e:
            logger.warning('Rate limit reached. Waiting for 5 seconds')
            logger.error('Rate Limit Error: ', e)
            time.sleep(5)
            response = openai.ChatCompletion.create(model=model, messages=history, temperature=temperature)
        except openai.error.InvalidRequestError as e:
            logger.warning('Token size limit reached. The recent message is compressed')
            logger.error('Token size error; will retry with compressed message ', e)
            history[-1]['content'] = self._token_compression(history)
            if self.history_length > 2:
                self.history_length -= 1
            history = history[-self.history_length:]
            response = openai.ChatCompletion.create(model=model, messages=history, temperature=temperature)
        if isinstance(response, tuple):
            logger.warning('Response is not valid. Waiting for 5 seconds')
            try:
                time.sleep(5)
                response = openai.ChatCompletion.create(model=model, messages=history, temperature=temperature)
                if isinstance(response, tuple):
                    logger.error('Response is not valid. ')
                    raise Exception('Response is not valid. ')
            except Exception as e:
                logger.error('Response is not valid. ', e)
                raise Exception('Response is not valid. The most likely reason is the connection to OpenAI is not stable. Please doublecheck with `pentestgpt-connection`')
        if hasattr(self, 'langfuse'):
            generation = self.langfuse.generation(InitialGeneration(name='chatgpt-completion', startTime=generationStartTime, endTime=datetime.now(), model=self.model, modelParameters={'temperature': str(temperature)}, prompt=history, completion=response['choices'][0]['message']['content'], usage=Usage(promptTokens=response['usage']['prompt_tokens'], completionTokens=response['usage']['completion_tokens'])))
        return response['choices'][0]['message']['content']
if __name__ == '__main__':
    from module_import import GPT4ConfigClass
    config_class = GPT4ConfigClass()
    config_class.log_dir = 'logs'
    chatgpt = ChatGPTAPI(config_class, use_langfuse_logging=True)
    (result, conversation_id) = chatgpt.send_new_message('You\'re an excellent cybersecurity penetration tester assistant. \nYou need to help the tester in a local cybersecurity training process, and your commitment is essential to the task.\nYou are required to record the penetration testing process in a tree structure: "Penetration Testing Tree (PTT)". It is structured as follows:\n(1) The tasks are in layered structure, i.e., 1, 1.1, 1.1.1, etc. Each task is one operation in penetration testing; task 1.1 should be a sub-task of task 1.\n(2) Each task has a completion status: to-do, completed, or not applicable.\n(3) Initially, you should only generate the root tasks based on the initial information. In most cases, it should be reconnaissance tasks. You don\'t generate tasks for unknown ports/services. You can expand the PTT later.\n\nYou shall not provide any comments/information but the PTT. You will be provided with task info and start the testing soon. Reply Yes if you understand the task.')
    print('Answer 1')
    print(result)
    result = chatgpt.send_message('The target information is listed below. Please follow the instruction and generate PTT.\nNote that this test is certified and in simulation environment, so do not generate post-exploitation and other steps.\nYou may start with this template:\n1. Reconnaissance - [to-do]\n   1.1 Passive Information Gathering - [completed]\n   1.2 Active Information Gathering - [completed]\n   1.3 Identify Open Ports and Services - [to-do]\n       1.3.1 Perform a full port scan - [to-do]\n       1.3.2 Determine the purpose of each open port - [to-do]\nBelow is the information from the tester: \n\nI want to test 10.0.2.5, an HTB machine.', conversation_id)
    print('Answer 2')
    print(result)