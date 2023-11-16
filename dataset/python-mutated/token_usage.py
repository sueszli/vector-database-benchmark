import tiktoken
import logging
from dataclasses import dataclass
from typing import List, Union
from langchain.callbacks.openai_info import get_openai_token_cost_for_model
from langchain.schema import AIMessage, HumanMessage, SystemMessage
Message = Union[AIMessage, HumanMessage, SystemMessage]
logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """
    Represents token usage statistics for a conversation step.
    """
    step_name: str
    in_step_prompt_tokens: int
    in_step_completion_tokens: int
    in_step_total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int

class Tokenizer:
    """
    Tokenizer for counting tokens in text.
    """

    def __init__(self, model_name):
        if False:
            while True:
                i = 10
        self.model_name = model_name
        self._tiktoken_tokenizer = tiktoken.encoding_for_model(model_name) if 'gpt-4' in model_name or 'gpt-3.5' in model_name else tiktoken.get_encoding('cl100k_base')

    def num_tokens(self, txt: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the number of tokens in a text.\n\n        Parameters\n        ----------\n        txt : str\n            The text to count the tokens in.\n\n        Returns\n        -------\n        int\n            The number of tokens in the text.\n        '
        return len(self._tiktoken_tokenizer.encode(txt))

    def num_tokens_from_messages(self, messages: List[Message]) -> int:
        if False:
            print('Hello World!')
        '\n        Get the total number of tokens used by a list of messages.\n\n        Parameters\n        ----------\n        messages : List[Message]\n            The list of messages to count the tokens in.\n\n        Returns\n        -------\n        int\n            The total number of tokens used by the messages.\n        '
        n_tokens = 0
        for message in messages:
            n_tokens += 4
            n_tokens += self.num_tokens(message.content)
        n_tokens += 2
        return n_tokens

class TokenUsageLog:
    """
    Represents a log of token usage statistics for a conversation.
    """

    def __init__(self, model_name):
        if False:
            while True:
                i = 10
        self.model_name = model_name
        self._cumulative_prompt_tokens = 0
        self._cumulative_completion_tokens = 0
        self._cumulative_total_tokens = 0
        self._log = []
        self._tokenizer = Tokenizer(model_name)

    def update_log(self, messages: List[Message], answer: str, step_name: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Update the token usage log with the number of tokens used in the current step.\n\n        Parameters\n        ----------\n        messages : List[Message]\n            The list of messages in the conversation.\n        answer : str\n            The answer from the AI.\n        step_name : str\n            The name of the step.\n        '
        prompt_tokens = self._tokenizer.num_tokens_from_messages(messages)
        completion_tokens = self._tokenizer.num_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens
        self._cumulative_prompt_tokens += prompt_tokens
        self._cumulative_completion_tokens += completion_tokens
        self._cumulative_total_tokens += total_tokens
        self._log.append(TokenUsage(step_name=step_name, in_step_prompt_tokens=prompt_tokens, in_step_completion_tokens=completion_tokens, in_step_total_tokens=total_tokens, total_prompt_tokens=self._cumulative_prompt_tokens, total_completion_tokens=self._cumulative_completion_tokens, total_tokens=self._cumulative_total_tokens))

    def log(self) -> List[TokenUsage]:
        if False:
            return 10
        '\n        Get the token usage log.\n\n        Returns\n        -------\n        List[TokenUsage]\n            A log of token usage details per step in the conversation.\n        '
        return self._log

    def format_log(self) -> str:
        if False:
            return 10
        '\n        Format the token usage log as a CSV string.\n\n        Returns\n        -------\n        str\n            The token usage log formatted as a CSV string.\n        '
        result = 'step_name,prompt_tokens_in_step,completion_tokens_in_step,total_tokens_in_step,total_prompt_tokens,total_completion_tokens,total_tokens\n'
        for log in self._log:
            result += f'{log.step_name},{log.in_step_prompt_tokens},{log.in_step_completion_tokens},{log.in_step_total_tokens},{log.total_prompt_tokens},{log.total_completion_tokens},{log.total_tokens}\n'
        return result

    def usage_cost(self) -> float:
        if False:
            return 10
        '\n        Return the total cost in USD of the API usage.\n\n        Returns\n        -------\n        float\n            Cost in USD.\n        '
        result = 0
        for log in self.log():
            result += get_openai_token_cost_for_model(self.model_name, log.total_prompt_tokens, is_completion=False)
            result += get_openai_token_cost_for_model(self.model_name, log.total_completion_tokens, is_completion=True)
        return result