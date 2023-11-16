from __future__ import annotations
import logging
import re
from typing import Optional, Dict, Any
from haystack import Answer
from haystack.errors import AgentError
logger = logging.getLogger(__name__)

class AgentStep:
    """
    The AgentStep class represents a single step in the execution of an agent.
    """

    def __init__(self, current_step: int=1, max_steps: int=10, final_answer_pattern: Optional[str]=None, prompt_node_response: str='', transcript: str=''):
        if False:
            return 10
        '\n        :param current_step: The current step in the execution of the agent.\n        :param max_steps: The maximum number of steps the agent can execute.\n        :param final_answer_pattern: The regex pattern to extract the final answer from the PromptNode response. If no\n        pattern is provided, entire prompt node response is considered the final answer.\n        :param prompt_node_response: The PromptNode response received.\n        text it generated during execution up to this step. The transcript is used to generate the next prompt.\n        '
        self.current_step = current_step
        self.max_steps = max_steps
        self.final_answer_pattern = final_answer_pattern or '^([\\s\\S]+)$'
        self.prompt_node_response = prompt_node_response
        self.transcript = transcript

    def create_next_step(self, prompt_node_response: Any, current_step: Optional[int]=None) -> AgentStep:
        if False:
            i = 10
            return i + 15
        '\n        Creates the next agent step based on the current step and the PromptNode response.\n        :param prompt_node_response: The PromptNode response received.\n        :param current_step: The current step in the execution of the agent.\n        '
        if not isinstance(prompt_node_response, list) or not prompt_node_response:
            raise AgentError(f'Agent output must be a non-empty list of str, but {prompt_node_response} received. Transcript:\n{self.transcript}')
        cls = type(self)
        return cls(current_step=current_step if current_step else self.current_step + 1, max_steps=self.max_steps, final_answer_pattern=self.final_answer_pattern, prompt_node_response=prompt_node_response[0], transcript=self.transcript)

    def final_answer(self, query: str) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        "\n        Formats an answer as a dict containing `query` and `answers` similar to the output of a Pipeline.\n        The full transcript based on the Agent's initial prompt template and the text it generated during execution.\n\n        :param query: The search query\n        "
        answer: Dict[str, Any] = {'query': query, 'answers': [Answer(answer='', type='generative')], 'transcript': self.transcript}
        if self.current_step > self.max_steps:
            logger.warning('Maximum number of iterations (%s) reached for query (%s). Increase max_steps or no answer can be provided for this query.', self.max_steps, query)
        else:
            final_answer = self.parse_final_answer()
            if not final_answer:
                logger.warning('Final answer parser (%s) could not parse PromptNode response (%s).', self.final_answer_pattern, self.prompt_node_response)
            else:
                answer = {'query': query, 'answers': [Answer(answer=final_answer, type='generative')], 'transcript': self.transcript}
        return answer

    def is_last(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Check if this is the last step of the Agent.\n        :return: True if this is the last step of the Agent, False otherwise.\n        '
        return bool(self.parse_final_answer()) or self.current_step > self.max_steps

    def completed(self, observation: Optional[str]) -> None:
        if False:
            return 10
        '\n        Update the transcript with the observation\n        :param observation: received observation from the Agent environment.\n        '
        self.transcript += f'{self.prompt_node_response}\nObservation: {observation}\nThought:' if observation else self.prompt_node_response

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Return a string representation of the AgentStep object.\n\n        :return: A string that represents the AgentStep object.\n        '
        return f'AgentStep(current_step={self.current_step}, max_steps={self.max_steps}, prompt_node_response={self.prompt_node_response}, final_answer_pattern={self.final_answer_pattern}, transcript={self.transcript})'

    def parse_final_answer(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        "\n        Parse the final answer from the response of the prompt node.\n\n        This function searches the prompt node's response for a match with the\n        pre-defined final answer pattern. If a match is found, it's returned as the\n        final answer after removing leading/trailing quotes and whitespaces.\n        If no match is found, it returns None.\n\n        :return: The final answer as a string if a match is found, otherwise None.\n        "
        final_answer_match = re.search(self.final_answer_pattern, self.prompt_node_response)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            return final_answer.strip('" ')
        else:
            return None