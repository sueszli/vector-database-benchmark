import jinja2
import logging
from typing import Dict, List, Tuple
from azure.ai.generative.synthetic.simulator._model_tools import RetryClient
from .conversation_turn import ConversationTurn
from .constants import ConversationRole

class DummyConversationBot:

    def __init__(self, role: ConversationRole, conversation_template: str, instantiation_parameters: Dict[str, str]):
        if False:
            return 10
        '\n        Create a ConversationBot with specific name, persona and a sentence that can be used as a conversation starter.\n\n        Parameters\n        ----------\n        role: The role of the bot in the conversation, either USER or ASSISTANT\n        model: The LLM model to use for generating responses\n        conversation_template: A jinja2 template that describes the conversation, this is used to generate the prompt for the LLM\n        instantiation_parameters: A dictionary of parameters that are used to instantiate the conversation template\n            Dedicated parameters:\n                - conversation_starter: A sentence that can be used as a conversation starter, if not provided,\n                    the first turn will be generated using the LLM\n        '
        self.role = role
        self.conversation_template: jinja2.Template = jinja2.Template(conversation_template, undefined=jinja2.StrictUndefined)
        self.persona_template_args = instantiation_parameters
        if self.role == ConversationRole.USER:
            self.name = self.persona_template_args.get('name', role.value)
        else:
            self.name = self.persona_template_args.get('chatbot_name', role.value) or 'Dummy'
        self.logger = logging.getLogger(repr(self))
        if role == ConversationRole.USER:
            if 'conversation_starter' in self.persona_template_args:
                self.logger.info(f'''This simulated bot will use the provided conversation starter "{repr(self.persona_template_args['conversation_starter'])[:400]}"instead of generating a turn using a LLM''')
                self.conversation_starter = self.persona_template_args['conversation_starter']
            else:
                self.logger.info('This simulated bot will generate the first turn as no conversation starter is provided')
                self.conversation_starter = None
        self.userMessages = ['Find the temperature in seattle and add it to the doc', 'what is the weight of an airplane', 'how may grams are there in a ton', 'what is the height of eiffel tower', 'where do you come from', 'what is the current time']

    async def generate_response(self, session: RetryClient, conversation_history: List[ConversationTurn], max_history: int, turn_number: int=0) -> Tuple[dict, dict, int, dict]:
        """
        Prompt the ConversationBot for a response.

        Parameters
        ----------
        session: The aiohttp session to use for the request.
        conversation_history: The turns in the conversation so far.
        request_params: Parameters used to query GPT-4 model.

        Returns
        -------
        response: The response from the ConversationBot.
        time_taken: The time taken to generate the response.
        full_response: The full response from the model.
        """
        if turn_number == 0 and self.conversation_starter is not None and (self.conversation_starter != ''):
            self.logger.info(f'Returning conversation starter: {self.conversation_starter}')
            time_taken = 0
            samples = [self.conversation_starter]
            finish_reason = ['stop']
            parsed_response = {'samples': samples, 'finish_reason': finish_reason, 'id': None}
            full_response = parsed_response
            return (parsed_response, {}, time_taken, full_response)
        prompt = self.conversation_template.render(conversation_turns=conversation_history[-max_history:], role=self.role.value, **self.persona_template_args)
        messages = [{'role': 'system', 'content': prompt}]
        if self.role == ConversationRole.USER:
            messages.extend([turn.to_openai_chat_format(reverse=True) for turn in conversation_history[-max_history:]])
            prompt_role = ConversationRole.USER.value
            response_data = {'id': 'cmpl-uqkvlQyYK7bGYrRHQ0eXlWi8', 'object': 'text_completion', 'created': 1589478378, 'model': 'text-davinci-003', 'choices': [{'text': f'{self.userMessages[turn_number]}', 'index': 0, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 5, 'completion_tokens': 7, 'total_tokens': 12}}
        else:
            messages.extend([turn.to_openai_chat_format() for turn in conversation_history[-max_history:]])
            prompt_role = self.role.value
            response_data = {'id': 'cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7', 'object': 'text_completion', 'created': 1589478378, 'model': 'text-davinci-003', 'choices': [{'text': 'This is indeed a test response', 'index': 0, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 5, 'completion_tokens': 7, 'total_tokens': 12}}
        parsed_response = self._parse_response(response_data)
        request = {'messages': messages}
        return (parsed_response, request, 0, response_data)

    def _parse_response(self, response_data: dict) -> dict:
        if False:
            return 10
        samples = []
        finish_reason = []
        for choice in response_data['choices']:
            if 'text' in choice:
                samples.append(choice['text'])
            if 'finish_reason' in choice:
                finish_reason.append(choice['finish_reason'])
        return {'samples': samples, 'finish_reason': finish_reason, 'id': response_data['id']}

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'Bot(name={self.name}, role={self.role.name}, model=dummy)'