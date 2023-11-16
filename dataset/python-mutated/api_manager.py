from __future__ import annotations
import logging
from typing import List, Optional
import openai
from openai import Model
from autogpt.core.resource.model_providers.openai import OPEN_AI_MODELS
from autogpt.core.resource.model_providers.schema import ChatModelInfo
from autogpt.singleton import Singleton
logger = logging.getLogger(__name__)

class ApiManager(metaclass=Singleton):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.models: Optional[list[Model]] = None

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0
        self.models = None

    def update_cost(self, prompt_tokens, completion_tokens, model):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the total cost, prompt tokens, and completion tokens.\n\n        Args:\n        prompt_tokens (int): The number of tokens used in the prompt.\n        completion_tokens (int): The number of tokens used in the completion.\n        model (str): The model used for the API call.\n        '
        model = model[:-3] if model.endswith('-v2') else model
        model_info = OPEN_AI_MODELS[model]
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += prompt_tokens * model_info.prompt_token_cost / 1000
        if isinstance(model_info, ChatModelInfo):
            self.total_cost += completion_tokens * model_info.completion_token_cost / 1000
        logger.debug(f'Total running cost: ${self.total_cost:.3f}')

    def set_total_budget(self, total_budget):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the total user-defined budget for API calls.\n\n        Args:\n        total_budget (float): The total budget for API calls.\n        '
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the total number of prompt tokens.\n\n        Returns:\n        int: The total number of prompt tokens.\n        '
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the total number of completion tokens.\n\n        Returns:\n        int: The total number of completion tokens.\n        '
        return self.total_completion_tokens

    def get_total_cost(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the total cost of API calls.\n\n        Returns:\n        float: The total cost of API calls.\n        '
        return self.total_cost

    def get_total_budget(self):
        if False:
            while True:
                i = 10
        '\n        Get the total user-defined budget for API calls.\n\n        Returns:\n        float: The total budget for API calls.\n        '
        return self.total_budget

    def get_models(self, **openai_credentials) -> List[Model]:
        if False:
            return 10
        '\n        Get list of available GPT models.\n\n        Returns:\n        list: List of available GPT models.\n\n        '
        if self.models is None:
            all_models = openai.Model.list(**openai_credentials)['data']
            self.models = [model for model in all_models if 'gpt' in model['id']]
        return self.models