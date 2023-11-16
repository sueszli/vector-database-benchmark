import importlib
import os
from typing import Optional
from langchain.llms import Replicate
from embedchain.config import BaseLlmConfig
from embedchain.helper.json_serializable import register_deserializable
from embedchain.llm.base import BaseLlm

@register_deserializable
class Llama2Llm(BaseLlm):

    def __init__(self, config: Optional[BaseLlmConfig]=None):
        if False:
            while True:
                i = 10
        try:
            importlib.import_module('replicate')
        except ModuleNotFoundError:
            raise ModuleNotFoundError('The required dependencies for Llama2 are not installed.Please install with `pip install --upgrade "embedchain[llama2]"`') from None
        if 'REPLICATE_API_TOKEN' not in os.environ:
            raise ValueError('Please set the REPLICATE_API_TOKEN environment variable.')
        if not config:
            config = BaseLlmConfig()
            config.max_tokens = 500
            config.temperature = 0.75
        if not config.model:
            config.model = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
        super().__init__(config=config)

    def get_llm_model_answer(self, prompt):
        if False:
            print('Hello World!')
        if self.config.system_prompt:
            raise ValueError('Llama2 does not support `system_prompt`')
        llm = Replicate(model=self.config.model, input={'temperature': self.config.temperature, 'max_length': self.config.max_tokens, 'top_p': self.config.top_p})
        return llm(prompt)