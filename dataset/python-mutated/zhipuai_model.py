from typing import List, Optional, Any
from langchain.callbacks.manager import Callbacks
from langchain.schema import LLMResult
from core.model_providers.error import LLMBadRequestError
from core.model_providers.models.llm.base import BaseLLM
from core.model_providers.models.entity.message import PromptMessage
from core.model_providers.models.entity.model_params import ModelMode, ModelKwargs
from core.third_party.langchain.llms.zhipuai_llm import ZhipuAIChatLLM

class ZhipuAIModel(BaseLLM):
    model_mode: ModelMode = ModelMode.CHAT

    def _init_client(self) -> Any:
        if False:
            return 10
        provider_model_kwargs = self._to_model_kwargs_input(self.model_rules, self.model_kwargs)
        return ZhipuAIChatLLM(model=self.name, streaming=self.streaming, callbacks=self.callbacks, **self.credentials, **provider_model_kwargs)

    def _run(self, messages: List[PromptMessage], stop: Optional[List[str]]=None, callbacks: Callbacks=None, **kwargs) -> LLMResult:
        if False:
            return 10
        '\n        run predict by prompt messages and stop words.\n\n        :param messages:\n        :param stop:\n        :param callbacks:\n        :return:\n        '
        prompts = self._get_prompt_from_messages(messages)
        return self._client.generate([prompts], stop, callbacks)

    def get_num_tokens(self, messages: List[PromptMessage]) -> int:
        if False:
            while True:
                i = 10
        '\n        get num tokens of prompt messages.\n\n        :param messages:\n        :return:\n        '
        prompts = self._get_prompt_from_messages(messages)
        return max(self._client.get_num_tokens_from_messages(prompts), 0)

    def _set_model_kwargs(self, model_kwargs: ModelKwargs):
        if False:
            print('Hello World!')
        provider_model_kwargs = self._to_model_kwargs_input(self.model_rules, model_kwargs)
        for (k, v) in provider_model_kwargs.items():
            if hasattr(self.client, k):
                setattr(self.client, k, v)

    def handle_exceptions(self, ex: Exception) -> Exception:
        if False:
            print('Hello World!')
        return LLMBadRequestError(f'ZhipuAI: {str(ex)}')

    @property
    def support_streaming(self):
        if False:
            i = 10
            return i + 15
        return True