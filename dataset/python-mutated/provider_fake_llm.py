from typing import Any, Iterator, List, Optional, Union
from langchain.schema.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.output import ChatGenerationChunk
from lwe.core.provider import Provider, PresetValue
from lwe.core import constants
import asyncio
import time
from typing import AsyncIterator, Dict
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult
from langchain.schema.output import ChatGeneration
DEFAULT_RESPONSE_MESSAGE = 'test response'

class FakeMessagesListChatModel(BaseChatModel):
    responses: Union[List[BaseMessage], List[List[BaseMessage]]]
    sleep: Optional[float] = None
    i: int = 0

    @property
    def _llm_type(self) -> str:
        if False:
            return 10
        return 'fake-messages-list-chat-model'

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'responses': self.responses}

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        if False:
            print('Hello World!')
        response = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Union[BaseMessage, List[BaseMessage]]:
        if False:
            for i in range(10):
                print('nop')
        "First try to lookup in queries, else return 'foo' or 'bar'."
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    def _stream(self, messages: List[BaseMessage], stop: Union[List[str], None]=None, run_manager: Union[CallbackManagerForLLMRun, None]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        if False:
            return 10
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for c in response:
            if self.sleep is not None:
                time.sleep(self.sleep)
            yield ChatGenerationChunk(message=c)

    async def _astream(self, messages: List[BaseMessage], stop: Union[List[str], None]=None, run_manager: Union[AsyncCallbackManagerForLLMRun, None]=None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for c in response:
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)
            yield ChatGenerationChunk(message=c)

class CustomFakeMessagesListChatModel(FakeMessagesListChatModel):
    model_name: str

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        if not kwargs.get('model_name'):
            kwargs['model_name'] = constants.API_BACKEND_DEFAULT_MODEL
        if not kwargs.get('responses'):
            kwargs['responses'] = []
        super().__init__(**kwargs)

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> BaseMessage:
        if False:
            i = 10
            return i + 15
        if not self.responses:
            self.responses = [AIMessage(content=DEFAULT_RESPONSE_MESSAGE)]
        return super()._call(messages, stop, run_manager, **kwargs)

    def _stream(self, messages: List[BaseMessage], stop: Union[List[str], None]=None, run_manager: Union[CallbackManagerForLLMRun, None]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        if False:
            print('Hello World!')
        if not self.responses:
            self.responses = [[AIMessageChunk(content=DEFAULT_RESPONSE_MESSAGE)]]
        return super()._stream(messages, stop, run_manager, **kwargs)

class ProviderFakeLlm(Provider):
    """
    Fake LLM provider.
    """

    @property
    def capabilities(self):
        if False:
            for i in range(10):
                print('nop')
        return {'chat': True, 'validate_models': False, 'models': {'gpt-3.5-turbo': {'max_tokens': 4096}, 'gpt-3.5-turbo-1106': {'max_tokens': 16384}, 'gpt-4': {'max_tokens': 8192}}}

    @property
    def default_model(self):
        if False:
            i = 10
            return i + 15
        return constants.API_BACKEND_DEFAULT_MODEL

    def prepare_messages_method(self):
        if False:
            return 10
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        if False:
            while True:
                i = 10
        return CustomFakeMessagesListChatModel

    def customization_config(self):
        if False:
            return 10
        return {'responses': None, 'model_name': PresetValue(str, options=self.available_models)}