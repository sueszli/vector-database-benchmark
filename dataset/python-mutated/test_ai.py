from gpt_engineer.core.ai import AI
from langchain.chat_models.fake import FakeListChatModel
from langchain.chat_models.base import BaseChatModel

def mock_create_chat_model(self) -> BaseChatModel:
    if False:
        for i in range(10):
            print('nop')
    return FakeListChatModel(responses=['response1', 'response2', 'response3'])

def mock_check_model_access_and_fallback(self, model_name):
    if False:
        for i in range(10):
            print('nop')
    return model_name

def test_start(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(AI, '_check_model_access_and_fallback', mock_check_model_access_and_fallback)
    monkeypatch.setattr(AI, '_create_chat_model', mock_create_chat_model)
    ai = AI('gpt-4')
    response_messages = ai.start('system prompt', 'user prompt', 'step name')
    assert response_messages[-1].content == 'response1'

def test_next(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr(AI, '_check_model_access_and_fallback', mock_check_model_access_and_fallback)
    monkeypatch.setattr(AI, '_create_chat_model', mock_create_chat_model)
    ai = AI('gpt-4')
    response_messages = ai.start('system prompt', 'user prompt', 'step name')
    response_messages = ai.next(response_messages, 'next user prompt', step_name='step name')
    assert response_messages[-1].content == 'response2'

def test_token_logging(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(AI, '_check_model_access_and_fallback', mock_check_model_access_and_fallback)
    monkeypatch.setattr(AI, '_create_chat_model', mock_create_chat_model)
    ai = AI('gpt-4')
    response_messages = ai.start('system prompt', 'user prompt', 'step name')
    usageCostAfterStart = ai.token_usage_log.usage_cost()
    ai.next(response_messages, 'next user prompt', step_name='step name')
    usageCostAfterNext = ai.token_usage_log.usage_cost()
    assert usageCostAfterStart > 0
    assert usageCostAfterNext > usageCostAfterStart