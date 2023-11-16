import argparse
import pytest
from fastapi_poe.types import ProtocolMessage, QueryRequest
from embedchain.bots.poe import PoeBot, start_command

@pytest.fixture
def poe_bot(mocker):
    if False:
        return 10
    bot = PoeBot()
    mocker.patch('fastapi_poe.run')
    return bot

@pytest.mark.asyncio
async def test_poe_bot_get_response(poe_bot, mocker):
    query = QueryRequest(version='test', type='query', query=[ProtocolMessage(role='system', content='Test content')], user_id='test_user_id', conversation_id='test_conversation_id', message_id='test_message_id')
    mocker.patch.object(poe_bot.app.llm, 'set_history')
    response_generator = poe_bot.get_response(query)
    await response_generator.__anext__()
    poe_bot.app.llm.set_history.assert_called_once()

def test_poe_bot_handle_message(poe_bot, mocker):
    if False:
        print('Hello World!')
    mocker.patch.object(poe_bot, 'ask_bot', return_value='Answer from the bot')
    response_ask = poe_bot.handle_message('What is the answer?')
    assert response_ask == 'Answer from the bot'

def test_start_command(mocker):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(api_key='test_api_key'))
    mocker.patch('embedchain.bots.poe.run')
    start_command()