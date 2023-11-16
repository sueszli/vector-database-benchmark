import argparse
import logging
import os
from typing import List, Optional
from embedchain.helper.json_serializable import register_deserializable
from .base import BaseBot
try:
    from fastapi_poe import PoeBot, run
except ModuleNotFoundError:
    raise ModuleNotFoundError('The required dependencies for Poe are not installed.Please install with `pip install "embedchain[poe]"`') from None

def start_command():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='EmbedChain PoeBot command line interface')
    parser.add_argument('--port', default=8080, type=int, help='Port to bind')
    parser.add_argument('--api-key', type=str, help='Poe API key')
    args = parser.parse_args()
    run(PoeBot(), api_key=args.api_key or os.environ.get('POE_API_KEY'))

@register_deserializable
class PoeBot(BaseBot, PoeBot):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.history_length = 5
        super().__init__()

    async def get_response(self, query):
        last_message = query.query[-1].content
        try:
            history = [f'{m.role}: {m.content}' for m in query.query[-(self.history_length + 1):-1]] if len(query.query) > 0 else None
        except Exception as e:
            logging.error(f'Error when processing the chat history. Message is being sent without history. Error: {e}')
        answer = self.handle_message(last_message, history)
        yield self.text_event(answer)

    def handle_message(self, message, history: Optional[List[str]]=None):
        if False:
            return 10
        if message.startswith('/add '):
            response = self.add_data(message)
        else:
            response = self.ask_bot(message, history)
        return response

    def ask_bot(self, message, history: List[str]):
        if False:
            while True:
                i = 10
        try:
            self.app.llm.set_history(history=history)
            response = self.query(message)
        except Exception:
            logging.exception(f'Failed to query {message}.')
            response = 'An error occurred. Please try again!'
        return response

    def start(self):
        if False:
            i = 10
            return i + 15
        start_command()
if __name__ == '__main__':
    start_command()