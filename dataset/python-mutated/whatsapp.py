import argparse
import importlib
import logging
import signal
import sys
from embedchain.helper.json_serializable import register_deserializable
from .base import BaseBot

@register_deserializable
class WhatsAppBot(BaseBot):

    def __init__(self):
        if False:
            while True:
                i = 10
        try:
            self.flask = importlib.import_module('flask')
            self.twilio = importlib.import_module('twilio')
        except ModuleNotFoundError:
            raise ModuleNotFoundError('The required dependencies for WhatsApp are not installed. Please install with `pip install --upgrade "embedchain[whatsapp]"`') from None
        super().__init__()

    def handle_message(self, message):
        if False:
            print('Hello World!')
        if message.startswith('add '):
            response = self.add_data(message)
        else:
            response = self.ask_bot(message)
        return response

    def add_data(self, message):
        if False:
            for i in range(10):
                print('nop')
        data = message.split(' ')[-1]
        try:
            self.add(data)
            response = f'Added data from: {data}'
        except Exception:
            logging.exception(f'Failed to add data {data}.')
            response = 'Some error occurred while adding data.'
        return response

    def ask_bot(self, message):
        if False:
            return 10
        try:
            response = self.query(message)
        except Exception:
            logging.exception(f'Failed to query {message}.')
            response = 'An error occurred. Please try again!'
        return response

    def start(self, host='0.0.0.0', port=5000, debug=True):
        if False:
            for i in range(10):
                print('nop')
        app = self.flask.Flask(__name__)

        def signal_handler(sig, frame):
            if False:
                print('Hello World!')
            logging.info('\nGracefully shutting down the WhatsAppBot...')
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        @app.route('/chat', methods=['POST'])
        def chat():
            if False:
                while True:
                    i = 10
            incoming_message = self.flask.request.values.get('Body', '').lower()
            response = self.handle_message(incoming_message)
            twilio_response = self.twilio.twiml.messaging_response.MessagingResponse()
            twilio_response.message(response)
            return str(twilio_response)
        app.run(host=host, port=port, debug=debug)

def start_command():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='EmbedChain WhatsAppBot command line interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host IP to bind')
    parser.add_argument('--port', default=5000, type=int, help='Port to bind')
    args = parser.parse_args()
    whatsapp_bot = WhatsAppBot()
    whatsapp_bot.start(host=args.host, port=args.port)
if __name__ == '__main__':
    start_command()