from websocket import create_connection
from mycroft.configuration import Configuration
from mycroft.messagebus.client import MessageBusClient
from mycroft.messagebus.message import Message

def send(message_to_send, data_to_send=None):
    if False:
        for i in range(10):
            print('nop')
    'Send a single message over the websocket.\n\n    Args:\n        message_to_send (str): Message to send\n        data_to_send (dict): data structure to go along with the\n            message, defaults to empty dict.\n    '
    data_to_send = data_to_send or {}
    config = Configuration.get(cache=False, remote=False)
    config = config.get('websocket')
    url = MessageBusClient.build_url(config.get('host'), config.get('port'), config.get('route'), config.get('ssl'))
    ws = create_connection(url)
    packet = Message(message_to_send, data_to_send).serialize()
    ws.send(packet)
    ws.close()