"""Define the web socket event handler for the message bus."""
import json
import sys
import traceback
from tornado.websocket import WebSocketHandler
from pyee import EventEmitter
from mycroft.messagebus.message import Message
from mycroft.util.log import LOG
client_connections = []

class MessageBusEventHandler(WebSocketHandler):

    def __init__(self, application, request, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(application, request, **kwargs)
        self.emitter = EventEmitter()

    def on(self, event_name, handler):
        if False:
            while True:
                i = 10
        self.emitter.on(event_name, handler)

    def on_message(self, message):
        if False:
            i = 10
            return i + 15
        LOG.debug(message)
        try:
            deserialized_message = Message.deserialize(message)
        except Exception:
            return
        try:
            self.emitter.emit(deserialized_message.msg_type, deserialized_message)
        except Exception as e:
            LOG.exception(e)
            traceback.print_exc(file=sys.stdout)
            pass
        for client in client_connections:
            client.write_message(message)

    def open(self):
        if False:
            for i in range(10):
                print('nop')
        self.write_message(Message('connected').serialize())
        client_connections.append(self)

    def on_close(self):
        if False:
            for i in range(10):
                print('nop')
        client_connections.remove(self)

    def emit(self, channel_message):
        if False:
            print('Hello World!')
        if hasattr(channel_message, 'serialize') and callable(getattr(channel_message, 'serialize')):
            self.write_message(channel_message.serialize())
        else:
            self.write_message(json.dumps(channel_message))

    def check_origin(self, origin):
        if False:
            for i in range(10):
                print('nop')
        return True