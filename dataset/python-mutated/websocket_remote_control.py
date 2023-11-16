import threading
import logging
from socketIO_client import BaseNamespace
from socketIO_client import SocketIO
from pokemongo_bot import inventory

class WebsocketRemoteControl(object):

    def __init__(self, bot):
        if False:
            while True:
                i = 10
        self.bot = bot
        (self.host, port_str) = self.bot.config.websocket_server_url.split(':')
        self.port = int(port_str)
        self.sio = SocketIO(self.host, self.port)
        self.sio.on('bot:process_request:{}'.format(self.bot.config.username), self.on_remote_command)
        self.thread = threading.Thread(target=self.process_messages)
        self.logger = logging.getLogger(type(self).__name__)

    def start(self):
        if False:
            print('Hello World!')
        self.thread.start()
        return self

    def process_messages(self):
        if False:
            while True:
                i = 10
        self.sio.wait()

    def on_remote_command(self, command):
        if False:
            for i in range(10):
                print('nop')
        name = command['name']
        command_handler = getattr(self, name, None)
        if not command_handler or not callable(command_handler):
            self.sio.emit('bot:send_reply', {'response': '', 'command': 'command_not_found', 'account': self.bot.config.username})
            return
        if 'args' in command:
            command_handler(*args)
            return
        command_handler()

    def get_player_info(self):
        if False:
            print('Hello World!')
        try:
            self.sio.emit('bot:send_reply', {'result': {'inventory': inventory.jsonify_inventory(), 'player': self.bot._player}, 'command': 'get_player_info', 'account': self.bot.config.username})
        except Exception as e:
            self.logger.error(e)