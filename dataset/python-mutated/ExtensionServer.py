import logging
import os
import os.path
from functools import lru_cache
from gi.repository import Gio, GObject
from ulauncher.api.shared.socket_path import get_socket_path
from ulauncher.modes.extensions.ExtensionController import ExtensionController
from ulauncher.utils.framer import JSONFramer
logger = logging.getLogger()

class ExtensionServer:

    @classmethod
    @lru_cache(maxsize=None)
    def get_instance(cls):
        if False:
            i = 10
            return i + 15
        return cls()

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.service = None
        self.socket_path = get_socket_path()
        self.controllers = {}
        self.pending = {}

    def start(self):
        if False:
            print('Hello World!')
        '\n        Starts extension server\n        '
        if self.is_running():
            raise ServerIsRunningError
        self.service = Gio.SocketService.new()
        self.service.connect('incoming', self.handle_incoming)
        if os.path.exists(self.socket_path):
            logger.debug('Removing existing socket path %s', self.socket_path)
            os.unlink(self.socket_path)
        self.service.add_address(Gio.UnixSocketAddress.new(self.socket_path), Gio.SocketType.STREAM, Gio.SocketProtocol.DEFAULT, None)
        self.pending = {}
        self.controllers = {}

    def handle_incoming(self, _service, conn, _source):
        if False:
            print('Hello World!')
        framer = JSONFramer()
        msg_handler_id = framer.connect('message_parsed', self.handle_registration)
        closed_handler_id = framer.connect('closed', self.handle_pending_close)
        self.pending[id(framer)] = (framer, msg_handler_id, closed_handler_id)
        framer.set_connection(conn)

    def handle_pending_close(self, framer):
        if False:
            print('Hello World!')
        self.pending.pop(id(framer))

    def handle_registration(self, framer, event):
        if False:
            return 10
        if isinstance(event, dict) and event.get('type') == 'extension:socket_connected':
            pended = self.pending.pop(id(framer))
            if pended:
                for msg_id in pended[1:]:
                    GObject.signal_handler_disconnect(framer, msg_id)
            ExtensionController(self.controllers, framer, event.get('ext_id'))
        else:
            logger.debug('Unhandled message received: %s', event)

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stops extension server\n        '
        if not self.is_running():
            raise ServerIsNotRunningError
        self.service.stop()
        self.service.close()
        self.service = None

    def is_running(self):
        if False:
            print('Hello World!')
        '\n        :rtype: bool\n        '
        return bool(self.service)

    def get_controllers(self):
        if False:
            print('Hello World!')
        '\n        :rtype: list of  :class:`~ulauncher.modes.extensions.ExtensionController.ExtensionController`\n        '
        return self.controllers.values()

    def get_controller_by_id(self, extension_id):
        if False:
            return 10
        '\n        :param str extension_id:\n        :rtype: ~ulauncher.modes.extensions.ExtensionController.ExtensionController\n        '
        return self.controllers.get(extension_id)

    def get_controller_by_keyword(self, keyword):
        if False:
            while True:
                i = 10
        '\n        :param str keyword:\n        :rtype: ~ulauncher.modes.extensions.ExtensionController.ExtensionController\n        '
        for controller in self.controllers.values():
            for trigger in controller.manifest.triggers.values():
                if keyword and keyword == trigger.user_keyword:
                    return controller
        return None

class ServerIsRunningError(RuntimeError):
    pass

class ServerIsNotRunningError(RuntimeError):
    pass
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    server = ExtensionServer.get_instance()
    server.start()