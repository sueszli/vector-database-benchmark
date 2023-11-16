import unittest
import unittest.mock as mock
from golem_messages import message
from golem.network.concent import handlers_library

class HandlerLibraryTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.library = handlers_library.HandlersLibrary()

    def test_duplicated_handler(self):
        if False:
            return 10
        handler = mock.Mock()
        handler2 = mock.Mock()
        self.library.register_handler(message.p2p.Ping)(handler)
        with self.assertWarns(handlers_library.DuplicatedHandler):
            self.library.register_handler(message.p2p.Ping)(handler2)

    def test_unknown_message(self):
        if False:
            i = 10
            return i + 15
        self.library.interpret(message.p2p.Ping())

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        handler_ping = mock.Mock()
        handler_pong = mock.Mock()
        self.library.register_handler(message.p2p.Ping)(handler_ping)
        self.library.register_handler(message.p2p.Pong)(handler_pong)
        msg_ping = message.p2p.Ping()
        msg_pong = message.p2p.Pong()
        self.library.interpret(msg_ping)
        self.library.interpret(msg_pong)
        handler_ping.assert_called_once_with(msg_ping)
        handler_pong.assert_called_once_with(msg_pong)