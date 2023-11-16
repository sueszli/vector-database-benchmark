from __future__ import annotations
import os
import sys
sys.path[0:0] = ['']
from test import SkipTest, client_context, unittest

class TestClientContext(unittest.TestCase):

    def test_must_connect(self):
        if False:
            for i in range(10):
                print('nop')
        if 'PYMONGO_MUST_CONNECT' not in os.environ:
            raise SkipTest('PYMONGO_MUST_CONNECT is not set')
        self.assertTrue(client_context.connected, 'client context must be connected when PYMONGO_MUST_CONNECT is set. Failed attempts:\n{}'.format(client_context.connection_attempt_info()))

    def test_serverless(self):
        if False:
            for i in range(10):
                print('nop')
        if 'TEST_SERVERLESS' not in os.environ:
            raise SkipTest('TEST_SERVERLESS is not set')
        self.assertTrue(client_context.connected and client_context.serverless, f'client context must be connected to serverless when TEST_SERVERLESS is set. Failed attempts:\n{client_context.connection_attempt_info()}')

    def test_enableTestCommands_is_disabled(self):
        if False:
            return 10
        if 'PYMONGO_DISABLE_TEST_COMMANDS' not in os.environ:
            raise SkipTest('PYMONGO_DISABLE_TEST_COMMANDS is not set')
        self.assertFalse(client_context.test_commands_enabled, 'enableTestCommands must be disabled when PYMONGO_DISABLE_TEST_COMMANDS is set.')

    def test_setdefaultencoding_worked(self):
        if False:
            i = 10
            return i + 15
        if 'SETDEFAULTENCODING' not in os.environ:
            raise SkipTest('SETDEFAULTENCODING is not set')
        self.assertEqual(sys.getdefaultencoding(), os.environ['SETDEFAULTENCODING'])
if __name__ == '__main__':
    unittest.main()