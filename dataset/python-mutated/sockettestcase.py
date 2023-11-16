__author__ = 'Steven Hiscocks'
__copyright__ = 'Copyright (c) 2013 Steven Hiscocks'
__license__ = 'GPL'
import os
import sys
import tempfile
import threading
import time
import unittest
from .utils import LogCaptureTestCase
from .. import protocol
from ..server.asyncserver import asyncore, RequestHandler, loop, AsyncServer, AsyncServerException
from ..server.utils import Utils
from ..client.csocket import CSocket
from .utils import LogCaptureTestCase

def TestMsgError(*args):
    if False:
        return 10
    raise Exception('test unpickle error')

class TestMsg(object):

    def __init__(self, unpickle=(TestMsgError, ())):
        if False:
            return 10
        self.unpickle = unpickle

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return self.unpickle

class Socket(LogCaptureTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        'Call before every test case.'
        LogCaptureTestCase.setUp(self)
        super(Socket, self).setUp()
        self.server = AsyncServer(self)
        (sock_fd, sock_name) = tempfile.mkstemp('fail2ban.sock', 'f2b-socket')
        os.close(sock_fd)
        os.remove(sock_name)
        self.sock_name = sock_name
        self.serverThread = None

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Call after every test case.'
        if self.serverThread:
            self.server.stop()
            self._stopServerThread()
        LogCaptureTestCase.tearDown(self)

    @staticmethod
    def proceed(message):
        if False:
            for i in range(10):
                print('nop')
        'Test transmitter proceed method which just returns first arg'
        return message

    def _createServerThread(self, force=False):
        if False:
            i = 10
            return i + 15
        self.serverThread = serverThread = threading.Thread(target=self.server.start, args=(self.sock_name, force))
        serverThread.daemon = True
        serverThread.start()
        self.assertTrue(Utils.wait_for(self.server.isActive, unittest.F2B.maxWaitTime(10)))
        return serverThread

    def _stopServerThread(self):
        if False:
            print('Hello World!')
        serverThread = self.serverThread
        Utils.wait_for(lambda : not serverThread.is_alive() or serverThread.join(Utils.DEFAULT_SLEEP_TIME), unittest.F2B.maxWaitTime(10))
        self.serverThread = None

    def testStopPerCloseUnexpected(self):
        if False:
            for i in range(10):
                print('nop')
        serverThread = self._createServerThread()
        self.server.close()
        self._stopServerThread()
        self.assertFalse(serverThread.is_alive())
        self.server.stop()
        self.assertFalse(self.server.isActive())
        self.assertFalse(os.path.exists(self.sock_name))

    def _serverSocket(self):
        if False:
            i = 10
            return i + 15
        try:
            return CSocket(self.sock_name)
        except Exception as e:
            return None

    def testSocket(self):
        if False:
            i = 10
            return i + 15
        serverThread = self._createServerThread()
        client = Utils.wait_for(self._serverSocket, 2)
        testMessage = ['A', 'test', 'message']
        self.assertEqual(client.send(testMessage), testMessage)
        self.assertEqual(client.send([[TestMsg()]]), 'ERROR: test unpickle error')
        self.assertLogged('PROTO-error: load message failed:', 'test unpickle error', all=True)
        self.assertEqual(client.send(testMessage), testMessage)
        client.close()
        client.close()
        self.server.stop_communication()
        client = Utils.wait_for(self._serverSocket, 2)
        self.assertEqual(client.send(testMessage), ['SHUTDOWN'])
        self.server.stop()
        self._stopServerThread()
        self.assertFalse(serverThread.is_alive())
        self.assertFalse(self.server.isActive())
        self.assertFalse(os.path.exists(self.sock_name))

    def testSocketConnectBroken(self):
        if False:
            return 10
        serverThread = self._createServerThread()
        client = Utils.wait_for(self._serverSocket, 2)
        testMessage = ['A', 'test', 'message', [protocol.CSPROTO.END]]
        org_handler = RequestHandler.found_terminator
        try:
            RequestHandler.found_terminator = lambda self: self.close()
            self.assertRaisesRegex(Exception, 'reset by peer|Broken pipe', lambda : client.send(testMessage, timeout=unittest.F2B.maxWaitTime(10)))
        finally:
            RequestHandler.found_terminator = org_handler

    def testStopByCommunicate(self):
        if False:
            i = 10
            return i + 15
        serverThread = self._createServerThread()
        client = Utils.wait_for(self._serverSocket, 2)
        testMessage = ['A', 'test', 'message']
        self.assertEqual(client.send(testMessage), testMessage)
        org_handler = RequestHandler.found_terminator
        try:
            RequestHandler.found_terminator = lambda self: TestMsgError()
            self.assertEqual(client.send(testMessage), 'ERROR: test unpickle error')
        finally:
            RequestHandler.found_terminator = org_handler
        self.assertLogged('Unexpected communication error', 'test unpickle error', all=True)
        self.server.stop()
        self._stopServerThread()
        self.assertFalse(serverThread.is_alive())

    def testLoopErrors(self):
        if False:
            return 10
        org_poll = asyncore.poll
        err = {'cntr': 0}

        def _produce_error(*args):
            if False:
                print('Hello World!')
            err['cntr'] += 1
            if err['cntr'] < 50:
                raise RuntimeError('test errors in poll')
            return org_poll(*args)
        try:
            asyncore.poll = _produce_error
            serverThread = self._createServerThread()
            self.assertTrue(Utils.wait_for(lambda : err['cntr'] > 50, unittest.F2B.maxWaitTime(10)))
        finally:
            asyncore.poll = org_poll
        self.assertLogged('Server connection was closed: test errors in poll', 'Too many errors - stop logging connection errors', all=True)

    def testSocketForce(self):
        if False:
            i = 10
            return i + 15
        open(self.sock_name, 'w').close()
        self.assertRaises(AsyncServerException, self.server.start, self.sock_name, False)
        serverThread = self._createServerThread(True)
        self.server.stop()
        self._stopServerThread()
        self.assertFalse(serverThread.is_alive())
        self.assertFalse(self.server.isActive())
        self.assertFalse(os.path.exists(self.sock_name))

class ClientMisc(LogCaptureTestCase):

    def testErrorsInLoop(self):
        if False:
            for i in range(10):
                print('nop')
        phase = {'cntr': 0}

        def _active():
            if False:
                for i in range(10):
                    print('nop')
            return phase['cntr'] < 40

        def _poll(*args):
            if False:
                return 10
            phase['cntr'] += 1
            raise Exception('test *%d*' % phase['cntr'])
        loop(_active, use_poll=_poll)
        self.assertLogged('test *1*', 'test *10*', 'test *20*', all=True)
        self.assertLogged('Too many errors - stop logging connection errors')
        self.assertNotLogged('test *21*', 'test *22*', 'test *23*', all=True)

    def testPrintFormattedAndWiki(self):
        if False:
            while True:
                i = 10
        saved_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            protocol.printFormatted()
            protocol.printWiki()
        finally:
            sys.stdout = saved_stdout