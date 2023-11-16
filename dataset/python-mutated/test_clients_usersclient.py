from unittest import mock
from twisted.internet import defer
from twisted.internet import reactor
from twisted.spread import pb
from twisted.trial import unittest
from buildbot.clients import usersclient

class TestUsersClient(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch(pb, 'PBClientFactory', self._fake_PBClientFactory)
        self.patch(reactor, 'connectTCP', self._fake_connectTCP)
        self.factory = mock.Mock(name='PBClientFactory')
        self.factory.login = self._fake_login
        self.factory.login_d = defer.Deferred()
        self.remote = mock.Mock(name='PB Remote')
        self.remote.callRemote = self._fake_callRemote
        self.remote.broker.transport.loseConnection = self._fake_loseConnection
        self.conn_host = self.conn_port = None
        self.lostConnection = False

    def _fake_PBClientFactory(self):
        if False:
            for i in range(10):
                print('nop')
        return self.factory

    def _fake_login(self, creds):
        if False:
            return 10
        return self.factory.login_d

    def _fake_connectTCP(self, host, port, factory):
        if False:
            i = 10
            return i + 15
        self.conn_host = host
        self.conn_port = port
        self.assertIdentical(factory, self.factory)
        self.factory.login_d.callback(self.remote)

    def _fake_callRemote(self, method, op, bb_username, bb_password, ids, info):
        if False:
            while True:
                i = 10
        self.assertEqual(method, 'commandline')
        self.called_with = {'op': op, 'bb_username': bb_username, 'bb_password': bb_password, 'ids': ids, 'info': info}
        return defer.succeed(None)

    def _fake_loseConnection(self):
        if False:
            return 10
        self.lostConnection = True

    def assertProcess(self, host, port, called_with):
        if False:
            print('Hello World!')
        self.assertEqual([host, port, called_with], [self.conn_host, self.conn_port, self.called_with])

    @defer.inlineCallbacks
    def test_usersclient_info(self):
        if False:
            return 10
        uc = usersclient.UsersClient('localhost', 'user', 'userpw', 1234)
        yield uc.send('update', 'bb_user', 'hashed_bb_pass', None, [{'identifier': 'x', 'svn': 'x'}])
        self.assertProcess('localhost', 1234, {'op': 'update', 'bb_username': 'bb_user', 'bb_password': 'hashed_bb_pass', 'ids': None, 'info': [{'identifier': 'x', 'svn': 'x'}]})

    @defer.inlineCallbacks
    def test_usersclient_ids(self):
        if False:
            return 10
        uc = usersclient.UsersClient('localhost', 'user', 'userpw', 1234)
        yield uc.send('remove', None, None, ['x'], None)
        self.assertProcess('localhost', 1234, {'op': 'remove', 'bb_username': None, 'bb_password': None, 'ids': ['x'], 'info': None})