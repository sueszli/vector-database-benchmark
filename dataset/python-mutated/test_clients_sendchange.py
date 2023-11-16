from unittest import mock
from twisted.internet import defer
from twisted.internet import reactor
from twisted.spread import pb
from twisted.trial import unittest
from buildbot.clients import sendchange

class Sender(unittest.TestCase):

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
        self.creds = None
        self.conn_host = self.conn_port = None
        self.lostConnection = False
        self.added_changes = []
        self.vc_used = None

    def _fake_PBClientFactory(self):
        if False:
            for i in range(10):
                print('nop')
        return self.factory

    def _fake_login(self, creds):
        if False:
            for i in range(10):
                print('nop')
        self.creds = creds
        return self.factory.login_d

    def _fake_connectTCP(self, host, port, factory):
        if False:
            i = 10
            return i + 15
        self.conn_host = host
        self.conn_port = port
        self.assertIdentical(factory, self.factory)
        self.factory.login_d.callback(self.remote)

    def _fake_callRemote(self, method, change):
        if False:
            return 10
        self.assertEqual(method, 'addChange')
        self.added_changes.append(change)
        return defer.succeed(None)

    def _fake_loseConnection(self):
        if False:
            while True:
                i = 10
        self.lostConnection = True

    def assertProcess(self, host, port, username, password, changes):
        if False:
            while True:
                i = 10
        self.assertEqual([host, port, username, password, changes], [self.conn_host, self.conn_port, self.creds.username, self.creds.password, self.added_changes])

    @defer.inlineCallbacks
    def test_send_minimal(self):
        if False:
            print('Hello World!')
        s = sendchange.Sender('localhost:1234')
        yield s.send('branch', 'rev', 'comm', ['a'])
        self.assertProcess('localhost', 1234, b'change', b'changepw', [{'project': '', 'repository': '', 'who': None, 'files': ['a'], 'comments': 'comm', 'branch': 'branch', 'revision': 'rev', 'category': None, 'when': None, 'properties': {}, 'revlink': '', 'src': None}])

    @defer.inlineCallbacks
    def test_send_auth(self):
        if False:
            print('Hello World!')
        s = sendchange.Sender('localhost:1234', auth=('me', 'sekrit'))
        yield s.send('branch', 'rev', 'comm', ['a'])
        self.assertProcess('localhost', 1234, b'me', b'sekrit', [{'project': '', 'repository': '', 'who': None, 'files': ['a'], 'comments': 'comm', 'branch': 'branch', 'revision': 'rev', 'category': None, 'when': None, 'properties': {}, 'revlink': '', 'src': None}])

    @defer.inlineCallbacks
    def test_send_full(self):
        if False:
            for i in range(10):
                print('nop')
        s = sendchange.Sender('localhost:1234')
        yield s.send('branch', 'rev', 'comm', ['a'], who='me', category='cats', when=1234, properties={'a': 'b'}, repository='r', vc='git', project='p', revlink='rl')
        self.assertProcess('localhost', 1234, b'change', b'changepw', [{'project': 'p', 'repository': 'r', 'who': 'me', 'files': ['a'], 'comments': 'comm', 'branch': 'branch', 'revision': 'rev', 'category': 'cats', 'when': 1234, 'properties': {'a': 'b'}, 'revlink': 'rl', 'src': 'git'}])

    @defer.inlineCallbacks
    def test_send_files_tuple(self):
        if False:
            i = 10
            return i + 15
        s = sendchange.Sender('localhost:1234')
        yield s.send('branch', 'rev', 'comm', ('a', 'b'))
        self.assertProcess('localhost', 1234, b'change', b'changepw', [{'project': '', 'repository': '', 'who': None, 'files': ['a', 'b'], 'comments': 'comm', 'branch': 'branch', 'revision': 'rev', 'category': None, 'when': None, 'properties': {}, 'revlink': '', 'src': None}])

    @defer.inlineCallbacks
    def test_send_codebase(self):
        if False:
            print('Hello World!')
        s = sendchange.Sender('localhost:1234')
        yield s.send('branch', 'rev', 'comm', ['a'], codebase='mycb')
        self.assertProcess('localhost', 1234, b'change', b'changepw', [{'project': '', 'repository': '', 'who': None, 'files': ['a'], 'comments': 'comm', 'branch': 'branch', 'revision': 'rev', 'category': None, 'when': None, 'properties': {}, 'revlink': '', 'src': None, 'codebase': 'mycb'}])

    @defer.inlineCallbacks
    def test_send_unicode(self):
        if False:
            return 10
        s = sendchange.Sender('localhost:1234')
        yield s.send('°', '💞', '〠', ['📁'], project='☠', repository='☃', who='๛', category='🙀', when=1234, properties={'ā': 'b'}, revlink='🔗')
        self.assertProcess('localhost', 1234, b'change', b'changepw', [{'project': '☠', 'repository': '☃', 'who': '๛', 'files': ['📁'], 'comments': '〠', 'branch': '°', 'revision': '💞', 'category': '🙀', 'when': 1234, 'properties': {'ā': 'b'}, 'revlink': '🔗', 'src': None}])

    @defer.inlineCallbacks
    def test_send_unicode_utf8(self):
        if False:
            while True:
                i = 10
        s = sendchange.Sender('localhost:1234')
        yield s.send('°'.encode('utf8'), '💞'.encode('utf8'), '〠'.encode('utf8'), ['📁'.encode('utf8')], project='☠'.encode('utf8'), repository='☃'.encode('utf8'), who='๛'.encode('utf8'), category='🙀'.encode('utf8'), when=1234, properties={'ā'.encode('utf8'): 'b'}, revlink='🔗'.encode('utf8'))
        self.assertProcess('localhost', 1234, b'change', b'changepw', [{'project': '☠', 'repository': '☃', 'who': '๛', 'files': ['📁'], 'comments': '〠', 'branch': '°', 'revision': '💞', 'category': '🙀', 'when': 1234, 'properties': {b'\xc4\x81': 'b'}, 'revlink': '🔗', 'src': None}])

    @defer.inlineCallbacks
    def test_send_unicode_latin1(self):
        if False:
            print('Hello World!')
        s = sendchange.Sender('localhost:1234', encoding='latin1')
        yield s.send('¥'.encode('latin1'), '£'.encode('latin1'), '¦'.encode('latin1'), ['¬'.encode('latin1')], project='°'.encode('latin1'), repository='§'.encode('latin1'), who='¯'.encode('latin1'), category='¶'.encode('latin1'), when=1234, properties={'¹'.encode('latin1'): 'b'}, revlink='¿'.encode('latin1'))
        self.assertProcess('localhost', 1234, b'change', b'changepw', [{'project': '°', 'repository': '§', 'who': '¯', 'files': ['¬'], 'comments': '¦', 'branch': '¥', 'revision': '£', 'category': '¶', 'when': 1234, 'properties': {b'\xb9': 'b'}, 'revlink': '¿', 'src': None}])