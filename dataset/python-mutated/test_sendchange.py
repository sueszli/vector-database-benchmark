from twisted.internet import defer
from twisted.internet import reactor
from twisted.trial import unittest
from buildbot.clients import sendchange as sendchange_client
from buildbot.scripts import sendchange
from buildbot.test.util import misc

class TestSendChange(misc.StdoutAssertionsMixin, unittest.TestCase):

    class FakeSender:

        def __init__(self, testcase, master, auth, encoding=None):
            if False:
                return 10
            self.master = master
            self.auth = auth
            self.encoding = encoding
            self.testcase = testcase

        def send(self, branch, revision, comments, files, **kwargs):
            if False:
                print('Hello World!')
            kwargs['branch'] = branch
            kwargs['revision'] = revision
            kwargs['comments'] = comments
            kwargs['files'] = files
            self.send_kwargs = kwargs
            d = defer.Deferred()
            if self.testcase.fail:
                reactor.callLater(0, d.errback, RuntimeError('oh noes'))
            else:
                reactor.callLater(0, d.callback, None)
            return d

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fail = False

        def Sender_constr(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self.sender = self.FakeSender(self, *args, **kwargs)
            return self.sender
        self.patch(sendchange_client, 'Sender', Sender_constr)
        self.patch(sendchange, 'sendchange', sendchange.sendchange._orig)
        self.setUpStdoutAssertions()

    @defer.inlineCallbacks
    def test_sendchange_config(self):
        if False:
            print('Hello World!')
        rc = (yield sendchange.sendchange({'encoding': 'utf16', 'who': 'me', 'auth': ['a', 'b'], 'master': 'm', 'branch': 'br', 'category': 'cat', 'revision': 'rr', 'properties': {'a': 'b'}, 'repository': 'rep', 'project': 'prj', 'vc': 'git', 'revlink': 'rl', 'when': 1234.0, 'comments': 'comm', 'files': ('a', 'b'), 'codebase': 'cb'}))
        self.assertEqual((self.sender.master, self.sender.auth, self.sender.encoding, self.sender.send_kwargs, self.getStdout(), rc), ('m', ['a', 'b'], 'utf16', {'branch': 'br', 'category': 'cat', 'codebase': 'cb', 'comments': 'comm', 'files': ('a', 'b'), 'project': 'prj', 'properties': {'a': 'b'}, 'repository': 'rep', 'revision': 'rr', 'revlink': 'rl', 'when': 1234.0, 'who': 'me', 'vc': 'git'}, 'change sent successfully', 0))

    @defer.inlineCallbacks
    def test_sendchange_config_no_codebase(self):
        if False:
            i = 10
            return i + 15
        rc = (yield sendchange.sendchange({'encoding': 'utf16', 'who': 'me', 'auth': ['a', 'b'], 'master': 'm', 'branch': 'br', 'category': 'cat', 'revision': 'rr', 'properties': {'a': 'b'}, 'repository': 'rep', 'project': 'prj', 'vc': 'git', 'revlink': 'rl', 'when': 1234.0, 'comments': 'comm', 'files': ('a', 'b')}))
        self.assertEqual((self.sender.master, self.sender.auth, self.sender.encoding, self.sender.send_kwargs, self.getStdout(), rc), ('m', ['a', 'b'], 'utf16', {'branch': 'br', 'category': 'cat', 'codebase': None, 'comments': 'comm', 'files': ('a', 'b'), 'project': 'prj', 'properties': {'a': 'b'}, 'repository': 'rep', 'revision': 'rr', 'revlink': 'rl', 'when': 1234.0, 'who': 'me', 'vc': 'git'}, 'change sent successfully', 0))

    @defer.inlineCallbacks
    def test_sendchange_fail(self):
        if False:
            print('Hello World!')
        self.fail = True
        rc = (yield sendchange.sendchange({}))
        self.assertEqual((self.getStdout().split('\n')[0], rc), ('change not sent:', 1))