import os
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.test.util import dirs
from buildbot.util import maildir

class TestMaildirService(dirs.DirsMixin, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.maildir = os.path.abspath('maildir')
        self.newdir = os.path.join(self.maildir, 'new')
        self.curdir = os.path.join(self.maildir, 'cur')
        self.tmpdir = os.path.join(self.maildir, 'tmp')
        self.setUpDirs(self.maildir, self.newdir, self.curdir, self.tmpdir)
        self.svc = None

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.svc and self.svc.running:
            self.svc.stopService()
        self.tearDownDirs()

    @defer.inlineCallbacks
    def test_start_stop_repeatedly(self):
        if False:
            for i in range(10):
                print('nop')
        self.svc = maildir.MaildirService(self.maildir)
        self.svc.startService()
        yield self.svc.stopService()
        self.svc.startService()
        yield self.svc.stopService()
        self.assertEqual(len(list(self.svc)), 0)

    @defer.inlineCallbacks
    def test_messageReceived(self):
        if False:
            return 10
        self.svc = maildir.MaildirService(self.maildir)
        messagesReceived = []

        def messageReceived(filename):
            if False:
                print('Hello World!')
            messagesReceived.append(filename)
            return defer.succeed(None)
        self.svc.messageReceived = messageReceived
        yield self.svc.startService()
        self.assertEqual(messagesReceived, [])
        tmpfile = os.path.join(self.tmpdir, 'newmsg')
        newfile = os.path.join(self.newdir, 'newmsg')
        with open(tmpfile, 'w', encoding='utf-8'):
            pass
        os.rename(tmpfile, newfile)
        yield self.svc.poll()
        self.assertEqual(messagesReceived, ['newmsg'])

    def test_moveToCurDir(self):
        if False:
            return 10
        self.svc = maildir.MaildirService(self.maildir)
        tmpfile = os.path.join(self.tmpdir, 'newmsg')
        newfile = os.path.join(self.newdir, 'newmsg')
        with open(tmpfile, 'w', encoding='utf-8'):
            pass
        os.rename(tmpfile, newfile)
        f = self.svc.moveToCurDir('newmsg')
        f.close()
        self.assertEqual([os.path.exists(os.path.join(d, 'newmsg')) for d in (self.newdir, self.curdir, self.tmpdir)], [False, True, False])