from twisted.internet import defer
from twisted.trial import unittest
from buildbot.process.users import users
from buildbot.test import fakedb
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin

class UsersTests(TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantDb=True)
        self.db = self.master.db
        self.test_sha = users.encrypt('cancer')

    @defer.inlineCallbacks
    def test_createUserObject_no_src(self):
        if False:
            i = 10
            return i + 15
        yield users.createUserObject(self.master, 'Tyler Durden', None)
        self.assertEqual(self.db.users.users, {})
        self.assertEqual(self.db.users.users_info, {})

    @defer.inlineCallbacks
    def test_createUserObject_unrecognized_src(self):
        if False:
            i = 10
            return i + 15
        yield users.createUserObject(self.master, 'Tyler Durden', 'blah')
        self.assertEqual(self.db.users.users, {})
        self.assertEqual(self.db.users.users_info, {})

    @defer.inlineCallbacks
    def test_createUserObject_git(self):
        if False:
            for i in range(10):
                print('nop')
        yield users.createUserObject(self.master, 'Tyler Durden <tyler@mayhem.net>', 'git')
        self.assertEqual(self.db.users.users, {1: {'identifier': 'Tyler Durden <tyler@mayhem.net>', 'bb_username': None, 'bb_password': None}})
        self.assertEqual(self.db.users.users_info, {1: [{'attr_type': 'git', 'attr_data': 'Tyler Durden <tyler@mayhem.net>'}]})

    @defer.inlineCallbacks
    def test_createUserObject_svn(self):
        if False:
            while True:
                i = 10
        yield users.createUserObject(self.master, 'tdurden', 'svn')
        self.assertEqual(self.db.users.users, {1: {'identifier': 'tdurden', 'bb_username': None, 'bb_password': None}})
        self.assertEqual(self.db.users.users_info, {1: [{'attr_type': 'svn', 'attr_data': 'tdurden'}]})

    @defer.inlineCallbacks
    def test_createUserObject_hg(self):
        if False:
            for i in range(10):
                print('nop')
        yield users.createUserObject(self.master, 'Tyler Durden <tyler@mayhem.net>', 'hg')
        self.assertEqual(self.db.users.users, {1: {'identifier': 'Tyler Durden <tyler@mayhem.net>', 'bb_username': None, 'bb_password': None}})
        self.assertEqual(self.db.users.users_info, {1: [{'attr_type': 'hg', 'attr_data': 'Tyler Durden <tyler@mayhem.net>'}]})

    @defer.inlineCallbacks
    def test_createUserObject_cvs(self):
        if False:
            return 10
        yield users.createUserObject(self.master, 'tdurden', 'cvs')
        self.assertEqual(self.db.users.users, {1: {'identifier': 'tdurden', 'bb_username': None, 'bb_password': None}})
        self.assertEqual(self.db.users.users_info, {1: [{'attr_type': 'cvs', 'attr_data': 'tdurden'}]})

    @defer.inlineCallbacks
    def test_createUserObject_darcs(self):
        if False:
            while True:
                i = 10
        yield users.createUserObject(self.master, 'tyler@mayhem.net', 'darcs')
        self.assertEqual(self.db.users.users, {1: {'identifier': 'tyler@mayhem.net', 'bb_username': None, 'bb_password': None}})
        self.assertEqual(self.db.users.users_info, {1: [{'attr_type': 'darcs', 'attr_data': 'tyler@mayhem.net'}]})

    @defer.inlineCallbacks
    def test_createUserObject_bzr(self):
        if False:
            print('Hello World!')
        yield users.createUserObject(self.master, 'Tyler Durden', 'bzr')
        self.assertEqual(self.db.users.users, {1: {'identifier': 'Tyler Durden', 'bb_username': None, 'bb_password': None}})
        self.assertEqual(self.db.users.users_info, {1: [{'attr_type': 'bzr', 'attr_data': 'Tyler Durden'}]})

    @defer.inlineCallbacks
    def test_getUserContact_found(self):
        if False:
            print('Hello World!')
        self.db.insert_test_data([fakedb.User(uid=1, identifier='tdurden'), fakedb.UserInfo(uid=1, attr_type='svn', attr_data='tdurden'), fakedb.UserInfo(uid=1, attr_type='email', attr_data='tyler@mayhem.net')])
        contact = (yield users.getUserContact(self.master, contact_types=['email'], uid=1))
        self.assertEqual(contact, 'tyler@mayhem.net')

    @defer.inlineCallbacks
    def test_getUserContact_key_not_found(self):
        if False:
            i = 10
            return i + 15
        self.db.insert_test_data([fakedb.User(uid=1, identifier='tdurden'), fakedb.UserInfo(uid=1, attr_type='svn', attr_data='tdurden'), fakedb.UserInfo(uid=1, attr_type='email', attr_data='tyler@mayhem.net')])
        contact = (yield users.getUserContact(self.master, contact_types=['blargh'], uid=1))
        self.assertEqual(contact, None)

    @defer.inlineCallbacks
    def test_getUserContact_uid_not_found(self):
        if False:
            return 10
        contact = (yield users.getUserContact(self.master, contact_types=['email'], uid=1))
        self.assertEqual(contact, None)

    def test_check_passwd(self):
        if False:
            for i in range(10):
                print('nop')
        res = users.check_passwd('cancer', self.test_sha)
        self.assertEqual(res, True)