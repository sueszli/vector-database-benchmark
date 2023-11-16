from twisted.trial import unittest
from buildbot.util import giturlparse

class Tests(unittest.TestCase):

    def test_github(self):
        if False:
            i = 10
            return i + 15
        for u in ['https://github.com/buildbot/buildbot', 'https://github.com/buildbot/buildbot.git', 'ssh://git@github.com:buildbot/buildbot.git', 'git://github.com/buildbot/buildbot.git']:
            u = giturlparse(u)
            self.assertIn(u.user, (None, 'git'))
            self.assertEqual(u.domain, 'github.com')
            self.assertEqual(u.owner, 'buildbot')
            self.assertEqual(u.repo, 'buildbot')
            self.assertIsNone(u.port)

    def test_gitlab(self):
        if False:
            return 10
        for u in ['ssh://git@mygitlab.com/group/subgrouptest/testproject.git', 'https://mygitlab.com/group/subgrouptest/testproject.git', 'git@mygitlab.com:group/subgrouptest/testproject.git', 'git://mygitlab.com/group/subgrouptest/testproject.git']:
            u = giturlparse(u)
            self.assertIsNone(u.port)
            self.assertIn(u.user, (None, 'git'))
            self.assertEqual(u.domain, 'mygitlab.com')
            self.assertEqual(u.owner, 'group/subgrouptest')
            self.assertEqual(u.repo, 'testproject')

    def test_gitlab_subsubgroup(self):
        if False:
            i = 10
            return i + 15
        for u in ['ssh://git@mygitlab.com/group/subgrouptest/subsubgroup/testproject.git', 'https://mygitlab.com/group/subgrouptest/subsubgroup/testproject.git', 'git://mygitlab.com/group/subgrouptest/subsubgroup/testproject.git']:
            u = giturlparse(u)
            self.assertIn(u.user, (None, 'git'))
            self.assertIsNone(u.port)
            self.assertEqual(u.domain, 'mygitlab.com')
            self.assertEqual(u.owner, 'group/subgrouptest/subsubgroup')
            self.assertEqual(u.repo, 'testproject')

    def test_gitlab_user(self):
        if False:
            for i in range(10):
                print('nop')
        for u in ['ssh://buildbot@mygitlab.com:group/subgrouptest/testproject.git', 'https://buildbot@mygitlab.com/group/subgrouptest/testproject.git']:
            u = giturlparse(u)
            self.assertEqual(u.domain, 'mygitlab.com')
            self.assertIsNone(u.port)
            self.assertEqual(u.user, 'buildbot')
            self.assertEqual(u.owner, 'group/subgrouptest')
            self.assertEqual(u.repo, 'testproject')

    def test_gitlab_port(self):
        if False:
            for i in range(10):
                print('nop')
        for u in ['ssh://buildbot@mygitlab.com:1234/group/subgrouptest/testproject.git']:
            u = giturlparse(u)
            self.assertEqual(u.domain, 'mygitlab.com')
            self.assertEqual(u.port, 1234)
            self.assertEqual(u.user, 'buildbot')
            self.assertEqual(u.owner, 'group/subgrouptest')
            self.assertEqual(u.repo, 'testproject')

    def test_bitbucket(self):
        if False:
            while True:
                i = 10
        for u in ['https://bitbucket.org/org/repo.git', 'ssh://git@bitbucket.org:org/repo.git', 'git@bitbucket.org:org/repo.git']:
            u = giturlparse(u)
            self.assertIn(u.user, (None, 'git'))
            self.assertEqual(u.domain, 'bitbucket.org')
            self.assertEqual(u.owner, 'org')
            self.assertEqual(u.repo, 'repo')

    def test_no_owner(self):
        if False:
            while True:
                i = 10
        for u in ['https://example.org/repo.git', 'ssh://example.org:repo.git', 'ssh://git@example.org:repo.git', 'git@example.org:repo.git']:
            u = giturlparse(u)
            self.assertIn(u.user, (None, 'git'))
            self.assertEqual(u.domain, 'example.org')
            self.assertIsNone(u.owner)
            self.assertEqual(u.repo, 'repo')

    def test_protos(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(giturlparse('https://bitbucket.org/org/repo.git').proto, 'https')
        self.assertEqual(giturlparse('git://bitbucket.org/org/repo.git').proto, 'git')
        self.assertEqual(giturlparse('ssh://git@bitbucket.org:org/repo.git').proto, 'ssh')
        self.assertEqual(giturlparse('git@bitbucket.org:org/repo.git').proto, 'ssh')