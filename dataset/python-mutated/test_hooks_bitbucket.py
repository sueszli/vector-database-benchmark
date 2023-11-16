from twisted.internet.defer import inlineCallbacks
from twisted.trial import unittest
from buildbot.test.fake.web import FakeRequest
from buildbot.test.fake.web import fakeMasterForHooks
from buildbot.test.reactor import TestReactorMixin
from buildbot.www import change_hook
from buildbot.www.hooks.bitbucket import _HEADER_EVENT
gitJsonPayload = b'{\n    "canon_url": "https://bitbucket.org",\n    "commits": [\n        {\n            "author": "marcus",\n            "branch": "master",\n            "files": [\n                {\n                    "file": "somefile.py",\n                    "type": "modified"\n                }\n            ],\n            "message": "Added some more things to somefile.py",\n            "node": "620ade18607a",\n            "parents": [\n                "702c70160afc"\n            ],\n            "raw_author": "Marcus Bertrand <marcus@somedomain.com>",\n            "raw_node": "620ade18607ac42d872b568bb92acaa9a28620e9",\n            "revision": null,\n            "size": -1,\n            "timestamp": "2012-05-30 05:58:56",\n            "utctimestamp": "2012-05-30 03:58:56+00:00"\n        }\n    ],\n    "repository": {\n        "absolute_url": "/marcus/project-x/",\n        "fork": false,\n        "is_private": true,\n        "name": "Project X",\n        "owner": "marcus",\n        "scm": "git",\n        "slug": "project-x",\n        "website": "https://atlassian.com/"\n    },\n    "user": "marcus"\n}'
mercurialJsonPayload = b'{\n    "canon_url": "https://bitbucket.org",\n    "commits": [\n        {\n            "author": "marcus",\n            "branch": "master",\n            "files": [\n                {\n                    "file": "somefile.py",\n                    "type": "modified"\n                }\n            ],\n            "message": "Added some more things to somefile.py",\n            "node": "620ade18607a",\n            "parents": [\n                "702c70160afc"\n            ],\n            "raw_author": "Marcus Bertrand <marcus@somedomain.com>",\n            "raw_node": "620ade18607ac42d872b568bb92acaa9a28620e9",\n            "revision": null,\n            "size": -1,\n            "timestamp": "2012-05-30 05:58:56",\n            "utctimestamp": "2012-05-30 03:58:56+00:00"\n        }\n    ],\n    "repository": {\n        "absolute_url": "/marcus/project-x/",\n        "fork": false,\n        "is_private": true,\n        "name": "Project X",\n        "owner": "marcus",\n        "scm": "hg",\n        "slug": "project-x",\n        "website": "https://atlassian.com/"\n    },\n    "user": "marcus"\n}'
gitJsonNoCommitsPayload = b'{\n    "canon_url": "https://bitbucket.org",\n    "commits": [\n    ],\n    "repository": {\n        "absolute_url": "/marcus/project-x/",\n        "fork": false,\n        "is_private": true,\n        "name": "Project X",\n        "owner": "marcus",\n        "scm": "git",\n        "slug": "project-x",\n        "website": "https://atlassian.com/"\n    },\n    "user": "marcus"\n}'
mercurialJsonNoCommitsPayload = b'{\n    "canon_url": "https://bitbucket.org",\n    "commits": [\n    ],\n    "repository": {\n        "absolute_url": "/marcus/project-x/",\n        "fork": false,\n        "is_private": true,\n        "name": "Project X",\n        "owner": "marcus",\n        "scm": "hg",\n        "slug": "project-x",\n        "website": "https://atlassian.com/"\n    },\n    "user": "marcus"\n}'

class TestChangeHookConfiguredWithBitbucketChange(unittest.TestCase, TestReactorMixin):
    """Unit tests for BitBucket Change Hook
    """

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.change_hook = change_hook.ChangeHookResource(dialects={'bitbucket': True}, master=fakeMasterForHooks(self))

    @inlineCallbacks
    def testGitWithChange(self):
        if False:
            print('Hello World!')
        change_dict = {b'payload': [gitJsonPayload]}
        request = FakeRequest(change_dict)
        request.received_headers[_HEADER_EVENT] = b'repo:push'
        request.uri = b'/change_hook/bitbucket'
        request.method = b'POST'
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        commit = self.change_hook.master.data.updates.changesAdded[0]
        self.assertEqual(commit['files'], ['somefile.py'])
        self.assertEqual(commit['repository'], 'https://bitbucket.org/marcus/project-x/')
        self.assertEqual(commit['when_timestamp'], 1338350336)
        self.assertEqual(commit['author'], 'Marcus Bertrand <marcus@somedomain.com>')
        self.assertEqual(commit['revision'], '620ade18607ac42d872b568bb92acaa9a28620e9')
        self.assertEqual(commit['comments'], 'Added some more things to somefile.py')
        self.assertEqual(commit['branch'], 'master')
        self.assertEqual(commit['revlink'], 'https://bitbucket.org/marcus/project-x/commits/620ade18607ac42d872b568bb92acaa9a28620e9')
        self.assertEqual(commit['properties']['event'], 'repo:push')

    @inlineCallbacks
    def testGitWithNoCommitsPayload(self):
        if False:
            for i in range(10):
                print('nop')
        change_dict = {b'payload': [gitJsonNoCommitsPayload]}
        request = FakeRequest(change_dict)
        request.uri = b'/change_hook/bitbucket'
        request.method = b'POST'
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 0)
        self.assertEqual(request.written, b'no change found')

    @inlineCallbacks
    def testMercurialWithChange(self):
        if False:
            i = 10
            return i + 15
        change_dict = {b'payload': [mercurialJsonPayload]}
        request = FakeRequest(change_dict)
        request.received_headers[_HEADER_EVENT] = b'repo:push'
        request.uri = b'/change_hook/bitbucket'
        request.method = b'POST'
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        commit = self.change_hook.master.data.updates.changesAdded[0]
        self.assertEqual(commit['files'], ['somefile.py'])
        self.assertEqual(commit['repository'], 'https://bitbucket.org/marcus/project-x/')
        self.assertEqual(commit['when_timestamp'], 1338350336)
        self.assertEqual(commit['author'], 'Marcus Bertrand <marcus@somedomain.com>')
        self.assertEqual(commit['revision'], '620ade18607ac42d872b568bb92acaa9a28620e9')
        self.assertEqual(commit['comments'], 'Added some more things to somefile.py')
        self.assertEqual(commit['branch'], 'master')
        self.assertEqual(commit['revlink'], 'https://bitbucket.org/marcus/project-x/commits/620ade18607ac42d872b568bb92acaa9a28620e9')
        self.assertEqual(commit['properties']['event'], 'repo:push')

    @inlineCallbacks
    def testMercurialWithNoCommitsPayload(self):
        if False:
            return 10
        change_dict = {b'payload': [mercurialJsonNoCommitsPayload]}
        request = FakeRequest(change_dict)
        request.uri = b'/change_hook/bitbucket'
        request.method = b'POST'
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 0)
        self.assertEqual(request.written, b'no change found')

    @inlineCallbacks
    def testWithNoJson(self):
        if False:
            while True:
                i = 10
        request = FakeRequest()
        request.uri = b'/change_hook/bitbucket'
        request.method = b'POST'
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 0)
        self.assertEqual(request.written, b'Error processing changes.')
        request.setResponseCode.assert_called_with(500, b'Error processing changes.')
        self.assertEqual(len(self.flushLoggedErrors()), 1)

    @inlineCallbacks
    def testGitWithChangeAndProject(self):
        if False:
            print('Hello World!')
        change_dict = {b'payload': [gitJsonPayload], b'project': [b'project-name']}
        request = FakeRequest(change_dict)
        request.uri = b'/change_hook/bitbucket'
        request.method = b'POST'
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        commit = self.change_hook.master.data.updates.changesAdded[0]
        self.assertEqual(commit['project'], 'project-name')