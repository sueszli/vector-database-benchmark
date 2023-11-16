from twisted.internet import defer
from twisted.trial import unittest
from buildbot.test.fake.web import FakeRequest
from buildbot.test.fake.web import fakeMasterForHooks
from buildbot.test.reactor import TestReactorMixin
from buildbot.www import change_hook
gitJsonPayload = b'\n{\n  "after": "df5744f7bc8663b39717f87742dc94f52ccbf4dd",\n  "before": "b4ca2d38e756695133cbd0e03d078804e1dc6610",\n  "commits": [\n    {\n      "author": {\n        "email": "jason@nospam.org",\n        "name": "jason"\n      },\n      "committed_at": "2012-01-10T11:02:27-07:00",\n      "id": "df5744f7bc8663b39717f87742dc94f52ccbf4dd",\n      "message": "added a place to put the docstring for Book",\n      "timestamp": "2012-01-10T11:02:27-07:00",\n      "url": "http://gitorious.org/q/mainline/commit/df5744f7bc8663b39717f87742dc94f52ccbf4dd"\n    }\n  ],\n  "project": {\n    "description": "a webapp to organize your ebook collectsion.",\n    "name": "q"\n  },\n  "pushed_at": "2012-01-10T11:09:25-07:00",\n  "pushed_by": "jason",\n  "ref": "new_look",\n  "repository": {\n    "clones": 4,\n    "description": "",\n    "name": "mainline",\n    "owner": {\n      "name": "jason"\n    },\n    "url": "http://gitorious.org/q/mainline"\n  }\n}\n'

class TestChangeHookConfiguredWithGitChange(unittest.TestCase, TestReactorMixin):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        dialects = {'gitorious': True}
        self.changeHook = change_hook.ChangeHookResource(dialects=dialects, master=fakeMasterForHooks(self))

    @defer.inlineCallbacks
    def testGitWithChange(self):
        if False:
            print('Hello World!')
        changeDict = {b'payload': [gitJsonPayload]}
        self.request = FakeRequest(changeDict)
        self.request.uri = b'/change_hook/gitorious'
        self.request.method = b'POST'
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['files'], [])
        self.assertEqual(change['repository'], 'http://gitorious.org/q/mainline')
        self.assertEqual(change['when_timestamp'], 1326218547)
        self.assertEqual(change['author'], 'jason <jason@nospam.org>')
        self.assertEqual(change['revision'], 'df5744f7bc8663b39717f87742dc94f52ccbf4dd')
        self.assertEqual(change['comments'], 'added a place to put the docstring for Book')
        self.assertEqual(change['branch'], 'new_look')
        revlink = 'http://gitorious.org/q/mainline/commit/df5744f7bc8663b39717f87742dc94f52ccbf4dd'
        self.assertEqual(change['revlink'], revlink)

    @defer.inlineCallbacks
    def testGitWithNoJson(self):
        if False:
            for i in range(10):
                print('nop')
        self.request = FakeRequest()
        self.request.uri = b'/change_hook/gitorious'
        self.request.method = b'GET'
        yield self.request.test_render(self.changeHook)
        expected = b'Error processing changes.'
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)
        self.request.setResponseCode.assert_called_with(500, expected)
        self.assertEqual(len(self.flushLoggedErrors()), 1)