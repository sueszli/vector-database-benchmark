import hmac
from copy import deepcopy
from hashlib import sha1
from io import BytesIO
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.plugins import util
from buildbot.secrets.manager import SecretManager
from buildbot.test.fake import httpclientservice as fakehttpclientservice
from buildbot.test.fake.secrets import FakeSecretStorage
from buildbot.test.fake.web import FakeRequest
from buildbot.test.fake.web import fakeMasterForHooks
from buildbot.test.reactor import TestReactorMixin
from buildbot.util import unicode2bytes
from buildbot.www.change_hook import ChangeHookResource
from buildbot.www.hooks.github import _HEADER_EVENT
from buildbot.www.hooks.github import _HEADER_SIGNATURE
from buildbot.www.hooks.github import GitHubEventHandler
gitJsonPayload = b'\n{\n  "before": "5aef35982fb2d34e9d9d4502f6ede1072793222d",\n  "repository": {\n    "url": "http://github.com/defunkt/github",\n    "html_url": "http://github.com/defunkt/github",\n    "name": "github",\n    "full_name": "defunkt/github",\n    "description": "You\'re lookin\' at it.",\n    "watchers": 5,\n    "forks": 2,\n    "private": 1,\n    "owner": {\n      "email": "fred@flinstone.org",\n      "name": "defunkt"\n    }\n  },\n  "commits": [\n    {\n      "id": "41a212ee83ca127e3c8cf465891ab7216a705f59",\n      "distinct": true,\n      "url": "http://github.com/defunkt/github/commit/41a212ee83ca127e3c8cf465891ab7216a705f59",\n      "author": {\n        "email": "fred@flinstone.org",\n        "name": "Fred Flinstone"\n      },\n      "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n      },\n      "message": "okay i give in",\n      "timestamp": "2008-02-15T14:57:17-08:00",\n      "added": ["filepath.rb"]\n    },\n    {\n      "id": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n      "url": "http://github.com/defunkt/github/commit/de8251ff97ee194a289832576287d6f8ad74e3d0",\n      "author": {\n        "email": "fred@flinstone.org",\n        "name": "Fred Flinstone"\n      },\n      "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n      },\n      "message": "update pricing a tad",\n      "timestamp": "2008-02-15T14:36:34-08:00",\n      "modified": ["modfile"],\n      "removed": ["removedFile"]\n    }\n  ],\n  "head_commit": {\n    "id": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n    "url": "http://github.com/defunkt/github/commit/de8251ff97ee194a289832576287d6f8ad74e3d0",\n    "author": {\n      "email": "fred@flinstone.org",\n      "name": "Fred Flinstone"\n    },\n    "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n    },\n    "message": "update pricing a tad",\n    "timestamp": "2008-02-15T14:36:34-08:00",\n    "modified": ["modfile"],\n    "removed": ["removedFile"]\n  },\n  "after": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n  "ref": "refs/heads/master"\n}\n'
gitJsonPayloadCiSkipTemplate = '\n{\n  "before": "5aef35982fb2d34e9d9d4502f6ede1072793222d",\n  "repository": {\n    "url": "http://github.com/defunkt/github",\n    "html_url": "http://github.com/defunkt/github",\n    "name": "github",\n    "full_name": "defunkt/github",\n    "description": "You\'re lookin\' at it.",\n    "watchers": 5,\n    "forks": 2,\n    "private": 1,\n    "owner": {\n      "email": "fred@flinstone.org",\n      "name": "defunkt"\n    }\n  },\n  "commits": [\n    {\n      "id": "41a212ee83ca127e3c8cf465891ab7216a705f59",\n      "distinct": true,\n      "url": "http://github.com/defunkt/github/commit/41a212ee83ca127e3c8cf465891ab7216a705f59",\n      "author": {\n        "email": "fred@flinstone.org",\n        "name": "Fred Flinstone"\n      },\n      "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n      },\n      "message": "okay i give in",\n      "timestamp": "2008-02-15T14:57:17-08:00",\n      "added": ["filepath.rb"]\n    },\n    {\n      "id": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n      "url": "http://github.com/defunkt/github/commit/de8251ff97ee194a289832576287d6f8ad74e3d0",\n      "author": {\n        "email": "fred@flinstone.org",\n        "name": "Fred Flinstone"\n      },\n      "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n      },\n      "message": "update pricing a tad %(skip)s",\n      "timestamp": "2008-02-15T14:36:34-08:00",\n      "modified": ["modfile"],\n      "removed": ["removedFile"]\n    }\n  ],\n  "head_commit": {\n    "id": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n    "url": "http://github.com/defunkt/github/commit/de8251ff97ee194a289832576287d6f8ad74e3d0",\n    "author": {\n      "email": "fred@flinstone.org",\n      "name": "Fred Flinstone"\n    },\n    "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n    },\n    "message": "update pricing a tad %(skip)s",\n    "timestamp": "2008-02-15T14:36:34-08:00",\n    "modified": ["modfile"],\n    "removed": ["removedFile"]\n  },\n  "after": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n  "ref": "refs/heads/master"\n}\n'
gitJsonPayloadTag = b'\n{\n  "before": "5aef35982fb2d34e9d9d4502f6ede1072793222d",\n  "repository": {\n    "url": "http://github.com/defunkt/github",\n    "html_url": "http://github.com/defunkt/github",\n    "name": "github",\n    "full_name": "defunkt/github",\n    "description": "You\'re lookin\' at it.",\n    "watchers": 5,\n    "forks": 2,\n    "private": 1,\n    "owner": {\n      "email": "fred@flinstone.org",\n      "name": "defunkt"\n    }\n  },\n  "commits": [\n    {\n      "id": "41a212ee83ca127e3c8cf465891ab7216a705f59",\n      "distinct": true,\n      "url": "http://github.com/defunkt/github/commit/41a212ee83ca127e3c8cf465891ab7216a705f59",\n      "author": {\n        "email": "fred@flinstone.org",\n        "name": "Fred Flinstone"\n      },\n      "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n      },\n      "message": "okay i give in",\n      "timestamp": "2008-02-15T14:57:17-08:00",\n      "added": ["filepath.rb"]\n    },\n    {\n      "id": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n      "url": "http://github.com/defunkt/github/commit/de8251ff97ee194a289832576287d6f8ad74e3d0",\n      "author": {\n        "email": "fred@flinstone.org",\n        "name": "Fred Flinstone"\n      },\n      "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n      },\n      "message": "update pricing a tad",\n      "timestamp": "2008-02-15T14:36:34-08:00",\n      "modified": ["modfile"],\n      "removed": ["removedFile"]\n    }\n  ],\n  "head_commit": {\n    "id": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n    "url": "http://github.com/defunkt/github/commit/de8251ff97ee194a289832576287d6f8ad74e3d0",\n    "author": {\n      "email": "fred@flinstone.org",\n      "name": "Fred Flinstone"\n    },\n    "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n    },\n    "message": "update pricing a tad",\n    "timestamp": "2008-02-15T14:36:34-08:00",\n    "modified": ["modfile"],\n    "removed": ["removedFile"]\n  },\n  "after": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n  "ref": "refs/tags/v1.0.0"\n}\n'
gitJsonPayloadNonBranch = b'\n{\n  "before": "5aef35982fb2d34e9d9d4502f6ede1072793222d",\n  "repository": {\n    "url": "http://github.com/defunkt/github",\n    "html_url": "http://github.com/defunkt/github",\n    "name": "github",\n    "full_name": "defunkt/github",\n    "description": "You\'re lookin\' at it.",\n    "watchers": 5,\n    "forks": 2,\n    "private": 1,\n    "owner": {\n      "email": "fred@flinstone.org",\n      "name": "defunkt"\n    }\n  },\n  "commits": [\n    {\n      "id": "41a212ee83ca127e3c8cf465891ab7216a705f59",\n      "distinct": true,\n      "url": "http://github.com/defunkt/github/commit/41a212ee83ca127e3c8cf465891ab7216a705f59",\n      "author": {\n        "email": "fred@flinstone.org",\n        "name": "Fred Flinstone"\n      },\n      "committer": {\n        "email": "freddy@flinstone.org",\n        "name": "Freddy Flinstone"\n      },\n      "message": "okay i give in",\n      "timestamp": "2008-02-15T14:57:17-08:00",\n      "added": ["filepath.rb"]\n    }\n  ],\n  "after": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n  "ref": "refs/garbage/master"\n}\n'
gitJsonPayloadPullRequest = b'\n{\n  "action": "opened",\n  "number": 50,\n  "pull_request": {\n    "url": "https://api.github.com/repos/defunkt/github/pulls/50",\n    "html_url": "https://github.com/defunkt/github/pull/50",\n    "number": 50,\n    "state": "open",\n    "title": "Update the README with new information",\n    "user": {\n      "login": "defunkt",\n      "id": 42,\n      "type": "User"\n    },\n    "body": "This is a pretty simple change that we need to pull into master.",\n    "created_at": "2014-10-10T00:09:50Z",\n    "updated_at": "2014-10-10T00:09:50Z",\n    "closed_at": null,\n    "merged_at": null,\n    "merge_commit_sha": "cd3ff078a350901f91f4c4036be74f91d0b0d5d6",\n    "head": {\n      "label": "defunkt:changes",\n      "ref": "changes",\n      "sha": "05c588ba8cd510ecbe112d020f215facb17817a7",\n      "user": {\n        "login": "defunkt",\n        "id": 42,\n        "type": "User"\n      },\n      "repo": {\n        "id": 43,\n        "name": "github",\n        "full_name": "defunkt/github",\n        "owner": {\n          "login": "defunkt",\n          "id": 42,\n          "type": "User"\n        },\n        "html_url": "https://github.com/defunkt/github",\n        "description": "",\n        "url": "https://api.github.com/repos/defunkt/github",\n        "created_at": "2014-05-20T22:39:43Z",\n        "updated_at": "2014-07-25T16:37:51Z",\n        "pushed_at": "2014-10-10T00:09:49Z",\n        "git_url": "git://github.com/defunkt/github.git",\n        "ssh_url": "git@github.com:defunkt/github.git",\n        "clone_url": "https://github.com/defunkt/github.git",\n        "default_branch": "master"\n      }\n    },\n    "base": {\n      "label": "defunkt:master",\n      "ref": "master",\n      "sha": "69a8b72e2d3d955075d47f03d902929dcaf74034",\n      "user": {\n        "login": "defunkt",\n        "id": 42,\n        "type": "User"\n      },\n      "repo": {\n        "id": 43,\n        "name": "github",\n        "full_name": "defunkt/github",\n        "owner": {\n          "login": "defunkt",\n          "id": 42,\n          "type": "User"\n        },\n        "html_url": "https://github.com/defunkt/github",\n        "description": "",\n        "url": "https://api.github.com/repos/defunkt/github",\n        "created_at": "2014-05-20T22:39:43Z",\n        "updated_at": "2014-07-25T16:37:51Z",\n        "pushed_at": "2014-10-10T00:09:49Z",\n        "git_url": "git://github.com/defunkt/github.git",\n        "ssh_url": "git@github.com:defunkt/github.git",\n        "clone_url": "https://github.com/defunkt/github.git",\n        "default_branch": "master"\n      }\n    },\n    "_links": {\n      "self": {\n        "href": "https://api.github.com/repos/defunkt/github/pulls/50"\n      },\n      "html": {\n        "href": "https://github.com/defunkt/github/pull/50"\n      },\n      "commits": {\n        "href": "https://api.github.com/repos/defunkt/github/pulls/50/commits"\n      }\n    },\n    "commits": 1,\n    "additions": 2,\n    "deletions": 0,\n    "changed_files": 1\n  },\n  "repository": {\n    "id": 43,\n    "name": "github",\n    "full_name": "defunkt/github",\n    "owner": {\n      "login": "defunkt",\n      "id": 42,\n      "type": "User"\n    },\n    "html_url": "https://github.com/defunkt/github",\n    "description": "",\n    "url": "https://api.github.com/repos/defunkt/github",\n    "created_at": "2014-05-20T22:39:43Z",\n    "updated_at": "2014-07-25T16:37:51Z",\n    "pushed_at": "2014-10-10T00:09:49Z",\n    "git_url": "git://github.com/defunkt/github.git",\n    "ssh_url": "git@github.com:defunkt/github.git",\n    "clone_url": "https://github.com/defunkt/github.git",\n    "default_branch": "master"\n  },\n  "sender": {\n    "login": "defunkt",\n    "id": 42,\n    "type": "User"\n  }\n}\n'
gitJsonPayloadCommit = {'sha': 'de8251ff97ee194a289832576287d6f8ad74e3d0', 'commit': {'author': {'name': 'defunkt', 'email': 'fred@flinstone.org', 'date': '2017-02-12T14:39:33Z'}, 'committer': {'name': 'defunkt', 'email': 'fred@flinstone.org', 'date': '2017-02-12T14:51:05Z'}, 'message': 'black magic', 'tree': {}, 'url': '...', 'comment_count': 0}, 'url': '...', 'html_url': '...', 'comments_url': '...', 'author': {}, 'committer': {}, 'parents': [], 'stats': {}, 'files': []}
gitJsonPayloadFiles = [{'filename': 'README.md', 'previous_filename': 'old_README.md'}]
gitPRproperties = {'pullrequesturl': 'https://github.com/defunkt/github/pull/50', 'github.head.sha': '05c588ba8cd510ecbe112d020f215facb17817a7', 'github.state': 'open', 'github.base.repo.full_name': 'defunkt/github', 'github.number': 50, 'github.base.ref': 'master', 'github.base.sha': '69a8b72e2d3d955075d47f03d902929dcaf74034', 'github.head.repo.full_name': 'defunkt/github', 'github.merged_at': None, 'github.head.ref': 'changes', 'github.closed_at': None, 'github.title': 'Update the README with new information', 'event': 'pull_request'}
gitJsonPayloadEmpty = b'\n{\n  "before": "5aef35982fb2d34e9d9d4502f6ede1072793222d",\n  "repository": {\n    "url": "http://github.com/defunkt/github",\n    "html_url": "http://github.com/defunkt/github",\n    "name": "github",\n    "full_name": "defunkt/github",\n    "description": "You\'re lookin\' at it.",\n    "watchers": 5,\n    "forks": 2,\n    "private": 1,\n    "owner": {\n      "email": "fred@flinstone.org",\n      "name": "defunkt"\n    }\n  },\n  "commits": [\n  ],\n  "head_commit": {\n  },\n  "after": "de8251ff97ee194a289832576287d6f8ad74e3d0",\n  "ref": "refs/heads/master"\n}\n'
gitJsonPayloadCreateTag = b'\n{\n  "ref": "refs/tags/v0.9.15.post1",\n  "before": "0000000000000000000000000000000000000000",\n  "after": "ffe1e9affb2b5399369443194c02068032f9295e",\n  "created": true,\n  "deleted": false,\n  "forced": false,\n  "base_ref": null,\n  "compare": "https://github.com/buildbot/buildbot/compare/v0.9.15.post1",\n  "commits": [\n\n  ],\n  "head_commit": {\n    "id": "57df618a4a450410c1dee440c7827ee105f5a226",\n    "tree_id": "f9768673dc968b5c8fcbb15f119ce237b50b3252",\n    "distinct": true,\n    "message": "...",\n    "timestamp": "2018-01-07T16:30:52+01:00",\n    "url": "https://github.com/buildbot/buildbot/commit/...",\n    "author": {\n      "name": "User",\n      "email": "userid@example.com",\n      "username": "userid"\n    },\n    "committer": {\n      "name": "GitHub",\n      "email": "noreply@github.com",\n      "username": "web-flow"\n    },\n    "added": [\n\n    ],\n    "removed": [\n      "master/buildbot/newsfragments/bit_length.bugfix",\n      "master/buildbot/newsfragments/localworker_umask.bugfix",\n      "master/buildbot/newsfragments/svn-utf8.bugfix"\n    ],\n    "modified": [\n      ".bbtravis.yml",\n      "circle.yml",\n      "master/docs/relnotes/index.rst"\n    ]\n  },\n  "repository": {\n    "html_url": "https://github.com/buildbot/buildbot",\n    "name": "buildbot",\n    "full_name": "buildbot"\n  },\n  "pusher": {\n    "name": "userid",\n    "email": "userid@example.com"\n  },\n  "organization": {\n    "login": "buildbot",\n    "url": "https://api.github.com/orgs/buildbot",\n    "description": "Continous integration and delivery framework"\n  },\n  "sender": {\n    "login": "userid",\n    "gravatar_id": "",\n    "type": "User",\n    "site_admin": false\n  },\n  "ref_name": "v0.9.15.post1",\n  "distinct_commits": [\n\n  ]\n}'
gitJsonPayloadNotFound = b'{"message":"Not Found"}'
_HEADER_CT = b'Content-Type'
_CT_ENCODED = b'application/x-www-form-urlencoded'
_CT_JSON = b'application/json'

def _prepare_github_change_hook(testcase, **params):
    if False:
        for i in range(10):
            print('nop')
    return ChangeHookResource(dialects={'github': params}, master=fakeMasterForHooks(testcase))

def _prepare_request(event, payload, _secret=None, headers=None):
    if False:
        i = 10
        return i + 15
    if headers is None:
        headers = {}
    request = FakeRequest()
    request.uri = b'/change_hook/github'
    request.method = b'GET'
    request.received_headers = {_HEADER_EVENT: event}
    assert isinstance(payload, (bytes, list)), f'payload can only be bytes or list, not {type(payload)}'
    if isinstance(payload, bytes):
        request.content = BytesIO(payload)
        request.received_headers[_HEADER_CT] = _CT_JSON
        if _secret is not None:
            signature = hmac.new(unicode2bytes(_secret), msg=unicode2bytes(payload), digestmod=sha1)
            request.received_headers[_HEADER_SIGNATURE] = f'sha1={signature.hexdigest()}'
    else:
        request.args[b'payload'] = payload
        request.received_headers[_HEADER_CT] = _CT_ENCODED
    request.received_headers.update(headers)
    return request

class TestChangeHookConfiguredWithGitChange(unittest.TestCase, TestReactorMixin):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.changeHook = _prepare_github_change_hook(self, strict=False, github_property_whitelist=['github.*'])
        self.master = self.changeHook.master
        fake_headers = {'User-Agent': 'Buildbot'}
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'https://api.github.com', headers=fake_headers, debug=False, verify=False))
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            while True:
                i = 10
        yield self.master.stopService()

    def assertDictSubset(self, expected_dict, response_dict):
        if False:
            return 10
        expected = {}
        for key in expected_dict.keys():
            self.assertIn(key, set(response_dict.keys()))
            expected[key] = response_dict[key]
        self.assertDictEqual(expected_dict, expected)

    @defer.inlineCallbacks
    def test_unknown_event(self):
        if False:
            print('Hello World!')
        bad_event = b'whatever'
        self.request = _prepare_request(bad_event, gitJsonPayload)
        yield self.request.test_render(self.changeHook)
        expected = b'Unknown event: ' + bad_event
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)

    @defer.inlineCallbacks
    def test_unknown_content_type(self):
        if False:
            i = 10
            return i + 15
        bad_content_type = b'application/x-useful'
        self.request = _prepare_request(b'push', gitJsonPayload, headers={_HEADER_CT: bad_content_type})
        yield self.request.test_render(self.changeHook)
        expected = b'Unknown content type: '
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertIn(expected, self.request.written)

    @defer.inlineCallbacks
    def _check_ping(self, payload):
        if False:
            return 10
        self.request = _prepare_request(b'ping', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)

    def test_ping_encoded(self):
        if False:
            return 10
        self._check_ping([b'{}'])

    def test_ping_json(self):
        if False:
            i = 10
            return i + 15
        self._check_ping(b'{}')

    @defer.inlineCallbacks
    def test_git_with_push_tag(self):
        if False:
            print('Hello World!')
        self.request = _prepare_request(b'push', gitJsonPayloadTag)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['author'], 'Fred Flinstone <fred@flinstone.org>')
        self.assertEqual(change['committer'], 'Freddy Flinstone <freddy@flinstone.org>')
        self.assertEqual(change['branch'], 'v1.0.0')
        self.assertEqual(change['category'], 'tag')

    @defer.inlineCallbacks
    def test_git_with_push_newtag(self):
        if False:
            i = 10
            return i + 15
        self.request = _prepare_request(b'push', gitJsonPayloadCreateTag)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['author'], 'User <userid@example.com>')
        self.assertEqual(change['branch'], 'v0.9.15.post1')
        self.assertEqual(change['category'], 'tag')

    @defer.inlineCallbacks
    def _check_git_with_change(self, payload):
        if False:
            print('Hello World!')
        self.request = _prepare_request(b'push', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['files'], ['filepath.rb'])
        self.assertEqual(change['repository'], 'http://github.com/defunkt/github')
        self.assertEqual(change['when_timestamp'], 1203116237)
        self.assertEqual(change['author'], 'Fred Flinstone <fred@flinstone.org>')
        self.assertEqual(change['committer'], 'Freddy Flinstone <freddy@flinstone.org>')
        self.assertEqual(change['revision'], '41a212ee83ca127e3c8cf465891ab7216a705f59')
        self.assertEqual(change['comments'], 'okay i give in')
        self.assertEqual(change['branch'], 'master')
        self.assertEqual(change['revlink'], 'http://github.com/defunkt/github/commit/41a212ee83ca127e3c8cf465891ab7216a705f59')
        change = self.changeHook.master.data.updates.changesAdded[1]
        self.assertEqual(change['files'], ['modfile', 'removedFile'])
        self.assertEqual(change['repository'], 'http://github.com/defunkt/github')
        self.assertEqual(change['when_timestamp'], 1203114994)
        self.assertEqual(change['author'], 'Fred Flinstone <fred@flinstone.org>')
        self.assertEqual(change['committer'], 'Freddy Flinstone <freddy@flinstone.org>')
        self.assertEqual(change['src'], 'git')
        self.assertEqual(change['revision'], 'de8251ff97ee194a289832576287d6f8ad74e3d0')
        self.assertEqual(change['comments'], 'update pricing a tad')
        self.assertEqual(change['branch'], 'master')
        self.assertEqual(change['revlink'], 'http://github.com/defunkt/github/commit/de8251ff97ee194a289832576287d6f8ad74e3d0')
        self.assertEqual(change['properties']['event'], 'push')

    def test_git_with_change_encoded(self):
        if False:
            print('Hello World!')
        self._check_git_with_change([gitJsonPayload])

    def test_git_with_change_json(self):
        if False:
            print('Hello World!')
        self._check_git_with_change(gitJsonPayload)

    @defer.inlineCallbacks
    def testGitWithDistinctFalse(self):
        if False:
            while True:
                i = 10
        self.request = _prepare_request(b'push', [gitJsonPayload.replace(b'"distinct": true,', b'"distinct": false,')])
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['files'], ['filepath.rb'])
        self.assertEqual(change['repository'], 'http://github.com/defunkt/github')
        self.assertEqual(change['when_timestamp'], 1203116237)
        self.assertEqual(change['author'], 'Fred Flinstone <fred@flinstone.org>')
        self.assertEqual(change['committer'], 'Freddy Flinstone <freddy@flinstone.org>')
        self.assertEqual(change['revision'], '41a212ee83ca127e3c8cf465891ab7216a705f59')
        self.assertEqual(change['comments'], 'okay i give in')
        self.assertEqual(change['branch'], 'master')
        self.assertEqual(change['revlink'], 'http://github.com/defunkt/github/commit/41a212ee83ca127e3c8cf465891ab7216a705f59')
        self.assertEqual(change['properties']['github_distinct'], False)
        change = self.changeHook.master.data.updates.changesAdded[1]
        self.assertEqual(change['files'], ['modfile', 'removedFile'])
        self.assertEqual(change['repository'], 'http://github.com/defunkt/github')
        self.assertEqual(change['when_timestamp'], 1203114994)
        self.assertEqual(change['author'], 'Fred Flinstone <fred@flinstone.org>')
        self.assertEqual(change['committer'], 'Freddy Flinstone <freddy@flinstone.org>')
        self.assertEqual(change['src'], 'git')
        self.assertEqual(change['revision'], 'de8251ff97ee194a289832576287d6f8ad74e3d0')
        self.assertEqual(change['comments'], 'update pricing a tad')
        self.assertEqual(change['branch'], 'master')
        self.assertEqual(change['revlink'], 'http://github.com/defunkt/github/commit/de8251ff97ee194a289832576287d6f8ad74e3d0')

    @defer.inlineCallbacks
    def testGitWithNoJson(self):
        if False:
            print('Hello World!')
        self.request = _prepare_request(b'push', b'')
        yield self.request.test_render(self.changeHook)
        expected = b'Expecting value: line 1 column 1 (char 0)'
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)
        self.request.setResponseCode.assert_called_with(400, expected)

    @defer.inlineCallbacks
    def _check_git_with_no_changes(self, payload):
        if False:
            i = 10
            return i + 15
        self.request = _prepare_request(b'push', payload)
        yield self.request.test_render(self.changeHook)
        expected = b'no change found'
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)

    def test_git_with_no_changes_encoded(self):
        if False:
            return 10
        self._check_git_with_no_changes([gitJsonPayloadEmpty])

    def test_git_with_no_changes_json(self):
        if False:
            i = 10
            return i + 15
        self._check_git_with_no_changes(gitJsonPayloadEmpty)

    @defer.inlineCallbacks
    def _check_git_with_non_branch_changes(self, payload):
        if False:
            i = 10
            return i + 15
        self.request = _prepare_request(b'push', payload)
        yield self.request.test_render(self.changeHook)
        expected = b'no change found'
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)

    def test_git_with_non_branch_changes_encoded(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_git_with_non_branch_changes([gitJsonPayloadNonBranch])

    def test_git_with_non_branch_changes_json(self):
        if False:
            return 10
        self._check_git_with_non_branch_changes(gitJsonPayloadNonBranch)

    @defer.inlineCallbacks
    def _check_git_with_pull(self, payload):
        if False:
            return 10
        self.request = _prepare_request('pull_request', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['repository'], 'https://github.com/defunkt/github')
        self.assertEqual(change['when_timestamp'], 1412899790)
        self.assertEqual(change['author'], 'defunkt')
        self.assertEqual(change['revision'], '05c588ba8cd510ecbe112d020f215facb17817a7')
        self.assertEqual(change['comments'], 'GitHub Pull Request #50 (1 commit)\nUpdate the README with new information\nThis is a pretty simple change that we need to pull into master.')
        self.assertEqual(change['branch'], 'refs/pull/50/merge')
        self.assertEqual(change['files'], [])
        self.assertEqual(change['revlink'], 'https://github.com/defunkt/github/pull/50')
        self.assertEqual(change['properties']['basename'], 'master')
        self.assertDictSubset(gitPRproperties, change['properties'])

    def test_git_with_pull_encoded(self):
        if False:
            return 10
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._check_git_with_pull([gitJsonPayloadPullRequest])

    def test_git_with_pull_json(self):
        if False:
            print('Hello World!')
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._check_git_with_pull(gitJsonPayloadPullRequest)

    @defer.inlineCallbacks
    def _check_git_push_with_skip_message(self, payload):
        if False:
            while True:
                i = 10
        self.request = _prepare_request(b'push', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)

    def test_git_push_with_skip_message(self):
        if False:
            for i in range(10):
                print('nop')
        gitJsonPayloadCiSkips = [unicode2bytes(gitJsonPayloadCiSkipTemplate % {'skip': '[ci skip]'}), unicode2bytes(gitJsonPayloadCiSkipTemplate % {'skip': '[skip ci]'}), unicode2bytes(gitJsonPayloadCiSkipTemplate % {'skip': '[  ci skip   ]'})]
        for payload in gitJsonPayloadCiSkips:
            self._check_git_push_with_skip_message(payload)

class TestChangeHookConfiguredWithGitChangeCustomPullrequestRef(unittest.TestCase, TestReactorMixin):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        self.changeHook = _prepare_github_change_hook(self, strict=False, github_property_whitelist=['github.*'], pullrequest_ref='head')
        self.master = self.changeHook.master
        fake_headers = {'User-Agent': 'Buildbot'}
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'https://api.github.com', headers=fake_headers, debug=False, verify=False))
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            return 10
        yield self.master.stopService()

    @defer.inlineCallbacks
    def test_git_pull_request_with_custom_ref(self):
        if False:
            i = 10
            return i + 15
        commit = deepcopy([gitJsonPayloadPullRequest])
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self.request = _prepare_request('pull_request', commit)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['branch'], 'refs/pull/50/head')

class TestChangeHookConfiguredWithGitChangeCustomPullrequestRefWithAuth(unittest.TestCase, TestReactorMixin):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        _token = '7e076f41-b73a-4045-a817'
        self.changeHook = _prepare_github_change_hook(self, strict=False, github_property_whitelist=['github.*'], pullrequest_ref='head', token=_token)
        self.master = self.changeHook.master
        fake_headers = {'User-Agent': 'Buildbot', 'Authorization': 'token ' + _token}
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'https://api.github.com', headers=fake_headers, debug=False, verify=False))
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            while True:
                i = 10
        yield self.master.stopService()

    @defer.inlineCallbacks
    def test_git_pull_request_with_custom_ref(self):
        if False:
            while True:
                i = 10
        commit = deepcopy([gitJsonPayloadPullRequest])
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadCommit)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadFiles)
        self.request = _prepare_request('pull_request', commit)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['branch'], 'refs/pull/50/head')

class TestChangeHookRefWithAuth(unittest.TestCase, TestReactorMixin):
    secret_name = 'secretkey'
    secret_value = 'githubtoken'

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.changeHook = _prepare_github_change_hook(self, strict=False, github_property_whitelist=['github.*'], token=util.Secret(self.secret_name))
        self.master = self.changeHook.master
        fake_headers = {'User-Agent': 'Buildbot', 'Authorization': 'token ' + self.secret_value}
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'https://api.github.com', headers=fake_headers, debug=False, verify=False))
        fake_storage = FakeSecretStorage()
        secret_service = SecretManager()
        secret_service.services = [fake_storage]
        yield secret_service.setServiceParent(self.master)
        yield self.master.startService()
        fake_storage.reconfigService(secretdict={self.secret_name: self.secret_value})

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            i = 10
            return i + 15
        yield self.master.stopService()

    @defer.inlineCallbacks
    def test_git_pull_request(self):
        if False:
            print('Hello World!')
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadCommit)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadFiles)
        self.request = _prepare_request('pull_request', gitJsonPayloadPullRequest)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['branch'], 'refs/pull/50/merge')

class TestChangeHookConfiguredWithAuthAndCustomSkips(unittest.TestCase, TestReactorMixin):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        _token = '7e076f41-b73a-4045-a817'
        self.changeHook = _prepare_github_change_hook(self, strict=False, skips=['\\[ *bb *skip *\\]'], token=_token)
        self.master = self.changeHook.master
        fake_headers = {'User-Agent': 'Buildbot', 'Authorization': 'token ' + _token}
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'https://api.github.com', headers=fake_headers, debug=False, verify=False))
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            while True:
                i = 10
        yield self.master.stopService()

    @defer.inlineCallbacks
    def _check_push_with_skip_message(self, payload):
        if False:
            return 10
        self.request = _prepare_request(b'push', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)

    def test_push_with_skip_message(self):
        if False:
            print('Hello World!')
        gitJsonPayloadCiSkips = [unicode2bytes(gitJsonPayloadCiSkipTemplate % {'skip': '[bb skip]'}), unicode2bytes(gitJsonPayloadCiSkipTemplate % {'skip': '[  bb skip   ]'})]
        for payload in gitJsonPayloadCiSkips:
            self._check_push_with_skip_message(payload)

    @defer.inlineCallbacks
    def _check_push_no_ci_skip(self, payload):
        if False:
            for i in range(10):
                print('nop')
        self.request = _prepare_request(b'push', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)

    def test_push_no_ci_skip(self):
        if False:
            for i in range(10):
                print('nop')
        payload = gitJsonPayloadCiSkipTemplate % {'skip': '[ci skip]'}
        payload = unicode2bytes(payload)
        self._check_push_no_ci_skip(payload)

    @defer.inlineCallbacks
    def _check_pull_request_with_skip_message(self, payload):
        if False:
            while True:
                i = 10
        self.request = _prepare_request(b'pull_request', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)

    def test_pull_request_with_skip_message(self):
        if False:
            print('Hello World!')
        api_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        commit = deepcopy(gitJsonPayloadCommit)
        msgs = ('black magic [bb skip]', 'black magic [  bb skip   ]')
        for msg in msgs:
            commit['commit']['message'] = msg
            self._http.expect('get', api_endpoint, content_json=commit)
            self._check_pull_request_with_skip_message(gitJsonPayloadPullRequest)

    @defer.inlineCallbacks
    def _check_pull_request_no_skip(self, payload):
        if False:
            for i in range(10):
                print('nop')
        self.request = _prepare_request(b'pull_request', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)

    def test_pull_request_no_skip(self):
        if False:
            while True:
                i = 10
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadCommit)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadFiles)
        commit = deepcopy(gitJsonPayloadCommit)
        commit['commit']['message'] = 'black magic [skip bb]'
        self._check_pull_request_no_skip(gitJsonPayloadPullRequest)

class TestChangeHookConfiguredWithAuth(unittest.TestCase, TestReactorMixin):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        _token = '7e076f41-b73a-4045-a817'
        self.changeHook = _prepare_github_change_hook(self, strict=False, token=_token, github_property_whitelist=['github.*'])
        self.master = self.changeHook.master
        fake_headers = {'User-Agent': 'Buildbot', 'Authorization': 'token ' + _token}
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'https://api.github.com', headers=fake_headers, debug=False, verify=False))
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            print('Hello World!')
        yield self.master.stopService()

    def assertDictSubset(self, expected_dict, response_dict):
        if False:
            i = 10
            return i + 15
        expected = {}
        for key in expected_dict.keys():
            self.assertIn(key, set(response_dict.keys()))
            expected[key] = response_dict[key]
        self.assertDictEqual(expected_dict, expected)

    @defer.inlineCallbacks
    def _check_pull_request(self, payload):
        if False:
            i = 10
            return i + 15
        self.request = _prepare_request(b'pull_request', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)

    def test_pull_request(self):
        if False:
            print('Hello World!')
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadCommit)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadFiles)
        self._check_pull_request(gitJsonPayloadPullRequest)

    @defer.inlineCallbacks
    def _check_git_with_pull(self, payload, valid_token=True):
        if False:
            i = 10
            return i + 15
        self.request = _prepare_request('pull_request', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['repository'], 'https://github.com/defunkt/github')
        self.assertEqual(change['when_timestamp'], 1412899790)
        self.assertEqual(change['author'], 'defunkt')
        self.assertEqual(change['revision'], '05c588ba8cd510ecbe112d020f215facb17817a7')
        self.assertEqual(change['comments'], 'GitHub Pull Request #50 (1 commit)\nUpdate the README with new information\nThis is a pretty simple change that we need to pull into master.')
        self.assertEqual(change['branch'], 'refs/pull/50/merge')
        if valid_token:
            self.assertEqual(change['files'], ['README.md', 'old_README.md'])
        else:
            self.assertEqual(change['files'], [])
        self.assertEqual(change['revlink'], 'https://github.com/defunkt/github/pull/50')
        self.assertEqual(change['properties']['basename'], 'master')
        self.assertDictSubset(gitPRproperties, change['properties'])

    def test_git_with_pull_encoded(self):
        if False:
            i = 10
            return i + 15
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadCommit)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadFiles)
        self._check_git_with_pull([gitJsonPayloadPullRequest])

    def test_git_with_pull_json(self):
        if False:
            i = 10
            return i + 15
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadCommit)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadFiles)
        self._check_git_with_pull(gitJsonPayloadPullRequest)

    def test_git_with_pull_encoded_and_bad_token(self):
        if False:
            for i in range(10):
                print('nop')
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._check_git_with_pull([gitJsonPayloadPullRequest], valid_token=False)

    def test_git_with_pull_json_and_bad_token(self):
        if False:
            return 10
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._check_git_with_pull(gitJsonPayloadPullRequest, valid_token=False)

    @defer.inlineCallbacks
    def _check_git_pull_request_with_skip_message(self, payload):
        if False:
            i = 10
            return i + 15
        self.request = _prepare_request(b'pull_request', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)

    def test_git_pull_request_with_skip_message(self):
        if False:
            print('Hello World!')
        api_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        commit = deepcopy(gitJsonPayloadCommit)
        msgs = ('black magic [ci skip]', 'black magic [skip ci]', 'black magic [  ci skip   ]')
        for msg in msgs:
            commit['commit']['message'] = msg
            self._http.expect('get', api_endpoint, content_json=commit)
            self._check_git_pull_request_with_skip_message(gitJsonPayloadPullRequest)

class TestChangeHookConfiguredWithCustomApiRoot(unittest.TestCase, TestReactorMixin):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()
        self.changeHook = _prepare_github_change_hook(self, strict=False, github_api_endpoint='https://black.magic.io')
        self.master = self.changeHook.master
        fake_headers = {'User-Agent': 'Buildbot'}
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'https://black.magic.io', headers=fake_headers, debug=False, verify=False))
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            while True:
                i = 10
        yield self.master.stopService()

    @defer.inlineCallbacks
    def _check_pull_request(self, payload):
        if False:
            return 10
        self.request = _prepare_request(b'pull_request', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)

    def test_pull_request(self):
        if False:
            print('Hello World!')
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadNotFound, code=404)
        self._check_pull_request(gitJsonPayloadPullRequest)

class TestChangeHookConfiguredWithCustomApiRootWithAuth(unittest.TestCase, TestReactorMixin):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        _token = '7e076f41-b73a-4045-a817'
        self.changeHook = _prepare_github_change_hook(self, strict=False, github_api_endpoint='https://black.magic.io', token=_token)
        self.master = self.changeHook.master
        fake_headers = {'User-Agent': 'Buildbot', 'Authorization': 'token ' + _token}
        self._http = (yield fakehttpclientservice.HTTPClientService.getService(self.master, self, 'https://black.magic.io', headers=fake_headers, debug=False, verify=False))
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.master.stopService()

    @defer.inlineCallbacks
    def _check_pull_request(self, payload):
        if False:
            for i in range(10):
                print('nop')
        self.request = _prepare_request(b'pull_request', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)

    def test_pull_request(self):
        if False:
            i = 10
            return i + 15
        commit_endpoint = '/repos/defunkt/github/commits/05c588ba8cd510ecbe112d020f215facb17817a7'
        files_endpoint = '/repos/defunkt/github/pulls/50/files'
        self._http.expect('get', commit_endpoint, content_json=gitJsonPayloadCommit)
        self._http.expect('get', files_endpoint, content_json=gitJsonPayloadFiles)
        self._check_pull_request(gitJsonPayloadPullRequest)

class TestChangeHookConfiguredWithStrict(unittest.TestCase, TestReactorMixin):
    _SECRET = 'somethingreallysecret'

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        fakeStorageService = FakeSecretStorage()
        fakeStorageService.reconfigService(secretdict={'secret_key': self._SECRET})
        secretService = SecretManager()
        secretService.services = [fakeStorageService]
        self.changeHook = _prepare_github_change_hook(self, strict=True, secret=util.Secret('secret_key'))
        self.changeHook.master.addService(secretService)

    @defer.inlineCallbacks
    def test_signature_ok(self):
        if False:
            print('Hello World!')
        self.request = _prepare_request(b'push', gitJsonPayload, _secret=self._SECRET)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['files'], ['filepath.rb'])
        self.assertEqual(change['repository'], 'http://github.com/defunkt/github')
        self.assertEqual(change['when_timestamp'], 1203116237)
        self.assertEqual(change['author'], 'Fred Flinstone <fred@flinstone.org>')
        self.assertEqual(change['committer'], 'Freddy Flinstone <freddy@flinstone.org>')
        self.assertEqual(change['revision'], '41a212ee83ca127e3c8cf465891ab7216a705f59')
        self.assertEqual(change['comments'], 'okay i give in')
        self.assertEqual(change['branch'], 'master')
        self.assertEqual(change['revlink'], 'http://github.com/defunkt/github/commit/41a212ee83ca127e3c8cf465891ab7216a705f59')
        change = self.changeHook.master.data.updates.changesAdded[1]
        self.assertEqual(change['files'], ['modfile', 'removedFile'])
        self.assertEqual(change['repository'], 'http://github.com/defunkt/github')
        self.assertEqual(change['when_timestamp'], 1203114994)
        self.assertEqual(change['author'], 'Fred Flinstone <fred@flinstone.org>')
        self.assertEqual(change['committer'], 'Freddy Flinstone <freddy@flinstone.org>')
        self.assertEqual(change['src'], 'git')
        self.assertEqual(change['revision'], 'de8251ff97ee194a289832576287d6f8ad74e3d0')
        self.assertEqual(change['comments'], 'update pricing a tad')
        self.assertEqual(change['branch'], 'master')
        self.assertEqual(change['revlink'], 'http://github.com/defunkt/github/commit/de8251ff97ee194a289832576287d6f8ad74e3d0')

    @defer.inlineCallbacks
    def test_unknown_hash(self):
        if False:
            while True:
                i = 10
        bad_hash_type = b'blah'
        self.request = _prepare_request(b'push', gitJsonPayload, headers={_HEADER_SIGNATURE: bad_hash_type + b'=doesnotmatter'})
        yield self.request.test_render(self.changeHook)
        expected = b'Unknown hash type: ' + bad_hash_type
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)

    @defer.inlineCallbacks
    def test_signature_nok(self):
        if False:
            i = 10
            return i + 15
        bad_signature = b'sha1=wrongstuff'
        self.request = _prepare_request(b'push', gitJsonPayload, headers={_HEADER_SIGNATURE: bad_signature})
        yield self.request.test_render(self.changeHook)
        expected = b'Hash mismatch'
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)

    @defer.inlineCallbacks
    def test_missing_secret(self):
        if False:
            for i in range(10):
                print('nop')
        self.changeHook = _prepare_github_change_hook(self, strict=True)
        self.request = _prepare_request(b'push', gitJsonPayload)
        yield self.request.test_render(self.changeHook)
        expected = b'Strict mode is requested while no secret is provided'
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)

    @defer.inlineCallbacks
    def test_wrong_signature_format(self):
        if False:
            while True:
                i = 10
        bad_signature = b'hash=value=something'
        self.request = _prepare_request(b'push', gitJsonPayload, headers={_HEADER_SIGNATURE: bad_signature})
        yield self.request.test_render(self.changeHook)
        expected = b'Wrong signature format: ' + bad_signature
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)

    @defer.inlineCallbacks
    def test_signature_missing(self):
        if False:
            return 10
        self.request = _prepare_request(b'push', gitJsonPayload)
        yield self.request.test_render(self.changeHook)
        expected = b'Request has no required signature'
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertEqual(self.request.written, expected)

class TestChangeHookConfiguredWithCodebaseValue(unittest.TestCase, TestReactorMixin):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.changeHook = _prepare_github_change_hook(self, codebase='foobar')

    @defer.inlineCallbacks
    def _check_git_with_change(self, payload):
        if False:
            i = 10
            return i + 15
        self.request = _prepare_request(b'push', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['codebase'], 'foobar')

    def test_git_with_change_encoded(self):
        if False:
            return 10
        return self._check_git_with_change([gitJsonPayload])

    def test_git_with_change_json(self):
        if False:
            print('Hello World!')
        return self._check_git_with_change(gitJsonPayload)

def _codebase_function(payload):
    if False:
        print('Hello World!')
    return 'foobar-' + payload['repository']['name']

class TestChangeHookConfiguredWithCodebaseFunction(unittest.TestCase, TestReactorMixin):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.changeHook = _prepare_github_change_hook(self, codebase=_codebase_function)

    @defer.inlineCallbacks
    def _check_git_with_change(self, payload):
        if False:
            i = 10
            return i + 15
        self.request = _prepare_request(b'push', payload)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['codebase'], 'foobar-github')

    def test_git_with_change_encoded(self):
        if False:
            for i in range(10):
                print('nop')
        return self._check_git_with_change([gitJsonPayload])

    def test_git_with_change_json(self):
        if False:
            for i in range(10):
                print('nop')
        return self._check_git_with_change(gitJsonPayload)

class TestChangeHookConfiguredWithCustomEventHandler(unittest.TestCase, TestReactorMixin):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()

        class CustomGitHubEventHandler(GitHubEventHandler):

            def handle_ping(self, _, __):
                if False:
                    print('Hello World!')
                self.master.hook_called = True
                return ([], None)
        self.changeHook = _prepare_github_change_hook(self, **{'class': CustomGitHubEventHandler})

    @defer.inlineCallbacks
    def test_ping(self):
        if False:
            return 10
        self.request = _prepare_request(b'ping', b'{}')
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertTrue(self.changeHook.master.hook_called)