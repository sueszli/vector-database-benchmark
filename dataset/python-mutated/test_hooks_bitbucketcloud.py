from io import BytesIO
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.test.fake.web import FakeRequest
from buildbot.test.fake.web import fakeMasterForHooks
from buildbot.test.reactor import TestReactorMixin
from buildbot.util import unicode2bytes
from buildbot.www import change_hook
from buildbot.www.hooks.bitbucketcloud import _HEADER_EVENT
_CT_JSON = b'application/json'
bitbucketPRproperties = {'pullrequesturl': 'http://localhost:7990/projects/CI/repos/py-repo/pull-requests/21', 'bitbucket.id': '21', 'bitbucket.link': 'http://localhost:7990/projects/CI/repos/py-repo/pull-requests/21', 'bitbucket.title': 'dot 1496311906', 'bitbucket.authorLogin': 'Buildbot', 'bitbucket.fromRef.branch.name': 'branch_1496411680', 'bitbucket.fromRef.branch.rawNode': 'a87e21f7433d8c16ac7be7413483fbb76c72a8ba', 'bitbucket.fromRef.commit.authorTimestamp': 0, 'bitbucket.fromRef.commit.date': None, 'bitbucket.fromRef.commit.hash': 'a87e21f7433d8c16ac7be7413483fbb76c72a8ba', 'bitbucket.fromRef.commit.message': None, 'bitbucket.fromRef.repository.fullName': 'CI/py-repo', 'bitbucket.fromRef.repository.links.self.href': 'http://localhost:7990/projects/CI/repos/py-repo', 'bitbucket.fromRef.repository.owner.display_name': 'CI', 'bitbucket.fromRef.repository.owner.nickname': 'CI', 'bitbucket.fromRef.repository.ownerName': 'CI', 'bitbucket.fromRef.repository.project.key': 'CI', 'bitbucket.fromRef.repository.project.name': 'Continuous Integration', 'bitbucket.fromRef.repository.public': False, 'bitbucket.fromRef.repository.scm': 'git', 'bitbucket.fromRef.repository.slug': 'py-repo', 'bitbucket.toRef.branch.name': 'master', 'bitbucket.toRef.branch.rawNode': '7aebbb0089c40fce138a6d0b36d2281ea34f37f5', 'bitbucket.toRef.commit.authorTimestamp': 0, 'bitbucket.toRef.commit.date': None, 'bitbucket.toRef.commit.hash': '7aebbb0089c40fce138a6d0b36d2281ea34f37f5', 'bitbucket.toRef.commit.message': None, 'bitbucket.toRef.repository.fullName': 'CI/py-repo', 'bitbucket.toRef.repository.links.self.href': 'http://localhost:7990/projects/CI/repos/py-repo', 'bitbucket.toRef.repository.owner.display_name': 'CI', 'bitbucket.toRef.repository.owner.nickname': 'CI', 'bitbucket.toRef.repository.ownerName': 'CI', 'bitbucket.toRef.repository.project.key': 'CI', 'bitbucket.toRef.repository.project.name': 'Continuous Integration', 'bitbucket.toRef.repository.public': False, 'bitbucket.toRef.repository.scm': 'git', 'bitbucket.toRef.repository.slug': 'py-repo'}
pushJsonPayload = '\n{\n    "actor": {\n        "nickname": "John",\n        "display_name": "John Smith"\n    },\n    "repository": {\n        "scm": "git",\n        "project": {\n            "key": "CI",\n            "name": "Continuous Integration"\n        },\n        "slug": "py-repo",\n        "links": {\n            "self": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            },\n            "html": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            }\n        },\n        "public": false,\n        "ownerName": "CI",\n        "owner": {\n            "nickname": "CI",\n            "display_name": "CI"\n        },\n        "fullName": "CI/py-repo"\n    },\n    "push": {\n        "changes": [\n            {\n                "created": false,\n                "closed": false,\n                "new": {\n                    "type": "branch",\n                    "name": "branch_1496411680",\n                    "target": {\n                        "type": "commit",\n                        "hash": "793d4754230023d85532f9a38dba3290f959beb4"\n                    }\n                },\n                "old": {\n                    "type": "branch",\n                    "name": "branch_1496411680",\n                    "target": {\n                        "type": "commit",\n                        "hash": "a87e21f7433d8c16ac7be7413483fbb76c72a8ba"\n                    }\n                }\n            }\n        ]\n    }\n}\n'
pullRequestCreatedJsonPayload = '\n{\n    "actor": {\n        "nickname": "John",\n        "display_name": "John Smith"\n    },\n    "pullrequest": {\n        "id": "21",\n        "title": "dot 1496311906",\n        "link": "http://localhost:7990/projects/CI/repos/py-repo/pull-requests/21",\n        "authorLogin": "Buildbot",\n        "fromRef": {\n            "repository": {\n                "scm": "git",\n                "project": {\n                    "key": "CI",\n                    "name": "Continuous Integration"\n                },\n                "slug": "py-repo",\n                "links": {\n                    "self": {\n                        "href": "http://localhost:7990/projects/CI/repos/py-repo"\n                    }\n                },\n                "public": false,\n                "ownerName": "CI",\n                "owner": {\n                    "nickname": "CI",\n                    "display_name": "CI"\n                },\n                "fullName": "CI/py-repo"\n            },\n            "commit": {\n                "message": null,\n                "date": null,\n                "hash": "a87e21f7433d8c16ac7be7413483fbb76c72a8ba",\n                "authorTimestamp": 0\n            },\n            "branch": {\n                "rawNode": "a87e21f7433d8c16ac7be7413483fbb76c72a8ba",\n                "name": "branch_1496411680"\n            }\n        },\n        "toRef": {\n            "repository": {\n                "scm": "git",\n                "project": {\n                    "key": "CI",\n                    "name": "Continuous Integration"\n                },\n                "slug": "py-repo",\n                "links": {\n                    "self": {\n                        "href": "http://localhost:7990/projects/CI/repos/py-repo"\n                    }\n                },\n                "public": false,\n                "ownerName": "CI",\n                "owner": {\n                    "nickname": "CI",\n                    "display_name": "CI"\n                },\n                "fullName": "CI/py-repo"\n            },\n            "commit": {\n                "message": null,\n                "date": null,\n                "hash": "7aebbb0089c40fce138a6d0b36d2281ea34f37f5",\n                "authorTimestamp": 0\n            },\n            "branch": {\n                "rawNode": "7aebbb0089c40fce138a6d0b36d2281ea34f37f5",\n                "name": "master"\n            }\n        }\n    },\n    "repository": {\n        "scm": "git",\n        "project": {\n            "key": "CI",\n            "name": "Continuous Integration"\n        },\n        "slug": "py-repo",\n        "links": {\n            "self": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            }\n        },\n        "public": false,\n        "ownerName": "CI",\n        "owner": {\n            "nickname": "CI",\n            "display_name": "CI"\n        },\n        "fullName": "CI/py-repo"\n    }\n}\n'
pullRequestUpdatedJsonPayload = '\n{\n    "actor": {\n        "nickname": "John",\n        "display_name": "John Smith"\n    },\n    "pullrequest": {\n        "id": "21",\n        "title": "dot 1496311906",\n        "link": "http://localhost:7990/projects/CI/repos/py-repo/pull-requests/21",\n        "authorLogin": "Buildbot",\n        "fromRef": {\n            "repository": {\n                "scm": "git",\n                "project": {\n                    "key": "CI",\n                    "name": "Continuous Integration"\n                },\n                "slug": "py-repo",\n                "links": {\n                    "self": {\n                        "href": "http://localhost:7990/projects/CI/repos/py-repo"\n                    }\n                },\n                "public": false,\n                "ownerName": "CI",\n                "owner": {\n                    "nickname": "CI",\n                    "display_name": "CI"\n                },\n                "fullName": "CI/py-repo"\n            },\n            "commit": {\n                "message": null,\n                "date": null,\n                "hash": "a87e21f7433d8c16ac7be7413483fbb76c72a8ba",\n                "authorTimestamp": 0\n            },\n            "branch": {\n                "rawNode": "a87e21f7433d8c16ac7be7413483fbb76c72a8ba",\n                "name": "branch_1496411680"\n            }\n        },\n        "toRef": {\n            "repository": {\n                "scm": "git",\n                "project": {\n                    "key": "CI",\n                    "name": "Continuous Integration"\n                },\n                "slug": "py-repo",\n                "links": {\n                    "self": {\n                        "href": "http://localhost:7990/projects/CI/repos/py-repo"\n                    }\n                },\n                "public": false,\n                "ownerName": "CI",\n                "owner": {\n                    "nickname": "CI",\n                    "display_name": "CI"\n                },\n                "fullName": "CI/py-repo"\n            },\n            "commit": {\n                "message": null,\n                "date": null,\n                "hash": "7aebbb0089c40fce138a6d0b36d2281ea34f37f5",\n                "authorTimestamp": 0\n            },\n            "branch": {\n                "rawNode": "7aebbb0089c40fce138a6d0b36d2281ea34f37f5",\n                "name": "master"\n            }\n        }\n    },\n    "repository": {\n        "scm": "git",\n        "project": {\n            "key": "CI",\n            "name": "Continuous Integration"\n        },\n        "slug": "py-repo",\n        "links": {\n            "self": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            }\n        },\n        "public": false,\n        "ownerName": "CI",\n        "owner": {\n            "nickname": "CI",\n            "display_name": "CI"\n        },\n        "fullName": "CI/py-repo"\n    }\n}\n'
pullRequestRejectedJsonPayload = '\n{\n    "actor": {\n        "nickname": "John",\n        "display_name": "John Smith"\n    },\n    "pullrequest": {\n        "id": "21",\n        "title": "dot 1496311906",\n        "link": "http://localhost:7990/projects/CI/repos/py-repo/pull-requests/21",\n        "authorLogin": "Buildbot",\n        "fromRef": {\n            "repository": {\n                "scm": "git",\n                "project": {\n                    "key": "CI",\n                    "name": "Continuous Integration"\n                },\n                "slug": "py-repo",\n                "links": {\n                    "self": {\n                        "href": "http://localhost:7990/projects/CI/repos/py-repo"\n                    }\n                },\n                "public": false,\n                "ownerName": "CI",\n                "owner": {\n                    "nickname": "CI",\n                    "display_name": "CI"\n                },\n                "fullName": "CI/py-repo"\n            },\n            "commit": {\n                "message": null,\n                "date": null,\n                "hash": "a87e21f7433d8c16ac7be7413483fbb76c72a8ba",\n                "authorTimestamp": 0\n            },\n            "branch": {\n                "rawNode": "a87e21f7433d8c16ac7be7413483fbb76c72a8ba",\n                "name": "branch_1496411680"\n            }\n        },\n        "toRef": {\n            "repository": {\n                "scm": "git",\n                "project": {\n                    "key": "CI",\n                    "name": "Continuous Integration"\n                },\n                "slug": "py-repo",\n                "links": {\n                    "self": {\n                        "href": "http://localhost:7990/projects/CI/repos/py-repo"\n                    }\n                },\n                "public": false,\n                "ownerName": "CI",\n                "owner": {\n                    "nickname": "CI",\n                    "display_name": "CI"\n                },\n                "fullName": "CI/py-repo"\n            },\n            "commit": {\n                "message": null,\n                "date": null,\n                "hash": "7aebbb0089c40fce138a6d0b36d2281ea34f37f5",\n                "authorTimestamp": 0\n            },\n            "branch": {\n                "rawNode": "7aebbb0089c40fce138a6d0b36d2281ea34f37f5",\n                "name": "master"\n            }\n        }\n    },\n    "repository": {\n        "scm": "git",\n        "project": {\n            "key": "CI",\n            "name": "Continuous Integration"\n        },\n        "slug": "py-repo",\n        "links": {\n            "self": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            }\n        },\n        "public": false,\n        "ownerName": "CI",\n        "owner": {\n            "nickname": "CI",\n            "display_name": "CI"\n        },\n        "fullName": "CI/py-repo"\n    }\n}\n'
pullRequestFulfilledJsonPayload = '\n{\n    "actor": {\n        "nickname": "John",\n        "display_name": "John Smith"\n    },\n    "pullrequest": {\n        "id": "21",\n        "title": "dot 1496311906",\n        "link": "http://localhost:7990/projects/CI/repos/py-repo/pull-requests/21",\n        "authorLogin": "Buildbot",\n        "fromRef": {\n            "repository": {\n                "scm": "git",\n                "project": {\n                    "key": "CI",\n                    "name": "Continuous Integration"\n                },\n                "slug": "py-repo",\n                "links": {\n                    "self": {\n                        "href": "http://localhost:7990/projects/CI/repos/py-repo"\n                    }\n                },\n                "public": false,\n                "ownerName": "CI",\n                "owner": {\n                    "nickname": "CI",\n                    "display_name": "CI"\n                },\n                "fullName": "CI/py-repo"\n            },\n            "commit": {\n                "message": null,\n                "date": null,\n                "hash": "a87e21f7433d8c16ac7be7413483fbb76c72a8ba",\n                "authorTimestamp": 0\n            },\n            "branch": {\n                "rawNode": "a87e21f7433d8c16ac7be7413483fbb76c72a8ba",\n                "name": "branch_1496411680"\n            }\n        },\n        "toRef": {\n            "repository": {\n                "scm": "git",\n                "project": {\n                    "key": "CI",\n                    "name": "Continuous Integration"\n                },\n                "slug": "py-repo",\n                "links": {\n                    "self": {\n                        "href": "http://localhost:7990/projects/CI/repos/py-repo"\n                    }\n                },\n                "public": false,\n                "ownerName": "CI",\n                "owner": {\n                    "nickname": "CI",\n                    "display_name": "CI"\n                },\n                "fullName": "CI/py-repo"\n            },\n            "commit": {\n                "message": null,\n                "date": null,\n                "hash": "7aebbb0089c40fce138a6d0b36d2281ea34f37f5",\n                "authorTimestamp": 0\n            },\n            "branch": {\n                "rawNode": "7aebbb0089c40fce138a6d0b36d2281ea34f37f5",\n                "name": "master"\n            }\n        }\n    },\n    "repository": {\n        "scm": "git",\n        "project": {\n            "key": "CI",\n            "name": "Continuous Integration"\n        },\n        "slug": "py-repo",\n        "links": {\n            "self": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            }\n        },\n        "public": false,\n        "ownerName": "CI",\n        "owner": {\n            "nickname": "CI",\n            "display_name": "CI"\n        },\n        "fullName": "CI/py-repo"\n    }\n}\n'
deleteTagJsonPayload = '\n{\n    "actor": {\n        "nickname": "John",\n        "display_name": "John Smith"\n    },\n    "repository": {\n        "scm": "git",\n        "project": {\n            "key": "CI",\n            "name": "Continuous Integration"\n        },\n        "slug": "py-repo",\n        "links": {\n            "self": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            },\n            "html": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            }\n        },\n        "ownerName": "BUIL",\n        "public": false,\n        "owner": {\n            "nickname": "CI",\n            "display_name": "CI"\n        },\n        "fullName": "CI/py-repo"\n    },\n    "push": {\n        "changes": [\n            {\n                "created": false,\n                "closed": true,\n                "old": {\n                    "type": "tag",\n                    "name": "1.0.0",\n                    "target": {\n                        "type": "commit",\n                        "hash": "793d4754230023d85532f9a38dba3290f959beb4"\n                    }\n                },\n                "new": null\n            }\n        ]\n    }\n}\n'
deleteBranchJsonPayload = '\n{\n    "actor": {\n        "nickname": "John",\n        "display_name": "John Smith"\n    },\n    "repository": {\n        "scm": "git",\n        "project": {\n            "key": "CI",\n            "name": "Continuous Integration"\n        },\n        "slug": "py-repo",\n        "links": {\n            "self": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            },\n            "html": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            }\n        },\n        "ownerName": "CI",\n        "public": false,\n        "owner": {\n            "nickname": "CI",\n            "display_name": "CI"\n        },\n        "fullName": "CI/py-repo"\n    },\n    "push": {\n        "changes": [\n            {\n                "created": false,\n                "closed": true,\n                "old": {\n                    "type": "branch",\n                    "name": "branch_1496758965",\n                    "target": {\n                        "type": "commit",\n                        "hash": "793d4754230023d85532f9a38dba3290f959beb4"\n                    }\n                },\n                "new": null\n            }\n        ]\n    }\n}\n'
newTagJsonPayload = '\n{\n    "actor": {\n        "nickname": "John",\n        "display_name": "John Smith"\n    },\n    "repository": {\n        "scm": "git",\n        "project": {\n            "key": "CI",\n            "name": "Continuous Integration"\n        },\n        "slug": "py-repo",\n        "links": {\n            "self": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            },\n            "html": {\n                "href": "http://localhost:7990/projects/CI/repos/py-repo"\n            }\n        },\n        "public": false,\n        "ownerName": "CI",\n        "owner": {\n            "nickname": "CI",\n            "display_name": "CI"\n        },\n        "fullName": "CI/py-repo"\n    },\n    "push": {\n        "changes": [\n            {\n                "created": true,\n                "closed": false,\n                "old": null,\n                "new": {\n                    "type": "tag",\n                    "name": "1.0.0",\n                    "target": {\n                        "type": "commit",\n                        "hash": "793d4754230023d85532f9a38dba3290f959beb4"\n                    }\n                }\n            }\n        ]\n    }\n}\n'

def _prepare_request(payload, headers=None, change_dict=None):
    if False:
        print('Hello World!')
    headers = headers or {}
    request = FakeRequest(change_dict)
    request.uri = b'/change_hook/bitbucketcloud'
    request.method = b'POST'
    if isinstance(payload, str):
        payload = unicode2bytes(payload)
    request.content = BytesIO(payload)
    request.received_headers[b'Content-Type'] = _CT_JSON
    request.received_headers.update(headers)
    return request

class TestChangeHookConfiguredWithGitChange(unittest.TestCase, TestReactorMixin):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.change_hook = change_hook.ChangeHookResource(dialects={'bitbucketcloud': {'bitbucket_property_whitelist': ['bitbucket.*']}}, master=fakeMasterForHooks(self))

    def assertDictSubset(self, expected_dict, response_dict):
        if False:
            i = 10
            return i + 15
        expected = {}
        for key in expected_dict.keys():
            self.assertIn(key, set(response_dict.keys()))
            expected[key] = response_dict[key]
        self.assertDictEqual(expected_dict, expected)

    def _checkPush(self, change):
        if False:
            return 10
        self.assertEqual(change['repository'], 'http://localhost:7990/projects/CI/repos/py-repo')
        self.assertEqual(change['author'], 'John Smith <John>')
        self.assertEqual(change['project'], 'Continuous Integration')
        self.assertEqual(change['revision'], '793d4754230023d85532f9a38dba3290f959beb4')
        self.assertEqual(change['comments'], 'Bitbucket Cloud commit 793d4754230023d85532f9a38dba3290f959beb4')
        self.assertEqual(change['revlink'], 'http://localhost:7990/projects/CI/repos/py-repo/commits/793d4754230023d85532f9a38dba3290f959beb4')

    @defer.inlineCallbacks
    def testHookWithChangeOnPushEvent(self):
        if False:
            return 10
        request = _prepare_request(pushJsonPayload, headers={_HEADER_EVENT: 'repo:push'})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        change = self.change_hook.master.data.updates.changesAdded[0]
        self._checkPush(change)
        self.assertEqual(change['branch'], 'refs/heads/branch_1496411680')
        self.assertEqual(change['category'], 'push')

    @defer.inlineCallbacks
    def testHookWithNonDictOption(self):
        if False:
            while True:
                i = 10
        self.change_hook.dialects = {'bitbucketcloud': True}
        yield self.testHookWithChangeOnPushEvent()

    def _checkPullRequest(self, change):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(change['repository'], 'http://localhost:7990/projects/CI/repos/py-repo')
        self.assertEqual(change['author'], 'John Smith <John>')
        self.assertEqual(change['project'], 'Continuous Integration')
        self.assertEqual(change['comments'], 'Bitbucket Cloud Pull Request #21')
        self.assertEqual(change['revlink'], 'http://localhost:7990/projects/CI/repos/py-repo/pull-requests/21')
        self.assertEqual(change['revision'], 'a87e21f7433d8c16ac7be7413483fbb76c72a8ba')
        self.assertDictSubset(bitbucketPRproperties, change['properties'])

    @defer.inlineCallbacks
    def testHookWithChangeOnPullRequestCreated(self):
        if False:
            return 10
        request = _prepare_request(pullRequestCreatedJsonPayload, headers={_HEADER_EVENT: 'pullrequest:created'})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        change = self.change_hook.master.data.updates.changesAdded[0]
        self._checkPullRequest(change)
        self.assertEqual(change['branch'], 'refs/pull-requests/21/merge')
        self.assertEqual(change['category'], 'pull-created')

    @defer.inlineCallbacks
    def testHookWithChangeOnPullRequestUpdated(self):
        if False:
            while True:
                i = 10
        request = _prepare_request(pullRequestUpdatedJsonPayload, headers={_HEADER_EVENT: 'pullrequest:updated'})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        change = self.change_hook.master.data.updates.changesAdded[0]
        self._checkPullRequest(change)
        self.assertEqual(change['branch'], 'refs/pull-requests/21/merge')
        self.assertEqual(change['category'], 'pull-updated')

    @defer.inlineCallbacks
    def testHookWithChangeOnPullRequestRejected(self):
        if False:
            while True:
                i = 10
        request = _prepare_request(pullRequestRejectedJsonPayload, headers={_HEADER_EVENT: 'pullrequest:rejected'})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        change = self.change_hook.master.data.updates.changesAdded[0]
        self._checkPullRequest(change)
        self.assertEqual(change['branch'], 'refs/heads/branch_1496411680')
        self.assertEqual(change['category'], 'pull-rejected')

    @defer.inlineCallbacks
    def testHookWithChangeOnPullRequestFulfilled(self):
        if False:
            for i in range(10):
                print('nop')
        request = _prepare_request(pullRequestFulfilledJsonPayload, headers={_HEADER_EVENT: 'pullrequest:fulfilled'})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        change = self.change_hook.master.data.updates.changesAdded[0]
        self._checkPullRequest(change)
        self.assertEqual(change['branch'], 'refs/heads/master')
        self.assertEqual(change['category'], 'pull-fulfilled')

    @defer.inlineCallbacks
    def _checkCodebase(self, event_type, expected_codebase):
        if False:
            while True:
                i = 10
        payloads = {'repo:push': pushJsonPayload, 'pullrequest:updated': pullRequestUpdatedJsonPayload}
        request = _prepare_request(payloads[event_type], headers={_HEADER_EVENT: event_type})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        change = self.change_hook.master.data.updates.changesAdded[0]
        self.assertEqual(change['codebase'], expected_codebase)

    @defer.inlineCallbacks
    def testHookWithCodebaseValueOnPushEvent(self):
        if False:
            for i in range(10):
                print('nop')
        self.change_hook.dialects = {'bitbucketcloud': {'codebase': 'super-codebase'}}
        yield self._checkCodebase('repo:push', 'super-codebase')

    @defer.inlineCallbacks
    def testHookWithCodebaseFunctionOnPushEvent(self):
        if False:
            i = 10
            return i + 15
        self.change_hook.dialects = {'bitbucketcloud': {'codebase': lambda payload: payload['repository']['project']['key']}}
        yield self._checkCodebase('repo:push', 'CI')

    @defer.inlineCallbacks
    def testHookWithCodebaseValueOnPullEvent(self):
        if False:
            print('Hello World!')
        self.change_hook.dialects = {'bitbucketcloud': {'codebase': 'super-codebase'}}
        yield self._checkCodebase('pullrequest:updated', 'super-codebase')

    @defer.inlineCallbacks
    def testHookWithCodebaseFunctionOnPullEvent(self):
        if False:
            print('Hello World!')
        self.change_hook.dialects = {'bitbucketcloud': {'codebase': lambda payload: payload['repository']['project']['key']}}
        yield self._checkCodebase('pullrequest:updated', 'CI')

    @defer.inlineCallbacks
    def testHookWithUnhandledEvent(self):
        if False:
            return 10
        request = _prepare_request(pushJsonPayload, headers={_HEADER_EVENT: 'invented:event'})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 0)
        self.assertEqual(request.written, b'Unknown event: invented_event')

    @defer.inlineCallbacks
    def testHookWithChangeOnCreateTag(self):
        if False:
            for i in range(10):
                print('nop')
        request = _prepare_request(newTagJsonPayload, headers={_HEADER_EVENT: 'repo:push'})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        change = self.change_hook.master.data.updates.changesAdded[0]
        self._checkPush(change)
        self.assertEqual(change['branch'], 'refs/tags/1.0.0')
        self.assertEqual(change['category'], 'push')

    @defer.inlineCallbacks
    def testHookWithChangeOnDeleteTag(self):
        if False:
            print('Hello World!')
        request = _prepare_request(deleteTagJsonPayload, headers={_HEADER_EVENT: 'repo:push'})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        change = self.change_hook.master.data.updates.changesAdded[0]
        self._checkPush(change)
        self.assertEqual(change['branch'], 'refs/tags/1.0.0')
        self.assertEqual(change['category'], 'ref-deleted')

    @defer.inlineCallbacks
    def testHookWithChangeOnDeleteBranch(self):
        if False:
            return 10
        request = _prepare_request(deleteBranchJsonPayload, headers={_HEADER_EVENT: 'repo:push'})
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 1)
        change = self.change_hook.master.data.updates.changesAdded[0]
        self._checkPush(change)
        self.assertEqual(change['branch'], 'refs/heads/branch_1496758965')
        self.assertEqual(change['category'], 'ref-deleted')

    @defer.inlineCallbacks
    def testHookWithInvalidContentType(self):
        if False:
            i = 10
            return i + 15
        request = _prepare_request(pushJsonPayload, headers={_HEADER_EVENT: b'repo:push'})
        request.received_headers[b'Content-Type'] = b'invalid/content'
        yield request.test_render(self.change_hook)
        self.assertEqual(len(self.change_hook.master.data.updates.changesAdded), 0)
        self.assertEqual(request.written, b'Unknown content type: invalid/content')