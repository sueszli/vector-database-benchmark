from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.plugins import util
from buildbot.secrets.manager import SecretManager
from buildbot.test.fake.secrets import FakeSecretStorage
from buildbot.test.fake.web import FakeRequest
from buildbot.test.fake.web import fakeMasterForHooks
from buildbot.test.reactor import TestReactorMixin
from buildbot.www import change_hook
from buildbot.www.hooks.gitlab import _HEADER_EVENT
from buildbot.www.hooks.gitlab import _HEADER_GITLAB_TOKEN
gitJsonPayload = b'\n{\n  "before": "95790bf891e76fee5e1747ab589903a6a1f80f22",\n  "after": "da1560886d4f094c3e6c9ef40349f7d38b5d27d7",\n  "ref": "refs/heads/master",\n  "user_id": 4,\n  "user_name": "John Smith",\n  "repository": {\n    "name": "Diaspora",\n    "url": "git@localhost:diaspora.git",\n    "description": "",\n    "homepage": "http://localhost/diaspora"\n  },\n  "commits": [\n    {\n      "id": "b6568db1bc1dcd7f8b4d5a946b0b91f9dacd7327",\n      "message": "Update Catalan translation to e38cb41.",\n      "timestamp": "2011-12-12T14:27:31+02:00",\n      "url": "http://localhost/diaspora/commits/b6568db1bc1dcd7f8b4d5a946b0b91f9dacd7327",\n      "author": {\n        "name": "Jordi Mallach",\n        "email": "jordi@softcatala.org"\n      }\n    },\n    {\n      "id": "da1560886d4f094c3e6c9ef40349f7d38b5d27d7",\n      "message": "fixed readme",\n      "timestamp": "2012-01-03T23:36:29+02:00",\n      "url": "http://localhost/diaspora/commits/da1560886d4f094c3e6c9ef40349f7d38b5d27d7",\n      "author": {\n        "name": "GitLab dev user",\n        "email": "gitlabdev@dv6700.(none)"\n      }\n    }\n  ],\n  "total_commits_count": 2\n}\n'
gitJsonPayloadTag = b'\n{\n  "object_kind": "tag_push",\n  "before": "0000000000000000000000000000000000000000",\n  "after": "82b3d5ae55f7080f1e6022629cdb57bfae7cccc7",\n  "ref": "refs/tags/v1.0.0",\n  "checkout_sha": "82b3d5ae55f7080f1e6022629cdb57bfae7cccc7",\n  "user_id": 1,\n  "user_name": "John Smith",\n  "repository":{\n    "name": "Example",\n    "url": "git@localhost:diaspora.git",\n    "description": "",\n    "homepage": "http://example.com/jsmith/example",\n    "git_http_url":"http://example.com/jsmith/example.git",\n    "git_ssh_url":"git@example.com:jsmith/example.git",\n    "visibility_level":0\n  },\n   "commits": [\n     {\n       "id": "b6568db1bc1dcd7f8b4d5a946b0b91f9dacd7327",\n       "message": "Update Catalan translation to e38cb41.",\n       "timestamp": "2011-12-12T14:27:31+02:00",\n       "url": "http://localhost/diaspora/commits/b6568db1bc1dcd7f8b4d5a946b0b91f9dacd7327",\n       "author": {\n         "name": "Jordi Mallach",\n         "email": "jordi@softcatala.org"\n       }\n     },\n     {\n       "id": "da1560886d4f094c3e6c9ef40349f7d38b5d27d7",\n       "message": "fixed readme",\n       "timestamp": "2012-01-03T23:36:29+02:00",\n       "url": "http://localhost/diaspora/commits/da1560886d4f094c3e6c9ef40349f7d38b5d27d7",\n       "author": {\n         "name": "GitLab dev user",\n         "email": "gitlabdev@dv6700.(none)"\n       }\n     }\n   ],\n   "total_commits_count": 2\n}\n'
gitJsonPayloadMR_open = b'\n{\n   "event_type" : "merge_request",\n   "object_attributes" : {\n      "action" : "open",\n      "assignee_id" : null,\n      "author_id" : 15,\n      "created_at" : "2018-05-15 07:45:37 -0700",\n      "description" : "This to both gitlab gateways!",\n      "head_pipeline_id" : 29931,\n      "human_time_estimate" : null,\n      "human_total_time_spent" : null,\n      "id" : 10850,\n      "iid" : 6,\n      "last_commit" : {\n         "author" : {\n            "email" : "mmusterman@example.com",\n            "name" : "Max Mustermann"\n         },\n         "id" : "92268bc781b24f0a61b907da062950e9e5252a69",\n         "message" : "Remove the dummy line again",\n         "timestamp" : "2018-05-14T07:54:04-07:00",\n         "url" : "https://gitlab.example.com/mmusterman/awesome_project/commit/92268bc781b24f0a61b907da062950e9e5252a69"\n      },\n      "last_edited_at" : null,\n      "last_edited_by_id" : null,\n      "merge_commit_sha" : null,\n      "merge_error" : null,\n      "merge_params" : {\n         "force_remove_source_branch" : 0\n      },\n      "merge_status" : "unchecked",\n      "merge_user_id" : null,\n      "merge_when_pipeline_succeeds" : false,\n      "milestone_id" : null,\n      "source" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "source_branch" : "ms-viewport",\n      "source_project_id" : 239,\n      "state" : "opened",\n      "target" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "target_branch" : "master",\n      "target_project_id" : 239,\n      "time_estimate" : 0,\n      "title" : "Remove the dummy line again",\n      "total_time_spent" : 0,\n      "updated_at" : "2018-05-15 07:45:37 -0700",\n      "updated_by_id" : null,\n      "url" : "https://gitlab.example.com/mmusterman/awesome_project/merge_requests/6",\n      "work_in_progress" : false\n   },\n   "object_kind" : "merge_request",\n   "project" : {\n      "avatar_url" : null,\n      "ci_config_path" : null,\n      "default_branch" : "master",\n      "description" : "Trivial project for testing build machinery quickly",\n      "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n      "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "id" : 239,\n      "name" : "awesome_project",\n      "namespace" : "mmusterman",\n      "path_with_namespace" : "mmusterman/awesome_project",\n      "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "visibility_level" : 0,\n      "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n   },\n   "user" : {\n      "avatar_url" : "http://www.gravatar.com/avatar/e64c7d89f26bd1972efa854d13d7dd61?s=40&d=identicon",\n      "name" : "Max Mustermann",\n      "username" : "mmusterman"\n   }\n}\n'
gitJsonPayloadMR_editdesc = b'\n{\n   "event_type" : "merge_request",\n   "object_attributes" : {\n      "action" : "update",\n      "assignee_id" : null,\n      "author_id" : 15,\n      "created_at" : "2018-05-15 07:45:37 -0700",\n      "description" : "Edited description.",\n      "head_pipeline_id" : 29931,\n      "human_time_estimate" : null,\n      "human_total_time_spent" : null,\n      "id" : 10850,\n      "iid" : 6,\n      "last_commit" : {\n         "author" : {\n            "email" : "mmusterman@example.com",\n            "name" : "Max Mustermann"\n         },\n         "id" : "92268bc781b24f0a61b907da062950e9e5252a69",\n         "message" : "Remove the dummy line again",\n         "timestamp" : "2018-05-14T07:54:04-07:00",\n         "url" : "https://gitlab.example.com/mmusterman/awesome_project/commit/92268bc781b24f0a61b907da062950e9e5252a69"\n      },\n      "last_edited_at" : "2018-05-15 07:49:55 -0700",\n      "last_edited_by_id" : 15,\n      "merge_commit_sha" : null,\n      "merge_error" : null,\n      "merge_params" : {\n         "force_remove_source_branch" : 0\n      },\n      "merge_status" : "can_be_merged",\n      "merge_user_id" : null,\n      "merge_when_pipeline_succeeds" : false,\n      "milestone_id" : null,\n      "source" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "source_branch" : "ms-viewport",\n      "source_project_id" : 239,\n      "state" : "opened",\n      "target" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "target_branch" : "master",\n      "target_project_id" : 239,\n      "time_estimate" : 0,\n      "title" : "Remove the dummy line again",\n      "total_time_spent" : 0,\n      "updated_at" : "2018-05-15 07:49:55 -0700",\n      "updated_by_id" : 15,\n      "url" : "https://gitlab.example.com/mmusterman/awesome_project/merge_requests/6",\n      "work_in_progress" : false\n   },\n   "object_kind" : "merge_request",\n   "project" : {\n      "avatar_url" : null,\n      "ci_config_path" : null,\n      "default_branch" : "master",\n      "description" : "Trivial project for testing build machinery quickly",\n      "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n      "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "id" : 239,\n      "name" : "awesome_project",\n      "namespace" : "mmusterman",\n      "path_with_namespace" : "mmusterman/awesome_project",\n      "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "visibility_level" : 0,\n      "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n   },\n   "user" : {\n      "avatar_url" : "http://www.gravatar.com/avatar/e64c7d89f26bd1972efa854d13d7dd61?s=40&d=identicon",\n      "name" : "Max Mustermann",\n      "username" : "mmusterman"\n   }\n}\n'
gitJsonPayloadMR_addcommit = b'\n{\n   "event_type" : "merge_request",\n   "object_attributes" : {\n      "action" : "update",\n      "assignee_id" : null,\n      "author_id" : 15,\n      "created_at" : "2018-05-15 07:45:37 -0700",\n      "description" : "Edited description.",\n      "head_pipeline_id" : 29931,\n      "human_time_estimate" : null,\n      "human_total_time_spent" : null,\n      "id" : 10850,\n      "iid" : 6,\n      "last_commit" : {\n         "author" : {\n            "email" : "mmusterman@example.com",\n            "name" : "Max Mustermann"\n         },\n         "id" : "cee8b01dcbaeed89563c2822f7c59a93c813eb6b",\n         "message" : "debian/compat: update to 9",\n         "timestamp" : "2018-05-15T07:51:11-07:00",\n         "url" : "https://gitlab.example.com/mmusterman/awesome_project/commit/cee8b01dcbaeed89563c2822f7c59a93c813eb6b"\n      },\n      "last_edited_at" : "2018-05-15 14:49:55 UTC",\n      "last_edited_by_id" : 15,\n      "merge_commit_sha" : null,\n      "merge_error" : null,\n      "merge_params" : {\n         "force_remove_source_branch" : 0\n      },\n      "merge_status" : "unchecked",\n      "merge_user_id" : null,\n      "merge_when_pipeline_succeeds" : false,\n      "milestone_id" : null,\n      "oldrev" : "92268bc781b24f0a61b907da062950e9e5252a69",\n      "source" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "source_branch" : "ms-viewport",\n      "source_project_id" : 239,\n      "state" : "opened",\n      "target" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "target_branch" : "master",\n      "target_project_id" : 239,\n      "time_estimate" : 0,\n      "title" : "Remove the dummy line again",\n      "total_time_spent" : 0,\n      "updated_at" : "2018-05-15 14:51:27 UTC",\n      "updated_by_id" : 15,\n      "url" : "https://gitlab.example.com/mmusterman/awesome_project/merge_requests/6",\n      "work_in_progress" : false\n   },\n   "object_kind" : "merge_request",\n   "project" : {\n      "avatar_url" : null,\n      "ci_config_path" : null,\n      "default_branch" : "master",\n      "description" : "Trivial project for testing build machinery quickly",\n      "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n      "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "id" : 239,\n      "name" : "awesome_project",\n      "namespace" : "mmusterman",\n      "path_with_namespace" : "mmusterman/awesome_project",\n      "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "visibility_level" : 0,\n      "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n   },\n   "user" : {\n      "avatar_url" : "http://www.gravatar.com/avatar/e64c7d89f26bd1972efa854d13d7dd61?s=40&d=identicon",\n      "name" : "Max Mustermann",\n      "username" : "mmusterman"\n   }\n}\n'
gitJsonPayloadMR_close = b'\n{\n   "event_type" : "merge_request",\n   "object_attributes" : {\n      "action" : "close",\n      "assignee_id" : null,\n      "author_id" : 15,\n      "created_at" : "2018-05-15 07:45:37 -0700",\n      "description" : "Edited description.",\n      "head_pipeline_id" : 29958,\n      "human_time_estimate" : null,\n      "human_total_time_spent" : null,\n      "id" : 10850,\n      "iid" : 6,\n      "last_commit" : {\n         "author" : {\n            "email" : "mmusterman@example.com",\n            "name" : "Max Mustermann"\n         },\n         "id" : "cee8b01dcbaeed89563c2822f7c59a93c813eb6b",\n         "message" : "debian/compat: update to 9",\n         "timestamp" : "2018-05-15T07:51:11-07:00",\n         "url" : "https://gitlab.example.com/mmusterman/awesome_project/commit/cee8b01dcbaeed89563c2822f7c59a93c813eb6b"\n      },\n      "last_edited_at" : "2018-05-15 07:49:55 -0700",\n      "last_edited_by_id" : 15,\n      "merge_commit_sha" : null,\n      "merge_error" : null,\n      "merge_params" : {\n         "force_remove_source_branch" : 0\n      },\n      "merge_status" : "can_be_merged",\n      "merge_user_id" : null,\n      "merge_when_pipeline_succeeds" : false,\n      "milestone_id" : null,\n      "source" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "source_branch" : "ms-viewport",\n      "source_project_id" : 239,\n      "state" : "closed",\n      "target" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "target_branch" : "master",\n      "target_project_id" : 239,\n      "time_estimate" : 0,\n      "title" : "Remove the dummy line again",\n      "total_time_spent" : 0,\n      "updated_at" : "2018-05-15 07:52:01 -0700",\n      "updated_by_id" : 15,\n      "url" : "https://gitlab.example.com/mmusterman/awesome_project/merge_requests/6",\n      "work_in_progress" : false\n   },\n   "object_kind" : "merge_request",\n   "project" : {\n      "avatar_url" : null,\n      "ci_config_path" : null,\n      "default_branch" : "master",\n      "description" : "Trivial project for testing build machinery quickly",\n      "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n      "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "id" : 239,\n      "name" : "awesome_project",\n      "namespace" : "mmusterman",\n      "path_with_namespace" : "mmusterman/awesome_project",\n      "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "visibility_level" : 0,\n      "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n   },\n   "user" : {\n      "avatar_url" : "http://www.gravatar.com/avatar/e64c7d89f26bd1972efa854d13d7dd61?s=40&d=identicon",\n      "name" : "Max Mustermann",\n      "username" : "mmusterman"\n   }\n}\n'
gitJsonPayloadMR_reopen = b'\n{\n   "event_type" : "merge_request",\n   "object_attributes" : {\n      "action" : "reopen",\n      "assignee_id" : null,\n      "author_id" : 15,\n      "created_at" : "2018-05-15 07:45:37 -0700",\n      "description" : "Edited description.",\n      "head_pipeline_id" : 29958,\n      "human_time_estimate" : null,\n      "human_total_time_spent" : null,\n      "id" : 10850,\n      "iid" : 6,\n      "last_commit" : {\n         "author" : {\n            "email" : "mmusterman@example.com",\n            "name" : "Max Mustermann"\n         },\n         "id" : "cee8b01dcbaeed89563c2822f7c59a93c813eb6b",\n         "message" : "debian/compat: update to 9",\n         "timestamp" : "2018-05-15T07:51:11-07:00",\n         "url" : "https://gitlab.example.com/mmusterman/awesome_project/commit/cee8b01dcbaeed89563c2822f7c59a93c813eb6b"\n      },\n      "last_edited_at" : "2018-05-15 07:49:55 -0700",\n      "last_edited_by_id" : 15,\n      "merge_commit_sha" : null,\n      "merge_error" : null,\n      "merge_params" : {\n         "force_remove_source_branch" : 0\n      },\n      "merge_status" : "can_be_merged",\n      "merge_user_id" : null,\n      "merge_when_pipeline_succeeds" : false,\n      "milestone_id" : null,\n      "source" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "source_branch" : "ms-viewport",\n      "source_project_id" : 239,\n      "state" : "opened",\n      "target" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "target_branch" : "master",\n      "target_project_id" : 239,\n      "time_estimate" : 0,\n      "title" : "Remove the dummy line again",\n      "total_time_spent" : 0,\n      "updated_at" : "2018-05-15 07:53:27 -0700",\n      "updated_by_id" : 15,\n      "url" : "https://gitlab.example.com/mmusterman/awesome_project/merge_requests/6",\n      "work_in_progress" : false\n   },\n   "object_kind" : "merge_request",\n   "project" : {\n      "avatar_url" : null,\n      "ci_config_path" : null,\n      "default_branch" : "master",\n      "description" : "Trivial project for testing build machinery quickly",\n      "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n      "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "id" : 239,\n      "name" : "awesome_project",\n      "namespace" : "mmusterman",\n      "path_with_namespace" : "mmusterman/awesome_project",\n      "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "visibility_level" : 0,\n      "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n   },\n   "user" : {\n      "avatar_url" : "http://www.gravatar.com/avatar/e64c7d89f26bd1972efa854d13d7dd61?s=40&d=identicon",\n      "name" : "Max Mustermann",\n      "username" : "mmusterman"\n   }\n}\n'
gitJsonPayloadMR_open_forked = b'\n{\n   "changes" : {\n      "total_time_spent" : {\n         "current" : 0,\n         "previous" : null\n      }\n   },\n   "event_type" : "merge_request",\n   "labels" : [],\n   "object_attributes" : {\n      "action" : "open",\n      "assignee_id" : null,\n      "author_id" : 15,\n      "created_at" : "2018-05-19 06:57:12 -0700",\n      "description" : "This is a merge request from a fork of the project.",\n      "head_pipeline_id" : null,\n      "human_time_estimate" : null,\n      "human_total_time_spent" : null,\n      "id" : 10914,\n      "iid" : 7,\n      "last_commit" : {\n         "author" : {\n            "email" : "mmusterman@example.com",\n            "name" : "Max Mustermann"\n         },\n         "id" : "e46ee239f3d6d41ade4d1e610669dd71ed86ec80",\n         "message" : "Add note to README",\n         "timestamp" : "2018-05-19T06:35:26-07:00",\n         "url" : "https://gitlab.example.com/mmusterman/awesome_project/commit/e46ee239f3d6d41ade4d1e610669dd71ed86ec80"\n      },\n      "last_edited_at" : null,\n      "last_edited_by_id" : null,\n      "merge_commit_sha" : null,\n      "merge_error" : null,\n      "merge_params" : {\n         "force_remove_source_branch" : "0"\n      },\n      "merge_status" : "unchecked",\n      "merge_user_id" : null,\n      "merge_when_pipeline_succeeds" : false,\n      "milestone_id" : null,\n      "source" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/build/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:build/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/build/awesome_project",\n         "http_url" : "https://gitlab.example.com/build/awesome_project.git",\n         "id" : 2337,\n         "name" : "awesome_project",\n         "namespace" : "build",\n         "path_with_namespace" : "build/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:build/awesome_project.git",\n         "url" : "git@gitlab.example.com:build/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/build/awesome_project"\n      },\n      "source_branch" : "ms-viewport",\n      "source_project_id" : 2337,\n      "state" : "opened",\n      "target" : {\n         "avatar_url" : null,\n         "ci_config_path" : null,\n         "default_branch" : "master",\n         "description" : "Trivial project for testing build machinery quickly",\n         "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n         "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n         "id" : 239,\n         "name" : "awesome_project",\n         "namespace" : "mmusterman",\n         "path_with_namespace" : "mmusterman/awesome_project",\n         "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n         "visibility_level" : 0,\n         "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n      },\n      "target_branch" : "master",\n      "target_project_id" : 239,\n      "time_estimate" : 0,\n      "title" : "Add note to README",\n      "total_time_spent" : 0,\n      "updated_at" : "2018-05-19 06:57:12 -0700",\n      "updated_by_id" : null,\n      "url" : "https://gitlab.example.com/mmusterman/awesome_project/merge_requests/7",\n      "work_in_progress" : false\n   },\n   "object_kind" : "merge_request",\n   "project" : {\n      "avatar_url" : null,\n      "ci_config_path" : null,\n      "default_branch" : "master",\n      "description" : "Trivial project for testing build machinery quickly",\n      "git_http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "git_ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n      "http_url" : "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "id" : 239,\n      "name" : "awesome_project",\n      "namespace" : "mmusterman",\n      "path_with_namespace" : "mmusterman/awesome_project",\n      "ssh_url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "url" : "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "visibility_level" : 0,\n      "web_url" : "https://gitlab.example.com/mmusterman/awesome_project"\n   },\n   "repository" : {\n      "description" : "Trivial project for testing build machinery quickly",\n      "homepage" : "https://gitlab.example.com/mmusterman/awesome_project",\n      "name" : "awesome_project",\n      "url" : "git@gitlab.example.com:mmusterman/awesome_project.git"\n   },\n   "user" : {\n      "avatar_url" : "http://www.gravatar.com/avatar/e64c7d89f26bd1972efa854d13d7dd61?s=40&d=identicon",\n      "name" : "Max Mustermann",\n      "username" : "mmusterman"\n   }\n}\n'
gitJsonPayloadMR_commented = b'\n{\n  "object_kind": "note",\n  "event_type": "note",\n  "user": {\n    "id": 343,\n    "name": "Name Surname",\n    "username": "rollo",\n    "avatar_url": "null",\n    "email": "[REDACTED]"\n  },\n  "project_id": 926,\n  "project": {\n    "id": 926,\n    "name": "awesome_project",\n    "description": "",\n    "web_url": "https://gitlab.example.com/mmusterman/awesome_project",\n    "avatar_url": null,\n    "git_ssh_url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n    "git_http_url": "https://gitlab.example.com/mmusterman/awesome_project.git",\n    "namespace" : "mmusterman",\n    "visibility_level": 0,\n    "path_with_namespace": "awesome_project",\n    "default_branch": "master",\n    "ci_config_path": null,\n    "homepage": "https://gitlab.example.com/mmusterman/awesome_project",\n    "url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n    "ssh_url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n    "http_url": "https://gitlab.example.com/mmusterman/awesome_project.git"\n  },\n  "object_attributes": {\n    "attachment": null,\n    "author_id": 343,\n    "change_position": {\n      "base_sha": null,\n      "start_sha": null,\n      "head_sha": null,\n      "old_path": null,\n      "new_path": null,\n      "position_type": "text",\n      "old_line": null,\n      "new_line": null,\n      "line_range": null\n    },\n    "commit_id": null,\n    "created_at": "2022-02-04 09:13:56 UTC",\n    "discussion_id": "0a307b85835ac7c3e2c1b4e6283d7baf42df0f8e",\n    "id": 83474,\n    "line_code": "762ab21851f67780cfb68832884fa1f859ccd00e_761_762",\n    "note": "036 #BB",\n    "noteable_id": 4085,\n    "noteable_type": "MergeRequest",\n    "original_position": {\n      "base_sha": "7e2c01527d87c36cf4f9e78dd9fc6aa4f602c365",\n      "start_sha": "7e2c01527d87c36cf4f9e78dd9fc6aa4f602c365",\n      "head_sha": "b91a85e84404932476f76ccbf0f42c963005501b",\n      "old_path": "run/exp.fun_R2B4",\n      "new_path": "run/exp.fun_R2B4",\n      "position_type": "text",\n      "old_line": null,\n      "new_line": 762,\n      "line_range": {\n        "start": {\n          "line_code": "762abg1851f67780cfb68832884fa1f859ccd00e_761_762",\n          "type": "new",\n          "old_line": null,\n          "new_line": 762\n        },\n        "end": {\n          "line_code": "762abg1851f67780cfb68832884fa1f859ccd00e_761_762",\n          "type": "new",\n          "old_line": null,\n          "new_line": 762\n        }\n      }\n    },\n    "position": {\n      "base_sha": "7e2c01527d87c36cf4f9e78dd9fc6aa4f602c365",\n      "start_sha": "7e2c01527d87c36cf4f9e78dd9fc6aa4f602c365",\n      "head_sha": "d1ce5517d3745dbd68e1eeb45f42380d76d0c490",\n      "old_path": "run/exp.esm_R2B4",\n      "new_path": "run/exp.esm_R2B4",\n      "position_type": "text",\n      "old_line": null,\n      "new_line": 762,\n      "line_range": {\n        "start": {\n          "line_code": "762ab21851f67780cfb68832884fa1f859ccd00e_761_762",\n          "type": "new",\n          "old_line": null,\n          "new_line": 762\n        },\n        "end": {\n          "line_code": "762ab21851f67780cfb68832884fa1f859ccd00e_761_762",\n          "type": "new",\n          "old_line": null,\n          "new_line": 762\n        }\n      }\n    },\n    "project_id": 926,\n    "resolved_at": null,\n    "resolved_by_id": null,\n    "resolved_by_push": null,\n    "st_diff": null,\n    "system": false,\n    "type": "DiffNote",\n    "updated_at": "2022-02-04 09:13:56 UTC",\n    "updated_by_id": null,\n    "description": "036 #BB",\n    "url": "https://gitlab.example.com/mmusterman/awesome_project_id/-/merge_requests/7#note_83474"\n  },\n  "repository": {\n    "name": "awesome_project",\n    "url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n    "description": "",\n    "homepage": "https://gitlab.example.com/mmusterman/awesome_project"\n  },\n  "merge_request": {\n    "assignee_id": 343,\n    "author_id": 343,\n    "created_at": "2022-01-28 09:17:41 UTC",\n    "description": "Some tests got disabled in the last merge. I will try to re-activate all infrastructure-related tests",\n    "head_pipeline_id": 14675,\n    "id": 4085,\n    "iid": 7,\n    "last_edited_at": "2022-02-01 15:10:38 UTC",\n    "last_edited_by_id": 343,\n    "merge_commit_sha": null,\n    "merge_error": null,\n    "merge_params": {\n      "force_remove_source_branch": "1"\n    },\n    "merge_status": "can_be_merged",\n    "merge_user_id": null,\n    "merge_when_pipeline_succeeds": false,\n    "milestone_id": null,\n    "source_branch": "fix-missing-tests",\n    "source_project_id": 926,\n    "state_id": 1,\n    "target_branch": "master",\n    "target_project_id": 926,\n    "time_estimate": 0,\n    "title": "Draft: Fix missing tests: pio",\n    "updated_at": "2022-02-04 09:13:17 UTC",\n    "updated_by_id": 343,\n    "url": "https://gitlab.example.com/mmusterman/awesome_project/-/merge_requests/7",\n    "source": {\n      "id": 926,\n      "name": "awesome_project",\n      "description": "",\n      "web_url": "https://gitlab.example.com/mmusterman/awesome_project",\n      "avatar_url": null,\n      "git_ssh_url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "git_http_url": "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "namespace": "mmusterman",\n      "visibility_level": 0,\n      "path_with_namespace": "awesome_project",\n      "default_branch": "master",\n      "ci_config_path": null,\n      "homepage": "https://gitlab.example.com/mmusterman/awesome_project",\n      "url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "ssh_url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "http_url": "https://gitlab.example.com/mmusterman/awesome_project.git"\n    },\n    "target": {\n      "id": 926,\n      "name": "awesome_project",\n      "description": "",\n      "web_url": "https://gitlab.example.com/mmusterman/awesome_project",\n      "avatar_url": null,\n      "git_ssh_url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "git_http_url": "https://gitlab.example.com/mmusterman/awesome_project.git",\n      "namespace": "mmusterman",\n      "visibility_level": 0,\n      "path_with_namespace": "awesome_project",\n      "default_branch": "master",\n      "ci_config_path": null,\n      "homepage": "https://gitlab.example.com/mmusterman/awesome_project",\n      "url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "ssh_url": "git@gitlab.example.com:mmusterman/awesome_project.git",\n      "http_url": "https://gitlab.example.com/mmusterman/awesome_project.git"\n    },\n    "last_commit": {\n      "id": "d1ce5517d3745dbd68e1eeb45f42380d76d0c490",\n      "message": "adopt radiation changes for ruby0 runs in bb",\n      "title": "adopt radiation changes for ruby0 runs in bb",\n      "timestamp": "2022-01-28T10:13:12+01:00",\n      "url": "https://gitlab.example.com/mmusterman/awesome_project/-/commit/d1ce5517d3745dbd68e1eeb45f42380d76d0c490",\n      "author": {\n        "name": "Name Surname",\n        "email": "surname@example.com"\n      }\n    },\n    "work_in_progress": true,\n    "total_time_spent": 0,\n    "time_change": 0,\n    "human_total_time_spent": null,\n    "human_time_change": null,\n    "human_time_estimate": null,\n    "assignee_ids": [\n      343\n    ],\n    "state": "opened"\n  }\n}\n'

def FakeRequestMR(content):
    if False:
        print('Hello World!')
    request = FakeRequest(content=content)
    request.uri = b'/change_hook/gitlab'
    request.args = {b'codebase': [b'MyCodebase']}
    request.received_headers[_HEADER_EVENT] = b'Merge Request Hook'
    request.method = b'POST'
    return request

class TestChangeHookConfiguredWithGitChange(unittest.TestCase, TestReactorMixin):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        self.changeHook = change_hook.ChangeHookResource(dialects={'gitlab': True}, master=fakeMasterForHooks(self))

    def check_changes_tag_event(self, r, project='', codebase=None):
        if False:
            return 10
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['repository'], 'git@localhost:diaspora.git')
        self.assertEqual(change['when_timestamp'], 1323692851)
        self.assertEqual(change['branch'], 'v1.0.0')

    def check_changes_mr_event(self, r, project='awesome_project', codebase=None, timestamp=1526309644, source_repo=None):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['repository'], 'https://gitlab.example.com/mmusterman/awesome_project.git')
        if source_repo is None:
            source_repo = 'https://gitlab.example.com/mmusterman/awesome_project.git'
        self.assertEqual(change['properties']['source_repository'], source_repo)
        self.assertEqual(change['properties']['target_repository'], 'https://gitlab.example.com/mmusterman/awesome_project.git')
        self.assertEqual(change['when_timestamp'], timestamp)
        self.assertEqual(change['branch'], 'master')
        self.assertEqual(change['properties']['source_branch'], 'ms-viewport')
        self.assertEqual(change['properties']['target_branch'], 'master')
        self.assertEqual(change['category'], 'merge_request')
        self.assertEqual(change.get('project'), project)

    def check_changes_mr_event_by_comment(self, r, project='awesome_project', codebase=None, timestamp=1526309644, source_repo=None, repo='https://gitlab.example.com/mmusterman/awesome_project.git', source_branch='ms-viewport', target_branch='master'):
        if False:
            while True:
                i = 10
        self.maxDiff = None
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 1)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['repository'], repo)
        if source_repo is None:
            source_repo = repo
        self.assertEqual(change['properties']['source_repository'], source_repo)
        self.assertEqual(change['properties']['target_repository'], repo)
        self.assertEqual(change['when_timestamp'], timestamp)
        self.assertEqual(change['branch'], target_branch)
        self.assertEqual(change['properties']['source_branch'], source_branch)
        self.assertEqual(change['properties']['target_branch'], target_branch)
        self.assertEqual(change['category'], 'note')
        self.assertEqual(change.get('project'), project)

    def check_changes_push_event(self, r, project='diaspora', codebase=None):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['repository'], 'git@localhost:diaspora.git')
        self.assertEqual(change['when_timestamp'], 1323692851)
        self.assertEqual(change['author'], 'Jordi Mallach <jordi@softcatala.org>')
        self.assertEqual(change['revision'], 'b6568db1bc1dcd7f8b4d5a946b0b91f9dacd7327')
        self.assertEqual(change['comments'], 'Update Catalan translation to e38cb41.')
        self.assertEqual(change['branch'], 'master')
        self.assertEqual(change['revlink'], 'http://localhost/diaspora/commits/b6568db1bc1dcd7f8b4d5a946b0b91f9dacd7327')
        change = self.changeHook.master.data.updates.changesAdded[1]
        self.assertEqual(change['repository'], 'git@localhost:diaspora.git')
        self.assertEqual(change['when_timestamp'], 1325626589)
        self.assertEqual(change['author'], 'GitLab dev user <gitlabdev@dv6700.(none)>')
        self.assertEqual(change['src'], 'git')
        self.assertEqual(change['revision'], 'da1560886d4f094c3e6c9ef40349f7d38b5d27d7')
        self.assertEqual(change['comments'], 'fixed readme')
        self.assertEqual(change['branch'], 'master')
        self.assertEqual(change['revlink'], 'http://localhost/diaspora/commits/da1560886d4f094c3e6c9ef40349f7d38b5d27d7')
        self.assertEqual(change.get('project').lower(), project.lower())
        self.assertEqual(change.get('codebase'), codebase)

    @defer.inlineCallbacks
    def testGitWithChange(self):
        if False:
            for i in range(10):
                print('nop')
        self.request = FakeRequest(content=gitJsonPayload)
        self.request.uri = b'/change_hook/gitlab'
        self.request.method = b'POST'
        self.request.received_headers[_HEADER_EVENT] = b'Push Hook'
        res = (yield self.request.test_render(self.changeHook))
        self.check_changes_push_event(res)

    @defer.inlineCallbacks
    def testGitWithChange_WithProjectToo(self):
        if False:
            print('Hello World!')
        self.request = FakeRequest(content=gitJsonPayload)
        self.request.uri = b'/change_hook/gitlab'
        self.request.args = {b'project': [b'Diaspora']}
        self.request.received_headers[_HEADER_EVENT] = b'Push Hook'
        self.request.method = b'POST'
        res = (yield self.request.test_render(self.changeHook))
        self.check_changes_push_event(res, project='Diaspora')

    @defer.inlineCallbacks
    def testGitWithChange_WithCodebaseToo(self):
        if False:
            i = 10
            return i + 15
        self.request = FakeRequest(content=gitJsonPayload)
        self.request.uri = b'/change_hook/gitlab'
        self.request.args = {b'codebase': [b'MyCodebase']}
        self.request.received_headers[_HEADER_EVENT] = b'Push Hook'
        self.request.method = b'POST'
        res = (yield self.request.test_render(self.changeHook))
        self.check_changes_push_event(res, codebase='MyCodebase')

    @defer.inlineCallbacks
    def testGitWithChange_WithPushTag(self):
        if False:
            for i in range(10):
                print('nop')
        self.request = FakeRequest(content=gitJsonPayloadTag)
        self.request.uri = b'/change_hook/gitlab'
        self.request.args = {b'codebase': [b'MyCodebase']}
        self.request.received_headers[_HEADER_EVENT] = b'Push Hook'
        self.request.method = b'POST'
        res = (yield self.request.test_render(self.changeHook))
        self.check_changes_tag_event(res, codebase='MyCodebase')

    @defer.inlineCallbacks
    def testGitWithNoJson(self):
        if False:
            i = 10
            return i + 15
        self.request = FakeRequest()
        self.request.uri = b'/change_hook/gitlab'
        self.request.method = b'POST'
        self.request.received_headers[_HEADER_EVENT] = b'Push Hook'
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)
        self.assertIn(b'Error loading JSON:', self.request.written)
        self.request.setResponseCode.assert_called_with(400, mock.ANY)

    @defer.inlineCallbacks
    def test_event_property(self):
        if False:
            i = 10
            return i + 15
        self.request = FakeRequest(content=gitJsonPayload)
        self.request.received_headers[_HEADER_EVENT] = b'Push Hook'
        self.request.uri = b'/change_hook/gitlab'
        self.request.method = b'POST'
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['properties']['event'], 'Push Hook')
        self.assertEqual(change['category'], 'Push Hook')

    @defer.inlineCallbacks
    def testGitWithChange_WithMR_open(self):
        if False:
            while True:
                i = 10
        self.request = FakeRequestMR(content=gitJsonPayloadMR_open)
        res = (yield self.request.test_render(self.changeHook))
        self.check_changes_mr_event(res, codebase='MyCodebase')
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['category'], 'merge_request')

    @defer.inlineCallbacks
    def testGitWithChange_WithMR_editdesc(self):
        if False:
            i = 10
            return i + 15
        self.request = FakeRequestMR(content=gitJsonPayloadMR_editdesc)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)

    @defer.inlineCallbacks
    def testGitWithChange_WithMR_addcommit(self):
        if False:
            for i in range(10):
                print('nop')
        self.request = FakeRequestMR(content=gitJsonPayloadMR_addcommit)
        res = (yield self.request.test_render(self.changeHook))
        self.check_changes_mr_event(res, codebase='MyCodebase', timestamp=1526395871)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['category'], 'merge_request')

    @defer.inlineCallbacks
    def testGitWithChange_WithMR_close(self):
        if False:
            i = 10
            return i + 15
        self.request = FakeRequestMR(content=gitJsonPayloadMR_close)
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)

    @defer.inlineCallbacks
    def testGitWithChange_WithMR_reopen(self):
        if False:
            i = 10
            return i + 15
        self.request = FakeRequestMR(content=gitJsonPayloadMR_reopen)
        res = (yield self.request.test_render(self.changeHook))
        self.check_changes_mr_event(res, codebase='MyCodebase', timestamp=1526395871)
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['category'], 'merge_request')

    @defer.inlineCallbacks
    def testGitWithChange_WithMR_open_forked(self):
        if False:
            print('Hello World!')
        self.request = FakeRequestMR(content=gitJsonPayloadMR_open_forked)
        res = (yield self.request.test_render(self.changeHook))
        self.check_changes_mr_event(res, codebase='MyCodebase', timestamp=1526736926, source_repo='https://gitlab.example.com/build/awesome_project.git')
        change = self.changeHook.master.data.updates.changesAdded[0]
        self.assertEqual(change['category'], 'merge_request')

    @defer.inlineCallbacks
    def testGitWithChange_WithMR_commented(self):
        if False:
            i = 10
            return i + 15
        self.request = FakeRequestMR(content=gitJsonPayloadMR_commented)
        res = (yield self.request.test_render(self.changeHook))
        self.check_changes_mr_event_by_comment(res, codebase='MyCodebase', timestamp=1643361192, project='awesome_project', source_repo='https://gitlab.example.com/mmusterman/awesome_project.git', source_branch='fix-missing-tests')

class TestChangeHookConfiguredWithSecret(unittest.TestCase, TestReactorMixin):
    _SECRET = 'thesecret'

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.master = fakeMasterForHooks(self)
        fakeStorageService = FakeSecretStorage()
        fakeStorageService.reconfigService(secretdict={'secret_key': self._SECRET})
        self.secretService = SecretManager()
        self.secretService.services = [fakeStorageService]
        self.master.addService(self.secretService)
        self.changeHook = change_hook.ChangeHookResource(dialects={'gitlab': {'secret': util.Secret('secret_key')}}, master=self.master)

    @defer.inlineCallbacks
    def test_missing_secret(self):
        if False:
            i = 10
            return i + 15
        self.request = FakeRequest(content=gitJsonPayloadTag)
        self.request.uri = b'/change_hook/gitlab'
        self.request.args = {b'codebase': [b'MyCodebase']}
        self.request.method = b'POST'
        self.request.received_headers[_HEADER_EVENT] = b'Push Hook'
        yield self.request.test_render(self.changeHook)
        expected = b'Invalid secret'
        self.assertEqual(self.request.written, expected)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 0)

    @defer.inlineCallbacks
    def test_valid_secret(self):
        if False:
            while True:
                i = 10
        self.request = FakeRequest(content=gitJsonPayload)
        self.request.received_headers[_HEADER_GITLAB_TOKEN] = self._SECRET
        self.request.received_headers[_HEADER_EVENT] = b'Push Hook'
        self.request.uri = b'/change_hook/gitlab'
        self.request.method = b'POST'
        yield self.request.test_render(self.changeHook)
        self.assertEqual(len(self.changeHook.master.data.updates.changesAdded), 2)