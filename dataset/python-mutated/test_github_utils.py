"""
.. module: security_monkey.tests.utilities.test_github_utils
    :platform: Unix
.. version:: $$VERSION$$
.. moduleauthor::  Mike Grima <mgrima@netflix.com>
"""
import json
from security_monkey import db, app
from security_monkey.common.github.util import get_github_creds, iter_org, strip_url_fields
from security_monkey.datastore import AccountType, Account, AccountTypeCustomValues
from security_monkey.exceptions import GitHubCredsError
from security_monkey.tests import SecurityMonkeyTestCase

class GitHubUtilsTestCase(SecurityMonkeyTestCase):

    def pre_test_setup(self):
        if False:
            for i in range(10):
                print('nop')
        if app.config.get('GITHUB_CREDENTIALS'):
            del app.config['GITHUB_CREDENTIALS']
        self.account_type = AccountType(name='GitHub')
        self.accounts = []
        db.session.add(self.account_type)
        db.session.commit()
        for x in ['one', 'two', 'three']:
            account = Account(name='Org-{}'.format(x), account_type_id=self.account_type.id, identifier='Org-{}'.format(x), active=True)
            account.custom_fields.append(AccountTypeCustomValues(name='access_token_file', value='security_monkey/tests/utilities/templates/github_creds'))
            db.session.add(account)
            self.accounts.append(account)
        db.session.commit()

    def test_get_creds_file(self):
        if False:
            while True:
                i = 10
        creds = get_github_creds(['Org-one', 'Org-two', 'Org-three'])
        for x in ['one', 'two', 'three']:
            assert creds['Org-{}'.format(x)] == 'token-{}'.format(x)
        db.session.add(Account(name='Org-BAD', account_type_id=self.account_type.id, identifier='Org-BAD', active=True))
        db.session.commit()
        with self.assertRaises(GitHubCredsError) as _:
            get_github_creds(['Org-BAD'])

    def test_get_creds_env(self):
        if False:
            print('Hello World!')
        app.config['GITHUB_CREDENTIALS'] = {'AnotherOrg': 'AnotherCred'}
        db.session.add(Account(name='AnotherOrg', account_type_id=self.account_type.id, identifier='AnotherOrg', active=True))
        db.session.commit()
        creds = get_github_creds(['AnotherOrg'])
        assert isinstance(creds, dict)
        assert creds['AnotherOrg'] == 'AnotherCred'
        creds = get_github_creds(['Org-one', 'Org-two', 'Org-three', 'AnotherOrg'])
        assert isinstance(creds, dict)
        assert len(list(creds.keys())) == 4

    def test_iter_org_decorator(self):
        if False:
            return 10
        org_list = ['Org-one', 'Org-two', 'Org-three']

        @iter_org(orgs=['Org-one', 'Org-two', 'Org-three'])
        def some_func(**kwargs):
            if False:
                return 10
            assert kwargs['exception_map'] is not None
            assert kwargs['account_name'] in org_list
            org_list.remove(kwargs['account_name'])
            return ([kwargs['account_name']], kwargs['exception_map'])
        results = some_func()
        assert len(results[0]) == 3
        assert isinstance(results[1], dict)

    def test_strip_url_fields(self):
        if False:
            print('Hello World!')
        github_blob = json.loads('\n        {\n            "id": 21290287,\n            "name": "security_monkey",\n            "full_name": "Netflix/security_monkey",\n            "owner": {\n                "login": "Netflix",\n                "id": 913567,\n                "avatar_url": "https://avatars3.githubusercontent.com/u/913567?v=4",\n                "gravatar_id": "",\n                "url": "https://api.github.com/users/Netflix",\n                "html_url": "https://github.com/Netflix",\n                "followers_url": "https://api.github.com/users/Netflix/followers",\n                "following_url": "https://api.github.com/users/Netflix/following{/other_user}",\n                "gists_url": "https://api.github.com/users/Netflix/gists{/gist_id}",\n                "starred_url": "https://api.github.com/users/Netflix/starred{/owner}{/repo}",\n                "subscriptions_url": "https://api.github.com/users/Netflix/subscriptions",\n                "organizations_url": "https://api.github.com/users/Netflix/orgs",\n                "repos_url": "https://api.github.com/users/Netflix/repos",\n                "events_url": "https://api.github.com/users/Netflix/events{/privacy}",\n                "received_events_url": "https://api.github.com/users/Netflix/received_events",\n                "type": "Organization",\n                "site_admin": false\n            },\n            "private": false,\n            "html_url": "https://github.com/Netflix/security_monkey",\n            "description": "Security Monkey",\n            "fork": false,\n            "url": "https://api.github.com/repos/Netflix/security_monkey",\n            "forks_url": "https://api.github.com/repos/Netflix/security_monkey/forks",\n            "keys_url": "https://api.github.com/repos/Netflix/security_monkey/keys{/key_id}",\n            "collaborators_url": "https://api.github.com/repos/Netflix/security_monkey/collaborators{/collaborator}",\n            "teams_url": "https://api.github.com/repos/Netflix/security_monkey/teams",\n            "hooks_url": "https://api.github.com/repos/Netflix/security_monkey/hooks",\n            "issue_events_url": "https://api.github.com/repos/Netflix/security_monkey/issues/events{/number}",\n            "events_url": "https://api.github.com/repos/Netflix/security_monkey/events",\n            "assignees_url": "https://api.github.com/repos/Netflix/security_monkey/assignees{/user}",\n            "branches_url": "https://api.github.com/repos/Netflix/security_monkey/branches{/branch}",\n            "tags_url": "https://api.github.com/repos/Netflix/security_monkey/tags",\n            "blobs_url": "https://api.github.com/repos/Netflix/security_monkey/git/blobs{/sha}",\n            "git_tags_url": "https://api.github.com/repos/Netflix/security_monkey/git/tags{/sha}",\n            "git_refs_url": "https://api.github.com/repos/Netflix/security_monkey/git/refs{/sha}",\n            "trees_url": "https://api.github.com/repos/Netflix/security_monkey/git/trees{/sha}",\n            "statuses_url": "https://api.github.com/repos/Netflix/security_monkey/statuses/{sha}",\n            "languages_url": "https://api.github.com/repos/Netflix/security_monkey/languages",\n            "stargazers_url": "https://api.github.com/repos/Netflix/security_monkey/stargazers",\n            "contributors_url": "https://api.github.com/repos/Netflix/security_monkey/contributors",\n            "subscribers_url": "https://api.github.com/repos/Netflix/security_monkey/subscribers",\n            "subscription_url": "https://api.github.com/repos/Netflix/security_monkey/subscription",\n            "commits_url": "https://api.github.com/repos/Netflix/security_monkey/commits{/sha}",\n            "git_commits_url": "https://api.github.com/repos/Netflix/security_monkey/git/commits{/sha}",\n            "comments_url": "https://api.github.com/repos/Netflix/security_monkey/comments{/number}",\n            "issue_comment_url": "https://api.github.com/repos/Netflix/security_monkey/issues/comments{/number}",\n            "contents_url": "https://api.github.com/repos/Netflix/security_monkey/contents/{+path}",\n            "compare_url": "https://api.github.com/repos/Netflix/security_monkey/compare/{base}...{head}",\n            "merges_url": "https://api.github.com/repos/Netflix/security_monkey/merges",\n            "archive_url": "https://api.github.com/repos/Netflix/security_monkey/{archive_format}{/ref}",\n            "downloads_url": "https://api.github.com/repos/Netflix/security_monkey/downloads",\n            "issues_url": "https://api.github.com/repos/Netflix/security_monkey/issues{/number}",\n            "pulls_url": "https://api.github.com/repos/Netflix/security_monkey/pulls{/number}",\n            "milestones_url": "https://api.github.com/repos/Netflix/security_monkey/milestones{/number}",\n            "notifications_url": "https://api.github.com/repos/Netflix/security_monkey/notifications{?since,all,participating}",\n            "labels_url": "https://api.github.com/repos/Netflix/security_monkey/labels{/name}",\n            "releases_url": "https://api.github.com/repos/Netflix/security_monkey/releases{/id}",\n            "deployments_url": "https://api.github.com/repos/Netflix/security_monkey/deployments",\n            "created_at": "2014-06-27T21:49:56Z",\n            "updated_at": "2017-08-18T07:27:14Z",\n            "pushed_at": "2017-08-19T08:27:57Z",\n            "git_url": "git://github.com/Netflix/security_monkey.git",\n            "ssh_url": "git@github.com:Netflix/security_monkey.git",\n            "clone_url": "https://github.com/Netflix/security_monkey.git",\n            "svn_url": "https://github.com/Netflix/security_monkey",\n            "homepage": null,\n            "size": 12497,\n            "stargazers_count": 1602,\n            "watchers_count": 1602,\n            "language": "Python",\n            "has_issues": true,\n            "has_projects": true,\n            "has_downloads": true,\n            "has_wiki": true,\n            "has_pages": false,\n            "forks_count": 320,\n            "mirror_url": null,\n            "open_issues_count": 54,\n            "forks": 320,\n            "open_issues": 54,\n            "watchers": 1602,\n            "default_branch": "develop",\n            "organization": {\n                "login": "Netflix",\n                "id": 913567,\n                "avatar_url": "https://avatars3.githubusercontent.com/u/913567?v=4",\n                "gravatar_id": "",\n                "url": "https://api.github.com/users/Netflix",\n                "html_url": "https://github.com/Netflix",\n                "followers_url": "https://api.github.com/users/Netflix/followers",\n                "following_url": "https://api.github.com/users/Netflix/following{/other_user}",\n                "gists_url": "https://api.github.com/users/Netflix/gists{/gist_id}",\n                "starred_url": "https://api.github.com/users/Netflix/starred{/owner}{/repo}",\n                "subscriptions_url": "https://api.github.com/users/Netflix/subscriptions",\n                "organizations_url": "https://api.github.com/users/Netflix/orgs",\n                "repos_url": "https://api.github.com/users/Netflix/repos",\n                "events_url": "https://api.github.com/users/Netflix/events{/privacy}",\n                "received_events_url": "https://api.github.com/users/Netflix/received_events",\n                "type": "Organization",\n                "site_admin": false\n            },\n            "network_count": 320,\n            "subscribers_count": 403\n        }\n        ')
        outer_fields_to_remove = []
        total_outer_fields = len(list(github_blob.keys()))
        org_fields_to_remove = []
        total_org_fields = len(list(github_blob['organization'].keys()))
        owner_fields_to_remove = []
        total_owner_fields = len(list(github_blob['owner'].keys()))
        for field in github_blob.keys():
            if '_url' in field:
                outer_fields_to_remove.append(field)
        for field in github_blob['organization'].keys():
            if '_url' in field:
                org_fields_to_remove.append(field)
        for field in github_blob['owner'].keys():
            if '_url' in field:
                owner_fields_to_remove.append(field)
        new_blob = strip_url_fields(github_blob)
        assert total_outer_fields - len(outer_fields_to_remove) == len(list(new_blob.keys()))
        assert total_org_fields - len(org_fields_to_remove) == len(list(new_blob['organization'].keys()))
        assert total_owner_fields - len(owner_fields_to_remove) == len(list(new_blob['owner'].keys()))
        for outer in outer_fields_to_remove:
            assert not new_blob.get(outer)
        for org in org_fields_to_remove:
            assert not new_blob['organization'].get(org)
        for owner in owner_fields_to_remove:
            assert not new_blob['owner'].get(owner)