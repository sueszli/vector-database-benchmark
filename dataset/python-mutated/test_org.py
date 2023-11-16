"""
.. module: security_monkey.tests.watchers.github.test_org
    :platform: Unix
.. version:: $$VERSION$$
.. moduleauthor::  Mike Grima <mgrima@netflix.com>
"""
import json
from security_monkey import app
from security_monkey.datastore import Account, Technology, AccountType, ExceptionLogs
from security_monkey.exceptions import InvalidResponseCodeFromGitHubError
from security_monkey.tests import SecurityMonkeyTestCase, db
import mock
ORG_RESPONSE = '\n{\n    "login": "Netflix",\n    "id": 913567,\n    "url": "https://api.github.com/orgs/Netflix",\n    "repos_url": "https://api.github.com/orgs/Netflix/repos",\n    "events_url": "https://api.github.com/orgs/Netflix/events",\n    "hooks_url": "https://api.github.com/orgs/Netflix/hooks",\n    "issues_url": "https://api.github.com/orgs/Netflix/issues",\n    "members_url": "https://api.github.com/orgs/Netflix/members{/member}",\n    "public_members_url": "https://api.github.com/orgs/Netflix/public_members{/member}",\n    "avatar_url": "https://avatars3.githubusercontent.com/u/913567?v=4",\n    "description": "Netflix Open Source Platform",\n    "name": "Netflix, Inc.",\n    "company": null,\n    "blog": "http://netflix.github.io/",\n    "location": "Los Gatos, California",\n    "email": "netflixoss@netflix.com",\n    "has_organization_projects": true,\n    "has_repository_projects": true,\n    "public_repos": 130,\n    "public_gists": 0,\n    "followers": 0,\n    "following": 0,\n    "html_url": "https://github.com/Netflix",\n    "created_at": "2011-07-13T20:20:01Z",\n    "updated_at": "2017-08-16T09:44:42Z",\n    "type": "Organization"\n}'
MEMBERS_PAGE_ONE = '[\n    {\n        "login": "----notarealuserone----",\n        "id": 1,\n        "avatar_url": "https://avatars0.githubusercontent.com/u/1?v=4",\n        "gravatar_id": "",\n        "url": "https://api.github.com/users/----notarealuserone----",\n        "html_url": "https://github.com/----notarealuserone----",\n        "followers_url": "https://api.github.com/users/----notarealuserone----/followers",\n        "following_url": "https://api.github.com/users/----notarealuserone----/following{/other_user}",\n        "gists_url": "https://api.github.com/users/----notarealuserone----/gists{/gist_id}",\n        "starred_url": "https://api.github.com/users/----notarealuserone----/starred{/owner}{/repo}",\n        "subscriptions_url": "https://api.github.com/users/----notarealuserone----/subscriptions",\n        "organizations_url": "https://api.github.com/users/----notarealuserone----/orgs",\n        "repos_url": "https://api.github.com/users/----notarealuserone----/repos",\n        "events_url": "https://api.github.com/users/----notarealuserone----/events{/privacy}",\n        "received_events_url": "https://api.github.com/users/----notarealuserone----/received_events",\n        "type": "User",\n        "site_admin": false\n    }\n]'
MEMBERS_PAGE_TWO = '\n[\n    {\n        "login": "----notarealusertwo----",\n        "id": 1728105,\n        "avatar_url": "https://avatars1.githubusercontent.com/u/1728105?v=4",\n        "gravatar_id": "",\n        "url": "https://api.github.com/users/----notarealusertwo----",\n        "html_url": "https://github.com/----notarealusertwo----",\n        "followers_url": "https://api.github.com/users/----notarealusertwo----/followers",\n        "following_url": "https://api.github.com/users/----notarealusertwo----/following{/other_user}",\n        "gists_url": "https://api.github.com/users/----notarealusertwo----/gists{/gist_id}",\n        "starred_url": "https://api.github.com/users/----notarealusertwo----/starred{/owner}{/repo}",\n        "subscriptions_url": "https://api.github.com/users/----notarealusertwo----/subscriptions",\n        "organizations_url": "https://api.github.com/users/----notarealusertwo----/orgs",\n        "repos_url": "https://api.github.com/users/----notarealusertwo----/repos",\n        "events_url": "https://api.github.com/users/----notarealusertwo----/events{/privacy}",\n        "received_events_url": "https://api.github.com/users/----notarealusertwo----/received_events",\n        "type": "User",\n        "site_admin": false\n    }\n]'
TEAMS_PAGE_ONE = '[\n    {\n        "id": 1,\n        "url": "https://api.github.com/teams/1",\n        "name": "Justice League",\n        "slug": "justice-league",\n        "description": "A great team.",\n        "privacy": "closed",\n        "permission": "admin",\n        "members_url": "https://api.github.com/teams/1/members{/member}",\n        "repositories_url": "https://api.github.com/teams/1/repos"\n    }\n]'
TEAMS_PAGE_TWO = '[\n    {\n        "id": 2,\n        "url": "https://api.github.com/teams/2",\n        "name": "Team2",\n        "slug": "Team2",\n        "description": "A great team.",\n        "privacy": "closed",\n        "permission": "admin",\n        "members_url": "https://api.github.com/teams/2/members{/member}",\n        "repositories_url": "https://api.github.com/teams/2/repos"\n    }\n]'

class MockOrgDetails:

    def __init__(self, status_code):
        if False:
            print('Hello World!')
        self.json_data = ORG_RESPONSE
        self.status_code = status_code

    def json(self):
        if False:
            return 10
        return json.loads(self.json_data)

def mock_get_org_details(*args, **kwargs):
    if False:
        while True:
            i = 10
    if args[0] == 'https://api.github.com/orgs/FAILURE':
        return MockOrgDetails(404)
    return MockOrgDetails(200)

class MockMemberDetails:

    def __init__(self, status_code, page):
        if False:
            for i in range(10):
                print('nop')
        if page == 1:
            self.json_data = MEMBERS_PAGE_ONE
            self.links = {'last': 'this is not the last page'}
        else:
            self.json_data = MEMBERS_PAGE_TWO
            self.links = {}
        self.status_code = status_code

    def json(self):
        if False:
            while True:
                i = 10
        return json.loads(self.json_data)

def mock_get_member_details(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    if 'FAILURE' in args[0]:
        return MockMemberDetails(404, 1)
    return MockMemberDetails(200, kwargs['params']['page'])

class MockTeamList:

    def __init__(self, status_code, page):
        if False:
            while True:
                i = 10
        if page == 1:
            self.json_data = TEAMS_PAGE_ONE
            self.links = {'last': 'this is not the last page'}
        else:
            self.json_data = TEAMS_PAGE_TWO
            self.links = {}
        self.status_code = status_code

    def json(self):
        if False:
            i = 10
            return i + 15
        return json.loads(self.json_data)

def mock_list_org_teams(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    if 'FAILURE' in args[0]:
        return MockTeamList(404, 1)
    return MockTeamList(200, kwargs['params']['page'])

def mock_slurp(*args, **kwargs):
    if False:
        return 10
    if 'members' in args[0] or 'outside_collaborators' in args[0]:
        return mock_get_member_details(*args, **kwargs)
    elif 'teams' in args[0]:
        return mock_list_org_teams(*args, **kwargs)
    return mock_get_org_details(*args, **kwargs)

class GitHubOrgWatcherTestCase(SecurityMonkeyTestCase):

    def pre_test_setup(self):
        if False:
            return 10
        self.account_type = AccountType(name='GitHub')
        db.session.add(self.account_type)
        db.session.commit()
        app.config['GITHUB_CREDENTIALS'] = {'Org-one': 'token-one', 'FAILURE': 'FAILURE'}
        db.session.add(Account(name='Org-one', account_type_id=self.account_type.id, identifier='Org-one', active=True, third_party=False))
        self.technology = Technology(name='organization')
        db.session.add(self.technology)
        db.session.commit()

    @mock.patch('requests.get', side_effect=mock_get_org_details)
    def test_get_org_details(self, mock_get):
        if False:
            for i in range(10):
                print('nop')
        from security_monkey.watchers.github.org import GitHubOrg
        org_watcher = GitHubOrg(accounts=['Org-one'])
        result = org_watcher.get_org_details('Org-one')
        assert json.dumps(result, indent=4, sort_keys=True) == json.dumps(json.loads(ORG_RESPONSE), indent=4, sort_keys=True)
        with self.assertRaises(InvalidResponseCodeFromGitHubError) as _:
            org_watcher.get_org_details('FAILURE')

    @mock.patch('requests.get', side_effect=mock_get_member_details)
    def test_list_org_members(self, mock_get):
        if False:
            print('Hello World!')
        from security_monkey.watchers.github.org import GitHubOrg
        org_watcher = GitHubOrg(accounts=['Org-one'])
        result = org_watcher.list_org_members('Org-one')
        assert len(result) == 2
        with self.assertRaises(InvalidResponseCodeFromGitHubError) as _:
            org_watcher.list_org_members('FAILURE')

    @mock.patch('requests.get', side_effect=mock_get_member_details)
    def test_list_org_outside_collabs(self, mock_get):
        if False:
            return 10
        from security_monkey.watchers.github.org import GitHubOrg
        org_watcher = GitHubOrg(accounts=['Org-one'])
        result = org_watcher.list_org_outside_collabs('Org-one')
        assert len(result) == 2
        with self.assertRaises(InvalidResponseCodeFromGitHubError) as _:
            org_watcher.list_org_outside_collabs('FAILURE')

    @mock.patch('requests.get', side_effect=mock_list_org_teams)
    def test_list_org_teams(self, mock_get):
        if False:
            while True:
                i = 10
        from security_monkey.watchers.github.org import GitHubOrg
        org_watcher = GitHubOrg(accounts=['Org-one'])
        result = org_watcher.list_org_teams('Org-one')
        assert len(result) == 2
        assert result[0] == 'Justice League'
        assert result[1] == 'Team2'
        with self.assertRaises(InvalidResponseCodeFromGitHubError) as _:
            org_watcher.list_org_teams('FAILURE')

    @mock.patch('requests.get', side_effect=mock_slurp)
    def test_slurp(self, mock_get):
        if False:
            for i in range(10):
                print('nop')
        from security_monkey.watchers.github.org import GitHubOrg
        org_watcher = GitHubOrg(accounts=['Org-one'])
        (result, exc) = org_watcher.slurp()
        assert exc == {}
        assert len(result) == 1
        assert result[0].account == 'Org-one'
        assert result[0].name == 'Org-one'
        assert result[0].index == 'organization'
        assert len(ExceptionLogs.query.all()) == 0
        db.session.add(Account(name='FAILURE', account_type_id=self.account_type.id, identifier='FAILURE', active=True, third_party=False))
        db.session.commit()
        org_watcher = GitHubOrg(accounts=['FAILURE'])
        (result, exc) = org_watcher.slurp()
        assert len(exc) == 1
        assert len(ExceptionLogs.query.all()) == 1