"""
github3.orgs
============

This module contains all of the classes related to organizations.

"""
from __future__ import unicode_literals
import warnings
from json import dumps
from .events import Event
from .models import BaseAccount, GitHubCore
from .repos import Repository
from .users import User
from .decorators import requires_auth
from uritemplate import URITemplate

class Team(GitHubCore):
    """The :class:`Team <Team>` object.

    Two team instances can be checked like so::

        t1 == t2
        t1 != t2

    And is equivalent to::

        t1.id == t2.id
        t1.id != t2.id

    See also: http://developer.github.com/v3/orgs/teams/

    """
    members_roles = frozenset(['member', 'maintainer', 'all'])

    def _update_attributes(self, team):
        if False:
            return 10
        self._api = team.get('url', '')
        self.name = team.get('name')
        self.id = team.get('id')
        self.permission = team.get('permission')
        self.members_count = team.get('members_count')
        members = team.get('members_url')
        self.members_urlt = URITemplate(members) if members else None
        self.repos_count = team.get('repos_count')
        self.repositories_url = team.get('repositories_url')

    def _repr(self):
        if False:
            return 10
        return '<Team [{0}]>'.format(self.name)

    @requires_auth
    def add_member(self, username):
        if False:
            return 10
        'Add ``username`` to this team.\n\n        :param str username: the username of the user you would like to add to\n            the team.\n        :returns: bool\n        '
        warnings.warn('This is no longer supported by the GitHub API, see https://developer.github.com/changes/2014-09-23-one-more-week-before-the-add-team-member-api-breaking-change/', DeprecationWarning)
        url = self._build_url('members', username, base_url=self._api)
        return self._boolean(self._put(url), 204, 404)

    @requires_auth
    def add_repository(self, repository):
        if False:
            while True:
                i = 10
        "Add ``repository`` to this team.\n\n        :param str repository: (required), form: 'user/repo'\n        :returns: bool\n        "
        url = self._build_url('repos', repository, base_url=self._api)
        return self._boolean(self._put(url), 204, 404)

    @requires_auth
    def delete(self):
        if False:
            return 10
        'Delete this team.\n\n        :returns: bool\n        '
        return self._boolean(self._delete(self._api), 204, 404)

    @requires_auth
    def edit(self, name, permission=''):
        if False:
            while True:
                i = 10
        "Edit this team.\n\n        :param str name: (required)\n        :param str permission: (optional), ('pull', 'push', 'admin')\n        :returns: bool\n        "
        if name:
            data = {'name': name, 'permission': permission}
            json = self._json(self._patch(self._api, data=dumps(data)), 200)
            if json:
                self._update_attributes(json)
                return True
        return False

    @requires_auth
    def has_repository(self, repository):
        if False:
            return 10
        "Check if this team has access to ``repository``.\n\n        :param str repository: (required), form: 'user/repo'\n        :returns: bool\n        "
        url = self._build_url('repos', repository, base_url=self._api)
        return self._boolean(self._get(url), 204, 404)

    @requires_auth
    def invite(self, username):
        if False:
            i = 10
            return i + 15
        "Invite the user to join this team.\n\n        This returns a dictionary like so::\n\n            {'state': 'pending', 'url': 'https://api.github.com/teams/...'}\n\n        :param str username: (required), user to invite to join this team.\n        :returns: dictionary\n        "
        url = self._build_url('memberships', username, base_url=self._api)
        return self._json(self._put(url), 200)

    @requires_auth
    def is_member(self, username):
        if False:
            print('Hello World!')
        'Check if ``login`` is a member of this team.\n\n        :param str username: (required), username name of the user\n        :returns: bool\n        '
        url = self._build_url('members', username, base_url=self._api)
        return self._boolean(self._get(url), 204, 404)

    @requires_auth
    def members(self, role=None, number=-1, etag=None):
        if False:
            while True:
                i = 10
        'Iterate over the members of this team.\n\n        :param str role: (optional), filter members returned by their role\n            in the team. Can be one of: ``"member"``, ``"maintainer"``,\n            ``"all"``. Default: ``"all"``.\n        :param int number: (optional), number of users to iterate over.\n            Default: -1 iterates over all values\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`User <github3.users.User>`\\ s\n        '
        headers = {}
        params = {}
        if role in self.members_roles:
            params['role'] = role
            headers['Accept'] = 'application/vnd.github.ironman-preview+json'
        url = self._build_url('members', base_url=self._api)
        return self._iter(int(number), url, User, params=params, etag=etag, headers=headers)

    @requires_auth
    def repositories(self, number=-1, etag=None):
        if False:
            while True:
                i = 10
        'Iterate over the repositories this team has access to.\n\n        :param int number: (optional), number of repos to iterate over.\n            Default: -1 iterates over all values\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`Repository <github3.repos.Repository>`\n            objects\n        '
        headers = {'Accept': 'application/vnd.github.ironman-preview+json'}
        url = self._build_url('repos', base_url=self._api)
        return self._iter(int(number), url, Repository, etag=etag, headers=headers)

    @requires_auth
    def membership_for(self, username):
        if False:
            i = 10
            return i + 15
        'Retrieve the membership information for the user.\n\n        :param str username: (required), name of the user\n        :returns: dictionary\n        '
        url = self._build_url('memberships', username, base_url=self._api)
        json = self._json(self._get(url), 200)
        return json or {}

    @requires_auth
    def remove_member(self, username):
        if False:
            i = 10
            return i + 15
        'Remove ``username`` from this team.\n\n        :param str username: (required), username of the member to remove\n        :returns: bool\n        '
        warnings.warn('This is no longer supported by the GitHub API, see https://developer.github.com/changes/2014-09-23-one-more-week-before-the-add-team-member-api-breaking-change/', DeprecationWarning)
        url = self._build_url('members', username, base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)

    @requires_auth
    def revoke_membership(self, username):
        if False:
            return 10
        "Revoke this user's team membership.\n\n        :param str username: (required), name of the team member\n        :returns: bool\n        "
        url = self._build_url('memberships', username, base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)

    @requires_auth
    def remove_repository(self, repository):
        if False:
            while True:
                i = 10
        "Remove ``repository`` from this team.\n\n        :param str repository: (required), form: 'user/repo'\n        :returns: bool\n        "
        url = self._build_url('repos', repository, base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)

class Organization(BaseAccount):
    """The :class:`Organization <Organization>` object.

    Two organization instances can be checked like so::

        o1 == o2
        o1 != o2

    And is equivalent to::

        o1.id == o2.id
        o1.id != o2.id

    See also: http://developer.github.com/v3/orgs/

    """
    members_filters = frozenset(['2fa_disabled', 'all'])
    members_roles = frozenset(['all', 'admin', 'member'])

    def _update_attributes(self, org):
        if False:
            return 10
        super(Organization, self)._update_attributes(org)
        self.type = self.type or 'Organization'
        self.events_url = org.get('events_url')
        self.private_repos = org.get('private_repos', 0)
        members = org.get('members_url')
        self.members_urlt = URITemplate(members) if members else None
        members = org.get('public_members_url')
        self.public_members_urlt = URITemplate(members) if members else None
        self.repos_url = org.get('repos_url')

    @requires_auth
    def add_member(self, username, team_id):
        if False:
            for i in range(10):
                print('nop')
        'Add ``username`` to ``team`` and thereby to this organization.\n\n        .. warning::\n            This method is no longer valid. To add a member to a team, you\n            must now retrieve the team directly, and use the ``invite``\n            method.\n\n        .. warning::\n            This method is no longer valid. To add a member to a team, you\n            must now retrieve the team directly, and use the ``invite``\n            method.\n\n        Any user that is to be added to an organization, must be added\n        to a team as per the GitHub api.\n\n        .. versionchanged:: 1.0\n\n            The second parameter used to be ``team`` but has been changed to\n            ``team_id``. This parameter is now required to be an integer to\n            improve performance of this method.\n\n        :param str username: (required), login name of the user to be added\n        :param int team_id: (required), team id\n        :returns: bool\n        '
        warnings.warn('This is no longer supported by the GitHub API, see https://developer.github.com/changes/2014-09-23-one-more-week-before-the-add-team-member-api-breaking-change/', DeprecationWarning)
        if int(team_id) < 0:
            return False
        url = self._build_url('teams', str(team_id), 'members', str(username))
        return self._boolean(self._put(url), 204, 404)

    @requires_auth
    def add_repository(self, repository, team_id):
        if False:
            print('Hello World!')
        "Add ``repository`` to ``team``.\n\n        .. versionchanged:: 1.0\n\n            The second parameter used to be ``team`` but has been changed to\n            ``team_id``. This parameter is now required to be an integer to\n            improve performance of this method.\n\n        :param str repository: (required), form: 'user/repo'\n        :param int team_id: (required), team id\n        :returns: bool\n        "
        if int(team_id) < 0:
            return False
        url = self._build_url('teams', str(team_id), 'repos', str(repository))
        return self._boolean(self._put(url), 204, 404)

    @requires_auth
    def create_repository(self, name, description='', homepage='', private=False, has_issues=True, has_wiki=True, team_id=0, auto_init=False, gitignore_template='', license_template=''):
        if False:
            print('Hello World!')
        'Create a repository for this organization.\n\n        If the client is authenticated and a member of the organization, this\n        will create a new repository in the organization.\n\n        :param str name: (required), name of the repository\n        :param str description: (optional)\n        :param str homepage: (optional)\n        :param bool private: (optional), If ``True``, create a private\n            repository. API default: ``False``\n        :param bool has_issues: (optional), If ``True``, enable issues for\n            this repository. API default: ``True``\n        :param bool has_wiki: (optional), If ``True``, enable the wiki for\n            this repository. API default: ``True``\n        :param int team_id: (optional), id of the team that will be granted\n            access to this repository\n        :param bool auto_init: (optional), auto initialize the repository.\n        :param str gitignore_template: (optional), name of the template; this\n            is ignored if auto_int = False.\n        :param str license_template: (optional), name of the license; this\n            is ignored if auto_int = False.\n        :returns: :class:`Repository <github3.repos.Repository>`\n\n        .. warning: ``name`` should be no longer than 100 characters\n        '
        url = self._build_url('repos', base_url=self._api)
        data = {'name': name, 'description': description, 'homepage': homepage, 'private': private, 'has_issues': has_issues, 'has_wiki': has_wiki, 'license_template': license_template, 'auto_init': auto_init, 'gitignore_template': gitignore_template}
        if int(team_id) > 0:
            data.update({'team_id': team_id})
        json = self._json(self._post(url, data), 201)
        return self._instance_or_null(Repository, json)

    @requires_auth
    def conceal_member(self, username):
        if False:
            for i in range(10):
                print('nop')
        "Conceal ``username``'s membership in this organization.\n\n        :param str username: username of the organization member to conceal\n        :returns: bool\n        "
        url = self._build_url('public_members', username, base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)

    @requires_auth
    def create_team(self, name, repo_names=[], permission=''):
        if False:
            print('Hello World!')
        "Create a new team and return it.\n\n        This only works if the authenticated user owns this organization.\n\n        :param str name: (required), name to be given to the team\n        :param list repo_names: (optional) repositories, e.g.\n            ['github/dotfiles']\n        :param str permission: (optional), options:\n\n            - ``pull`` -- (default) members can not push or administer\n                repositories accessible by this team\n            - ``push`` -- members can push and pull but not administer\n                repositories accessible by this team\n            - ``admin`` -- members can push, pull and administer\n                repositories accessible by this team\n\n        :returns: :class:`Team <Team>`\n        "
        data = {'name': name, 'repo_names': repo_names, 'permission': permission}
        url = self._build_url('teams', base_url=self._api)
        json = self._json(self._post(url, data), 201)
        return self._instance_or_null(Team, json)

    @requires_auth
    def edit(self, billing_email=None, company=None, email=None, location=None, name=None):
        if False:
            print('Hello World!')
        'Edit this organization.\n\n        :param str billing_email: (optional) Billing email address (private)\n        :param str company: (optional)\n        :param str email: (optional) Public email address\n        :param str location: (optional)\n        :param str name: (optional)\n        :returns: bool\n        '
        json = None
        data = {'billing_email': billing_email, 'company': company, 'email': email, 'location': location, 'name': name}
        self._remove_none(data)
        if data:
            json = self._json(self._patch(self._api, data=dumps(data)), 200)
        if json:
            self._update_attributes(json)
            return True
        return False

    def is_member(self, username):
        if False:
            while True:
                i = 10
        "Check if the user named ``username`` is a member.\n\n        :param str username: name of the user you'd like to check\n        :returns: bool\n        "
        url = self._build_url('members', username, base_url=self._api)
        return self._boolean(self._get(url), 204, 404)

    def is_public_member(self, username):
        if False:
            return 10
        "Check if the user named ``username`` is a public member.\n\n        :param str username: name of the user you'd like to check\n        :returns: bool\n        "
        url = self._build_url('public_members', username, base_url=self._api)
        return self._boolean(self._get(url), 204, 404)

    def events(self, number=-1, etag=None):
        if False:
            print('Hello World!')
        'Iterate over events for this org.\n\n        :param int number: (optional), number of events to return. Default: -1\n            iterates over all events available.\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`Event <github3.events.Event>`\\ s\n        '
        url = self._build_url('events', base_url=self._api)
        return self._iter(int(number), url, Event, etag=etag)

    def members(self, filter=None, role=None, number=-1, etag=None):
        if False:
            for i in range(10):
                print('nop')
        'Iterate over members of this organization.\n\n        :param str filter: (optional), filter members returned by this method.\n            Can be one of: ``"2fa_disabled"``, ``"all",``. Default: ``"all"``.\n            Filtering by ``"2fa_disabled"`` is only available for organization\n            owners with private repositories.\n        :param str role: (optional), filter members returned by their role.\n            Can be one of: ``"all"``, ``"admin"``, ``"member"``. Default:\n            ``"all"``.\n        :param int number: (optional), number of members to return. Default:\n            -1 will return all available.\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`User <github3.users.User>`\\ s\n        '
        headers = {}
        params = {}
        if filter in self.members_filters:
            params['filter'] = filter
        if role in self.members_roles:
            params['role'] = role
            headers['Accept'] = 'application/vnd.github.ironman-preview+json'
        url = self._build_url('members', base_url=self._api)
        return self._iter(int(number), url, User, params=params, etag=etag, headers=headers)

    def public_members(self, number=-1, etag=None):
        if False:
            while True:
                i = 10
        'Iterate over public members of this organization.\n\n        :param int number: (optional), number of members to return. Default:\n            -1 will return all available.\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`User <github3.users.User>`\\ s\n        '
        url = self._build_url('public_members', base_url=self._api)
        return self._iter(int(number), url, User, etag=etag)

    def repositories(self, type='', number=-1, etag=None):
        if False:
            while True:
                i = 10
        "Iterate over repos for this organization.\n\n        :param str type: (optional), accepted values:\n            ('all', 'public', 'member', 'private', 'forks', 'sources'), API\n            default: 'all'\n        :param int number: (optional), number of members to return. Default:\n            -1 will return all available.\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`Repository <github3.repos.Repository>`\n        "
        url = self._build_url('repos', base_url=self._api)
        params = {}
        if type in ('all', 'public', 'member', 'private', 'forks', 'sources'):
            params['type'] = type
        return self._iter(int(number), url, Repository, params, etag)

    @requires_auth
    def teams(self, number=-1, etag=None):
        if False:
            i = 10
            return i + 15
        'Iterate over teams that are part of this organization.\n\n        :param int number: (optional), number of teams to return. Default: -1\n            returns all available teams.\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`Team <Team>`\\ s\n        '
        url = self._build_url('teams', base_url=self._api)
        return self._iter(int(number), url, Team, etag=etag)

    @requires_auth
    def publicize_member(self, username):
        if False:
            while True:
                i = 10
        "Make ``username``'s membership in this organization public.\n\n        :param str username: the name of the user whose membership you wish to\n            publicize\n        :returns: bool\n        "
        url = self._build_url('public_members', username, base_url=self._api)
        return self._boolean(self._put(url), 204, 404)

    @requires_auth
    def remove_member(self, username):
        if False:
            while True:
                i = 10
        'Remove the user named ``username`` from this organization.\n\n        :param str username: name of the user to remove from the org\n        :returns: bool\n        '
        url = self._build_url('members', username, base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)

    @requires_auth
    def remove_repository(self, repository, team_id):
        if False:
            i = 10
            return i + 15
        "Remove ``repository`` from the team with ``team_id``.\n\n        :param str repository: (required), form: 'user/repo'\n        :param int team_id: (required)\n        :returns: bool\n        "
        if int(team_id) > 0:
            url = self._build_url('teams', str(team_id), 'repos', str(repository))
            return self._boolean(self._delete(url), 204, 404)
        return False

    @requires_auth
    def team(self, team_id):
        if False:
            return 10
        'Return the team specified by ``team_id``.\n\n        :param int team_id: (required), unique id for the team\n        :returns: :class:`Team <Team>`\n        '
        json = None
        if int(team_id) > 0:
            url = self._build_url('teams', str(team_id))
            json = self._json(self._get(url), 200)
        return self._instance_or_null(Team, json)

class Membership(GitHubCore):
    """The wrapper for information about Team and Organization memberships."""

    def _repr(self):
        if False:
            for i in range(10):
                print('nop')
        return '<Membership [{0}]>'.format(self.organization)

    def _update_attributes(self, membership):
        if False:
            for i in range(10):
                print('nop')
        self._api = membership.get('url')
        self.organization = Organization(membership.get('organization', {}), self)
        self.state = membership.get('state', '')
        self.organization_url = membership.get('organization_url')
        self.active = self.state.lower() == 'active'
        self.pending = self.state.lower() == 'pending'

    @requires_auth
    def edit(self, state):
        if False:
            for i in range(10):
                print('nop')
        'Edit the user\'s membership.\n\n        :param str state: (required), the state the membership should be in.\n            Only accepts ``"active"``.\n        :returns: whether the edit was successful or not\n        :rtype: bool\n        '
        if state and state.lower() == 'active':
            data = dumps({'state': state.lower()})
            json = self._json(self._patch(self._api, data=data))
            self._update_attributes(json)
            return True
        return False