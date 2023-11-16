from __future__ import annotations
import pickle
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar
import urllib3
from urllib3.util import Retry
import github.ApplicationOAuth
import github.Auth
import github.AuthenticatedUser
import github.Enterprise
import github.Event
import github.Gist
import github.GithubApp
import github.GithubIntegration
import github.GithubRetry
import github.GitignoreTemplate
import github.License
import github.NamedUser
import github.Topic
from github import Consts
from github.GithubIntegration import GithubIntegration
from github.GithubObject import GithubObject, NotSet, Opt, is_defined
from github.GithubRetry import GithubRetry
from github.HookDelivery import HookDelivery, HookDeliverySummary
from github.HookDescription import HookDescription
from github.PaginatedList import PaginatedList
from github.RateLimit import RateLimit
from github.Requester import Requester
if TYPE_CHECKING:
    from github.AppAuthentication import AppAuthentication
    from github.ApplicationOAuth import ApplicationOAuth
    from github.AuthenticatedUser import AuthenticatedUser
    from github.Commit import Commit
    from github.ContentFile import ContentFile
    from github.Event import Event
    from github.Gist import Gist
    from github.GithubApp import GithubApp
    from github.GitignoreTemplate import GitignoreTemplate
    from github.Issue import Issue
    from github.License import License
    from github.NamedUser import NamedUser
    from github.Organization import Organization
    from github.Project import Project
    from github.ProjectColumn import ProjectColumn
    from github.Repository import Repository
    from github.Topic import Topic
TGithubObject = TypeVar('TGithubObject', bound=GithubObject)

class Github:
    """
    This is the main class you instantiate to access the Github API v3. Optional parameters allow different authentication methods.
    """
    __requester: Requester
    default_retry = GithubRetry()

    def __init__(self, login_or_token: str | None=None, password: str | None=None, jwt: str | None=None, app_auth: AppAuthentication | None=None, base_url: str=Consts.DEFAULT_BASE_URL, timeout: int=Consts.DEFAULT_TIMEOUT, user_agent: str=Consts.DEFAULT_USER_AGENT, per_page: int=Consts.DEFAULT_PER_PAGE, verify: bool | str=True, retry: int | Retry | None=default_retry, pool_size: int | None=None, seconds_between_requests: float | None=Consts.DEFAULT_SECONDS_BETWEEN_REQUESTS, seconds_between_writes: float | None=Consts.DEFAULT_SECONDS_BETWEEN_WRITES, auth: github.Auth.Auth | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param login_or_token: string deprecated, use auth=github.Auth.Login(...) or auth=github.Auth.Token(...) instead\n        :param password: string deprecated, use auth=github.Auth.Login(...) instead\n        :param jwt: string deprecated, use auth=github.Auth.AppAuth(...) or auth=github.Auth.AppAuthToken(...) instead\n        :param app_auth: github.AppAuthentication deprecated, use auth=github.Auth.AppInstallationAuth(...) instead\n        :param base_url: string\n        :param timeout: integer\n        :param user_agent: string\n        :param per_page: int\n        :param verify: boolean or string\n        :param retry: int or urllib3.util.retry.Retry object,\n                      defaults to github.Github.default_retry,\n                      set to None to disable retries\n        :param pool_size: int\n        :param seconds_between_requests: float\n        :param seconds_between_writes: float\n        :param auth: authentication method\n        '
        assert login_or_token is None or isinstance(login_or_token, str), login_or_token
        assert password is None or isinstance(password, str), password
        assert jwt is None or isinstance(jwt, str), jwt
        assert isinstance(base_url, str), base_url
        assert isinstance(timeout, int), timeout
        assert user_agent is None or isinstance(user_agent, str), user_agent
        assert isinstance(per_page, int), per_page
        assert isinstance(verify, (bool, str)), verify
        assert retry is None or isinstance(retry, int) or isinstance(retry, urllib3.util.Retry), retry
        assert pool_size is None or isinstance(pool_size, int), pool_size
        assert seconds_between_requests is None or seconds_between_requests >= 0
        assert seconds_between_writes is None or seconds_between_writes >= 0
        assert auth is None or isinstance(auth, github.Auth.Auth), auth
        if password is not None:
            warnings.warn('Arguments login_or_token and password are deprecated, please use auth=github.Auth.Login(...) instead', category=DeprecationWarning)
            auth = github.Auth.Login(login_or_token, password)
        elif login_or_token is not None:
            warnings.warn('Argument login_or_token is deprecated, please use auth=github.Auth.Token(...) instead', category=DeprecationWarning)
            auth = github.Auth.Token(login_or_token)
        elif jwt is not None:
            warnings.warn('Argument jwt is deprecated, please use auth=github.Auth.AppAuth(...) or auth=github.Auth.AppAuthToken(...) instead', category=DeprecationWarning)
            auth = github.Auth.AppAuthToken(jwt)
        elif app_auth is not None:
            warnings.warn('Argument app_auth is deprecated, please use auth=github.Auth.AppInstallationAuth(...) instead', category=DeprecationWarning)
            auth = app_auth
        self.__requester = Requester(auth, base_url, timeout, user_agent, per_page, verify, retry, pool_size, seconds_between_requests, seconds_between_writes)

    def close(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Close connections to the server. Alternatively, use the Github object as a context manager:\n\n        .. code-block:: python\n\n          with github.Github(...) as gh:\n            # do something\n        '
        self.__requester.close()

    def __enter__(self) -> Github:
        if False:
            print('Hello World!')
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if False:
            while True:
                i = 10
        self.close()

    @property
    def FIX_REPO_GET_GIT_REF(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.__requester.FIX_REPO_GET_GIT_REF

    @FIX_REPO_GET_GIT_REF.setter
    def FIX_REPO_GET_GIT_REF(self, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.__requester.FIX_REPO_GET_GIT_REF = value

    @property
    def per_page(self) -> int:
        if False:
            return 10
        return self.__requester.per_page

    @per_page.setter
    def per_page(self, value: int) -> None:
        if False:
            return 10
        self.__requester.per_page = value

    @property
    def rate_limiting(self) -> tuple[int, int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        First value is requests remaining, second value is request limit.\n        '
        (remaining, limit) = self.__requester.rate_limiting
        if limit < 0:
            self.get_rate_limit()
        return self.__requester.rate_limiting

    @property
    def rate_limiting_resettime(self) -> int:
        if False:
            print('Hello World!')
        '\n        Unix timestamp indicating when rate limiting will reset.\n        '
        if self.__requester.rate_limiting_resettime == 0:
            self.get_rate_limit()
        return self.__requester.rate_limiting_resettime

    def get_rate_limit(self) -> RateLimit:
        if False:
            i = 10
            return i + 15
        '\n        Rate limit status for different resources (core/search/graphql).\n\n        :calls: `GET /rate_limit <https://docs.github.com/en/rest/reference/rate-limit>`_\n        '
        (headers, data) = self.__requester.requestJsonAndCheck('GET', '/rate_limit')
        return RateLimit(self.__requester, headers, data['resources'], True)

    @property
    def oauth_scopes(self) -> list[str] | None:
        if False:
            print('Hello World!')
        '\n        :type: list of string\n        '
        return self.__requester.oauth_scopes

    def get_license(self, key: Opt[str]=NotSet) -> License:
        if False:
            while True:
                i = 10
        '\n        :calls: `GET /license/{license} <https://docs.github.com/en/rest/reference/licenses#get-a-license>`_\n        '
        assert isinstance(key, str), key
        (headers, data) = self.__requester.requestJsonAndCheck('GET', f'/licenses/{key}')
        return github.License.License(self.__requester, headers, data, completed=True)

    def get_licenses(self) -> PaginatedList[License]:
        if False:
            for i in range(10):
                print('nop')
        '\n        :calls: `GET /licenses <https://docs.github.com/en/rest/reference/licenses#get-all-commonly-used-licenses>`_\n        '
        url_parameters: dict[str, Any] = {}
        return PaginatedList(github.License.License, self.__requester, '/licenses', url_parameters)

    def get_events(self) -> PaginatedList[Event]:
        if False:
            print('Hello World!')
        '\n        :calls: `GET /events <https://docs.github.com/en/rest/reference/activity#list-public-events>`_\n        '
        return PaginatedList(github.Event.Event, self.__requester, '/events', None)

    def get_user(self, login: Opt[str]=NotSet) -> NamedUser | AuthenticatedUser:
        if False:
            return 10
        '\n        :calls: `GET /users/{user} <https://docs.github.com/en/rest/reference/users>`_ or `GET /user <https://docs.github.com/en/rest/reference/users>`_\n        '
        assert login is NotSet or isinstance(login, str), login
        if login is NotSet:
            return github.AuthenticatedUser.AuthenticatedUser(self.__requester, {}, {'url': '/user'}, completed=False)
        else:
            (headers, data) = self.__requester.requestJsonAndCheck('GET', f'/users/{login}')
            return github.NamedUser.NamedUser(self.__requester, headers, data, completed=True)

    def get_user_by_id(self, user_id: int) -> NamedUser:
        if False:
            for i in range(10):
                print('nop')
        '\n        :calls: `GET /user/{id} <https://docs.github.com/en/rest/reference/users>`_\n        :param user_id: int\n        :rtype: :class:`github.NamedUser.NamedUser`\n        '
        assert isinstance(user_id, int), user_id
        (headers, data) = self.__requester.requestJsonAndCheck('GET', f'/user/{user_id}')
        return github.NamedUser.NamedUser(self.__requester, headers, data, completed=True)

    def get_users(self, since: Opt[int]=NotSet) -> PaginatedList[NamedUser]:
        if False:
            print('Hello World!')
        '\n        :calls: `GET /users <https://docs.github.com/en/rest/reference/users>`_\n        '
        assert since is NotSet or isinstance(since, int), since
        url_parameters = dict()
        if since is not NotSet:
            url_parameters['since'] = since
        return PaginatedList(github.NamedUser.NamedUser, self.__requester, '/users', url_parameters)

    def get_organization(self, login: str) -> Organization:
        if False:
            while True:
                i = 10
        '\n        :calls: `GET /orgs/{org} <https://docs.github.com/en/rest/reference/orgs>`_\n        '
        assert isinstance(login, str), login
        (headers, data) = self.__requester.requestJsonAndCheck('GET', f'/orgs/{login}')
        return github.Organization.Organization(self.__requester, headers, data, completed=True)

    def get_organizations(self, since: Opt[int]=NotSet) -> PaginatedList[Organization]:
        if False:
            i = 10
            return i + 15
        '\n        :calls: `GET /organizations <https://docs.github.com/en/rest/reference/orgs#list-organizations>`_\n        '
        assert since is NotSet or isinstance(since, int), since
        url_parameters = dict()
        if since is not NotSet:
            url_parameters['since'] = since
        return PaginatedList(github.Organization.Organization, self.__requester, '/organizations', url_parameters)

    def get_enterprise(self, enterprise: str) -> github.Enterprise.Enterprise:
        if False:
            while True:
                i = 10
        '\n        :calls: `GET /enterprises/{enterprise} <https://docs.github.com/en/enterprise-cloud@latest/rest/enterprise-admin>`_\n        :param enterprise: string\n        :rtype: :class:`Enterprise`\n        '
        assert isinstance(enterprise, str), enterprise
        return github.Enterprise.Enterprise(self.__requester, enterprise)

    def get_repo(self, full_name_or_id: int | str, lazy: bool=False) -> Repository:
        if False:
            print('Hello World!')
        '\n        :calls: `GET /repos/{owner}/{repo} <https://docs.github.com/en/rest/reference/repos>`_ or `GET /repositories/{id} <https://docs.github.com/en/rest/reference/repos>`_\n        '
        assert isinstance(full_name_or_id, (str, int)), full_name_or_id
        url_base = '/repositories/' if isinstance(full_name_or_id, int) else '/repos/'
        url = f'{url_base}{full_name_or_id}'
        if lazy:
            return github.Repository.Repository(self.__requester, {}, {'url': url}, completed=False)
        (headers, data) = self.__requester.requestJsonAndCheck('GET', url)
        return github.Repository.Repository(self.__requester, headers, data, completed=True)

    def get_repos(self, since: Opt[int]=NotSet, visibility: Opt[str]=NotSet) -> PaginatedList[Repository]:
        if False:
            print('Hello World!')
        "\n        :calls: `GET /repositories <https://docs.github.com/en/rest/reference/repos#list-public-repositories>`_\n        :param since: integer\n        :param visibility: string ('all','public')\n        "
        assert since is NotSet or isinstance(since, int), since
        url_parameters: dict[str, Any] = {}
        if since is not NotSet:
            url_parameters['since'] = since
        if visibility is not NotSet:
            assert visibility in ('public', 'all'), visibility
            url_parameters['visibility'] = visibility
        return PaginatedList(github.Repository.Repository, self.__requester, '/repositories', url_parameters)

    def get_project(self, id: int) -> Project:
        if False:
            while True:
                i = 10
        '\n        :calls: `GET /projects/{project_id} <https://docs.github.com/en/rest/reference/projects#get-a-project>`_\n        '
        (headers, data) = self.__requester.requestJsonAndCheck('GET', f'/projects/{id:d}', headers={'Accept': Consts.mediaTypeProjectsPreview})
        return github.Project.Project(self.__requester, headers, data, completed=True)

    def get_project_column(self, id: int) -> ProjectColumn:
        if False:
            print('Hello World!')
        '\n        :calls: `GET /projects/columns/{column_id} <https://docs.github.com/en/rest/reference/projects#get-a-project-column>`_\n        '
        (headers, data) = self.__requester.requestJsonAndCheck('GET', '/projects/columns/%d' % id, headers={'Accept': Consts.mediaTypeProjectsPreview})
        return github.ProjectColumn.ProjectColumn(self.__requester, headers, data, completed=True)

    def get_gist(self, id: str) -> Gist:
        if False:
            return 10
        '\n        :calls: `GET /gists/{id} <https://docs.github.com/en/rest/reference/gists>`_\n        '
        assert isinstance(id, str), id
        (headers, data) = self.__requester.requestJsonAndCheck('GET', f'/gists/{id}')
        return github.Gist.Gist(self.__requester, headers, data, completed=True)

    def get_gists(self, since: Opt[datetime]=NotSet) -> PaginatedList[Gist]:
        if False:
            return 10
        '\n        :calls: `GET /gists/public <https://docs.github.com/en/rest/reference/gists>`_\n        '
        assert since is NotSet or isinstance(since, datetime), since
        url_parameters = dict()
        if is_defined(since):
            url_parameters['since'] = since.strftime('%Y-%m-%dT%H:%M:%SZ')
        return PaginatedList(github.Gist.Gist, self.__requester, '/gists/public', url_parameters)

    def search_repositories(self, query: str, sort: Opt[str]=NotSet, order: Opt[str]=NotSet, **qualifiers: Any) -> PaginatedList[Repository]:
        if False:
            i = 10
            return i + 15
        "\n        :calls: `GET /search/repositories <https://docs.github.com/en/rest/reference/search>`_\n        :param query: string\n        :param sort: string ('stars', 'forks', 'updated')\n        :param order: string ('asc', 'desc')\n        :param qualifiers: keyword dict query qualifiers\n        "
        assert isinstance(query, str), query
        url_parameters = dict()
        if sort is not NotSet:
            assert sort in ('stars', 'forks', 'updated'), sort
            url_parameters['sort'] = sort
        if order is not NotSet:
            assert order in ('asc', 'desc'), order
            url_parameters['order'] = order
        query_chunks = []
        if query:
            query_chunks.append(query)
        for (qualifier, value) in qualifiers.items():
            query_chunks.append(f'{qualifier}:{value}')
        url_parameters['q'] = ' '.join(query_chunks)
        assert url_parameters['q'], 'need at least one qualifier'
        return PaginatedList(github.Repository.Repository, self.__requester, '/search/repositories', url_parameters)

    def search_users(self, query: str, sort: Opt[str]=NotSet, order: Opt[str]=NotSet, **qualifiers: Any) -> PaginatedList[NamedUser]:
        if False:
            i = 10
            return i + 15
        "\n        :calls: `GET /search/users <https://docs.github.com/en/rest/reference/search>`_\n        :param query: string\n        :param sort: string ('followers', 'repositories', 'joined')\n        :param order: string ('asc', 'desc')\n        :param qualifiers: keyword dict query qualifiers\n        :rtype: :class:`PaginatedList` of :class:`github.NamedUser.NamedUser`\n        "
        assert isinstance(query, str), query
        url_parameters = dict()
        if sort is not NotSet:
            assert sort in ('followers', 'repositories', 'joined'), sort
            url_parameters['sort'] = sort
        if order is not NotSet:
            assert order in ('asc', 'desc'), order
            url_parameters['order'] = order
        query_chunks = []
        if query:
            query_chunks.append(query)
        for (qualifier, value) in qualifiers.items():
            query_chunks.append(f'{qualifier}:{value}')
        url_parameters['q'] = ' '.join(query_chunks)
        assert url_parameters['q'], 'need at least one qualifier'
        return PaginatedList(github.NamedUser.NamedUser, self.__requester, '/search/users', url_parameters)

    def search_issues(self, query: str, sort: Opt[str]=NotSet, order: Opt[str]=NotSet, **qualifiers: Any) -> PaginatedList[Issue]:
        if False:
            i = 10
            return i + 15
        "\n        :calls: `GET /search/issues <https://docs.github.com/en/rest/reference/search>`_\n        :param query: string\n        :param sort: string ('comments', 'created', 'updated')\n        :param order: string ('asc', 'desc')\n        :param qualifiers: keyword dict query qualifiers\n        :rtype: :class:`PaginatedList` of :class:`github.Issue.Issue`\n        "
        assert isinstance(query, str), query
        url_parameters = dict()
        if sort is not NotSet:
            assert sort in ('comments', 'created', 'updated'), sort
            url_parameters['sort'] = sort
        if order is not NotSet:
            assert order in ('asc', 'desc'), order
            url_parameters['order'] = order
        query_chunks = []
        if query:
            query_chunks.append(query)
        for (qualifier, value) in qualifiers.items():
            query_chunks.append(f'{qualifier}:{value}')
        url_parameters['q'] = ' '.join(query_chunks)
        assert url_parameters['q'], 'need at least one qualifier'
        return PaginatedList(github.Issue.Issue, self.__requester, '/search/issues', url_parameters)

    def search_code(self, query: str, sort: Opt[str]=NotSet, order: Opt[str]=NotSet, highlight: bool=False, **qualifiers: Any) -> PaginatedList[ContentFile]:
        if False:
            while True:
                i = 10
        "\n        :calls: `GET /search/code <https://docs.github.com/en/rest/reference/search>`_\n        :param query: string\n        :param sort: string ('indexed')\n        :param order: string ('asc', 'desc')\n        :param highlight: boolean (True, False)\n        :param qualifiers: keyword dict query qualifiers\n        :rtype: :class:`PaginatedList` of :class:`github.ContentFile.ContentFile`\n        "
        assert isinstance(query, str), query
        url_parameters = dict()
        if sort is not NotSet:
            assert sort in ('indexed',), sort
            url_parameters['sort'] = sort
        if order is not NotSet:
            assert order in ('asc', 'desc'), order
            url_parameters['order'] = order
        query_chunks = []
        if query:
            query_chunks.append(query)
        for (qualifier, value) in qualifiers.items():
            query_chunks.append(f'{qualifier}:{value}')
        url_parameters['q'] = ' '.join(query_chunks)
        assert url_parameters['q'], 'need at least one qualifier'
        headers = {'Accept': Consts.highLightSearchPreview} if highlight else None
        return PaginatedList(github.ContentFile.ContentFile, self.__requester, '/search/code', url_parameters, headers=headers)

    def search_commits(self, query: str, sort: Opt[str]=NotSet, order: Opt[str]=NotSet, **qualifiers: Any) -> PaginatedList[Commit]:
        if False:
            while True:
                i = 10
        "\n        :calls: `GET /search/commits <https://docs.github.com/en/rest/reference/search>`_\n        :param query: string\n        :param sort: string ('author-date', 'committer-date')\n        :param order: string ('asc', 'desc')\n        :param qualifiers: keyword dict query qualifiers\n        :rtype: :class:`PaginatedList` of :class:`github.Commit.Commit`\n        "
        assert isinstance(query, str), query
        url_parameters = dict()
        if sort is not NotSet:
            assert sort in ('author-date', 'committer-date'), sort
            url_parameters['sort'] = sort
        if order is not NotSet:
            assert order in ('asc', 'desc'), order
            url_parameters['order'] = order
        query_chunks = []
        if query:
            query_chunks.append(query)
        for (qualifier, value) in qualifiers.items():
            query_chunks.append(f'{qualifier}:{value}')
        url_parameters['q'] = ' '.join(query_chunks)
        assert url_parameters['q'], 'need at least one qualifier'
        return PaginatedList(github.Commit.Commit, self.__requester, '/search/commits', url_parameters, headers={'Accept': Consts.mediaTypeCommitSearchPreview})

    def search_topics(self, query: str, **qualifiers: Any) -> PaginatedList[Topic]:
        if False:
            while True:
                i = 10
        '\n        :calls: `GET /search/topics <https://docs.github.com/en/rest/reference/search>`_\n        :param query: string\n        :param qualifiers: keyword dict query qualifiers\n        :rtype: :class:`PaginatedList` of :class:`github.Topic.Topic`\n        '
        assert isinstance(query, str), query
        url_parameters = dict()
        query_chunks = []
        if query:
            query_chunks.append(query)
        for (qualifier, value) in qualifiers.items():
            query_chunks.append(f'{qualifier}:{value}')
        url_parameters['q'] = ' '.join(query_chunks)
        assert url_parameters['q'], 'need at least one qualifier'
        return PaginatedList(github.Topic.Topic, self.__requester, '/search/topics', url_parameters, headers={'Accept': Consts.mediaTypeTopicsPreview})

    def render_markdown(self, text: str, context: Opt[Repository]=NotSet) -> str:
        if False:
            while True:
                i = 10
        '\n        :calls: `POST /markdown <https://docs.github.com/en/rest/reference/markdown>`_\n        :param text: string\n        :param context: :class:`github.Repository.Repository`\n        :rtype: string\n        '
        assert isinstance(text, str), text
        assert context is NotSet or isinstance(context, github.Repository.Repository), context
        post_parameters = {'text': text}
        if is_defined(context):
            post_parameters['mode'] = 'gfm'
            post_parameters['context'] = context._identity
        (status, headers, data) = self.__requester.requestJson('POST', '/markdown', input=post_parameters)
        return data

    def get_hook(self, name: str) -> HookDescription:
        if False:
            for i in range(10):
                print('nop')
        '\n        :calls: `GET /hooks/{name} <https://docs.github.com/en/rest/reference/repos#webhooks>`_\n        '
        assert isinstance(name, str), name
        (headers, attributes) = self.__requester.requestJsonAndCheck('GET', f'/hooks/{name}')
        return HookDescription(self.__requester, headers, attributes, completed=True)

    def get_hooks(self) -> list[HookDescription]:
        if False:
            for i in range(10):
                print('nop')
        '\n        :calls: `GET /hooks <https://docs.github.com/en/rest/reference/repos#webhooks>`_\n        :rtype: list of :class:`github.HookDescription.HookDescription`\n        '
        (headers, data) = self.__requester.requestJsonAndCheck('GET', '/hooks')
        return [HookDescription(self.__requester, headers, attributes, completed=True) for attributes in data]

    def get_hook_delivery(self, hook_id: int, delivery_id: int) -> HookDelivery:
        if False:
            i = 10
            return i + 15
        '\n        :calls: `GET /hooks/{hook_id}/deliveries/{delivery_id} <https://docs.github.com/en/rest/reference/repos#webhooks>`_\n        :param hook_id: integer\n        :param delivery_id: integer\n        :rtype: :class:`HookDelivery`\n        '
        assert isinstance(hook_id, int), hook_id
        assert isinstance(delivery_id, int), delivery_id
        (headers, attributes) = self.__requester.requestJsonAndCheck('GET', f'/hooks/{hook_id}/deliveries/{delivery_id}')
        return HookDelivery(self.__requester, headers, attributes, completed=True)

    def get_hook_deliveries(self, hook_id: int) -> list[HookDeliverySummary]:
        if False:
            for i in range(10):
                print('nop')
        '\n        :calls: `GET /hooks/{hook_id}/deliveries <https://docs.github.com/en/rest/reference/repos#webhooks>`_\n        :param hook_id: integer\n        :rtype: list of :class:`HookDeliverySummary`\n        '
        assert isinstance(hook_id, int), hook_id
        (headers, data) = self.__requester.requestJsonAndCheck('GET', f'/hooks/{hook_id}/deliveries')
        return [HookDeliverySummary(self.__requester, headers, attributes, completed=True) for attributes in data]

    def get_gitignore_templates(self) -> list[str]:
        if False:
            while True:
                i = 10
        '\n        :calls: `GET /gitignore/templates <https://docs.github.com/en/rest/reference/gitignore>`_\n        '
        (headers, data) = self.__requester.requestJsonAndCheck('GET', '/gitignore/templates')
        return data

    def get_gitignore_template(self, name: str) -> GitignoreTemplate:
        if False:
            i = 10
            return i + 15
        '\n        :calls: `GET /gitignore/templates/{name} <https://docs.github.com/en/rest/reference/gitignore>`_\n        '
        assert isinstance(name, str), name
        (headers, attributes) = self.__requester.requestJsonAndCheck('GET', f'/gitignore/templates/{name}')
        return github.GitignoreTemplate.GitignoreTemplate(self.__requester, headers, attributes, completed=True)

    def get_emojis(self) -> dict[str, str]:
        if False:
            while True:
                i = 10
        '\n        :calls: `GET /emojis <https://docs.github.com/en/rest/reference/emojis>`_\n        :rtype: dictionary of type => url for emoji`\n        '
        (headers, attributes) = self.__requester.requestJsonAndCheck('GET', '/emojis')
        return attributes

    def create_from_raw_data(self, klass: type[TGithubObject], raw_data: dict[str, Any], headers: dict[str, str | int] | None=None) -> TGithubObject:
        if False:
            print('Hello World!')
        '\n        Creates an object from raw_data previously obtained by :attr:`GithubObject.raw_data`,\n        and optionally headers previously obtained by :attr:`GithubObject.raw_headers`.\n\n        :param klass: the class of the object to create\n        :param raw_data: dict\n        :param headers: dict\n        :rtype: instance of class ``klass``\n        '
        if headers is None:
            headers = {}
        return klass(self.__requester, headers, raw_data, completed=True)

    def dump(self, obj: GithubObject, file: BinaryIO, protocol: int=0) -> None:
        if False:
            print('Hello World!')
        "\n        Dumps (pickles) a PyGithub object to a file-like object.\n        Some effort is made to not pickle sensitive information like the Github credentials used in the :class:`Github` instance.\n        But NO EFFORT is made to remove sensitive information from the object's attributes.\n\n        :param obj: the object to pickle\n        :param file: the file-like object to pickle to\n        :param protocol: the `pickling protocol <https://python.readthedocs.io/en/latest/library/pickle.html#data-stream-format>`_\n        "
        pickle.dump((obj.__class__, obj.raw_data, obj.raw_headers), file, protocol)

    def load(self, f: BinaryIO) -> Any:
        if False:
            return 10
        '\n        Loads (unpickles) a PyGithub object from a file-like object.\n\n        :param f: the file-like object to unpickle from\n        :return: the unpickled object\n        '
        return self.create_from_raw_data(*pickle.load(f))

    def get_oauth_application(self, client_id: str, client_secret: str) -> ApplicationOAuth:
        if False:
            i = 10
            return i + 15
        return github.ApplicationOAuth.ApplicationOAuth(self.__requester, headers={}, attributes={'client_id': client_id, 'client_secret': client_secret}, completed=False)

    def get_app(self, slug: Opt[str]=NotSet) -> GithubApp:
        if False:
            for i in range(10):
                print('nop')
        '\n        :calls: `GET /apps/{slug} <https://docs.github.com/en/rest/reference/apps>`_ or `GET /app <https://docs.github.com/en/rest/reference/apps>`_\n        '
        assert slug is NotSet or isinstance(slug, str), slug
        if slug is NotSet:
            warnings.warn('Argument slug is mandatory, calling this method without the slug argument is deprecated, please use github.GithubIntegration(auth=github.Auth.AppAuth(...)).get_app() instead', category=DeprecationWarning)
            return GithubIntegration(**self.__requester.kwargs).get_app()
        else:
            return github.GithubApp.GithubApp(self.__requester, {}, {'url': f'/apps/{slug}'}, completed=False)