from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, NotSet
from github.PaginatedList import PaginatedList
from github.Repository import Repository
from github.Variable import Variable

class OrganizationVariable(Variable):
    """
    This class represents a org level GitHub variable. The reference can be found here https://docs.github.com/en/rest/actions/variables
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._name: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._visibility: Attribute[str] = NotSet
        self._selected_repositories: Attribute[PaginatedList[Repository]] = NotSet
        self._selected_repositories_url: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet

    @property
    def visibility(self) -> str:
        if False:
            print('Hello World!')
        '\n        :type: string\n        '
        self._completeIfNotSet(self._visibility)
        return self._visibility.value

    @property
    def selected_repositories(self) -> PaginatedList[Repository]:
        if False:
            for i in range(10):
                print('nop')
        return PaginatedList(Repository, self._requester, self._selected_repositories_url.value, None, list_item='repositories')

    def edit(self, value: str, visibility: str='all') -> bool:
        if False:
            return 10
        '\n        :calls: `PATCH /orgs/{org}/actions/variables/{variable_name} <https://docs.github.com/en/rest/reference/actions/variables#update-an-organization-variable>`_\n        :param variable_name: string\n        :param value: string\n        :param visibility: string\n        :rtype: bool\n        '
        assert isinstance(value, str), value
        assert isinstance(visibility, str), visibility
        patch_parameters: Dict[str, Any] = {'name': self.name, 'value': value, 'visibility': visibility}
        (status, _, _) = self._requester.requestJson('PATCH', f'{self.url}/actions/variables/{self.name}', input=patch_parameters)
        return status == 204

    def add_repo(self, repo: Repository) -> bool:
        if False:
            return 10
        "\n        :calls: 'PUT {org_url}/actions/variables/{variable_name} <https://docs.github.com/en/rest/actions/variables#add-selected-repository-to-an-organization-secret>`_\n        :param repo: github.Repository.Repository\n        :rtype: bool\n        "
        if self.visibility != 'selected':
            return False
        self._requester.requestJsonAndCheck('PUT', f'{self._selected_repositories_url.value}/{repo.id}')
        return True

    def remove_repo(self, repo: Repository) -> bool:
        if False:
            while True:
                i = 10
        "\n        :calls: 'DELETE {org_url}/actions/variables/{variable_name} <https://docs.github.com/en/rest/actions/variables#add-selected-repository-to-an-organization-secret>`_\n        :param repo: github.Repository.Repository\n        :rtype: bool\n        "
        if self.visibility != 'selected':
            return False
        self._requester.requestJsonAndCheck('DELETE', f'{self._selected_repositories_url.value}/{repo.id}')
        return True

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'visibility' in attributes:
            self._visibility = self._makeStringAttribute(attributes['visibility'])
        if 'selected_repositories_url' in attributes:
            self._selected_repositories_url = self._makeStringAttribute(attributes['selected_repositories_url'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])