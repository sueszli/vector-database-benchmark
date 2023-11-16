from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.Repository
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
if TYPE_CHECKING:
    from github.Repository import Repository

class RepositoryPreferences(NonCompletableGithubObject):
    """
    This class represents repository preferences.
    The reference can be found here https://docs.github.com/en/free-pro-team@latest/rest/reference/checks#update-repository-preferences-for-check-suites
    """

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._preferences: Attribute[dict[str, list[dict[str, bool | int]]]] = NotSet
        self._repository: Attribute[Repository] = NotSet

    @property
    def preferences(self) -> dict[str, list[dict[str, bool | int]]]:
        if False:
            for i in range(10):
                print('nop')
        return self._preferences.value

    @property
    def repository(self) -> Repository:
        if False:
            for i in range(10):
                print('nop')
        return self._repository.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        if 'preferences' in attributes:
            self._preferences = self._makeDictAttribute(attributes['preferences'])
        if 'repository' in attributes:
            self._repository = self._makeClassAttribute(github.Repository.Repository, attributes['repository'])