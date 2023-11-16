from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class StatsParticipation(NonCompletableGithubObject):
    """
    This class represents StatsParticipations. The reference can be found here https://docs.github.com/en/rest/reference/repos#get-the-weekly-commit-count
    """

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._all: Attribute[list[int]] = NotSet
        self._owner: Attribute[list[int]] = NotSet

    @property
    def all(self) -> list[int]:
        if False:
            while True:
                i = 10
        return self._all.value

    @property
    def owner(self) -> list[int]:
        if False:
            i = 10
            return i + 15
        return self._owner.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'all' in attributes:
            self._all = self._makeListOfIntsAttribute(attributes['all'])
        if 'owner' in attributes:
            self._owner = self._makeListOfIntsAttribute(attributes['owner'])