from __future__ import annotations
from typing import Any
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class RepositoryAdvisoryCreditDetailed(NonCompletableGithubObject):
    """
    This class represents a credit that is assigned to a SecurityAdvisory.
    The reference can be found here https://docs.github.com/en/rest/security-advisories/repository-advisories
    """

    @property
    def state(self) -> str:
        if False:
            print('Hello World!')
        '\n        :type: string\n        '
        return self._state.value

    @property
    def type(self) -> str:
        if False:
            return 10
        '\n        :type: string\n        '
        return self._type.value

    @property
    def user(self) -> github.NamedUser.NamedUser:
        if False:
            print('Hello World!')
        '\n        :type: :class:`github.NamedUser.NamedUser`\n        '
        return self._user.value

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._state: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet
        self._user: Attribute[github.NamedUser.NamedUser] = NotSet

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'state' in attributes:
            self._state = self._makeStringAttribute(attributes['state'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])