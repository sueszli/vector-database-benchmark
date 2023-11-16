from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Organization
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
if TYPE_CHECKING:
    from github.NamedUser import NamedUser
    from github.Organization import Organization

class Membership(CompletableGithubObject):
    """
    This class represents Membership of an organization. The reference can be found here https://docs.github.com/en/rest/reference/orgs
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._url: Attribute[str] = NotSet
        self._state: Attribute[str] = NotSet
        self._role: Attribute[str] = NotSet
        self._organization_url: Attribute[str] = NotSet
        self._organization: Attribute[Organization] = NotSet
        self._user: Attribute[NamedUser] = NotSet

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return self.get__repr__({'url': self._url.value})

    @property
    def url(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._url)
        return self._url.value

    @property
    def state(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        self._completeIfNotSet(self._state)
        return self._state.value

    @property
    def role(self) -> str:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._role)
        return self._role.value

    @property
    def organization_url(self) -> str:
        if False:
            return 10
        self._completeIfNotSet(self._organization_url)
        return self._organization_url.value

    @property
    def organization(self) -> Organization:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._organization)
        return self._organization.value

    @property
    def user(self) -> NamedUser:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._user)
        return self._user.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'state' in attributes:
            self._state = self._makeStringAttribute(attributes['state'])
        if 'role' in attributes:
            self._role = self._makeStringAttribute(attributes['role'])
        if 'organization_url' in attributes:
            self._organization_url = self._makeStringAttribute(attributes['organization_url'])
        if 'organization' in attributes:
            self._organization = self._makeClassAttribute(github.Organization.Organization, attributes['organization'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])