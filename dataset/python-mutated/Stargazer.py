from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
if TYPE_CHECKING:
    from github.NamedUser import NamedUser

class Stargazer(NonCompletableGithubObject):
    """
    This class represents Stargazers. The reference can be found here https://docs.github.com/en/rest/reference/activity#starring
    """

    def _initAttributes(self) -> None:
        if False:
            return 10
        self._starred_at: Attribute[datetime] = NotSet
        self._user: Attribute[NamedUser] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'user': self._user.value._login.value})

    @property
    def starred_at(self) -> datetime:
        if False:
            while True:
                i = 10
        return self._starred_at.value

    @property
    def user(self) -> NamedUser:
        if False:
            i = 10
            return i + 15
        return self._user.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'starred_at' in attributes:
            self._starred_at = self._makeDatetimeAttribute(attributes['starred_at'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])