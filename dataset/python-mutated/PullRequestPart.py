from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Repository
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
if TYPE_CHECKING:
    from github.NamedUser import NamedUser
    from github.Repository import Repository

class PullRequestPart(NonCompletableGithubObject):
    """
    This class represents PullRequestParts
    """

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._label: Attribute[str] = NotSet
        self._ref: Attribute[str] = NotSet
        self._repo: Attribute[Repository] = NotSet
        self._sha: Attribute[str] = NotSet
        self._user: Attribute[NamedUser] = NotSet

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.get__repr__({'sha': self._sha.value})

    @property
    def label(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._label.value

    @property
    def ref(self) -> str:
        if False:
            return 10
        return self._ref.value

    @property
    def repo(self) -> Repository:
        if False:
            print('Hello World!')
        return self._repo.value

    @property
    def sha(self) -> str:
        if False:
            return 10
        return self._sha.value

    @property
    def user(self) -> NamedUser:
        if False:
            i = 10
            return i + 15
        return self._user.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'label' in attributes:
            self._label = self._makeStringAttribute(attributes['label'])
        if 'ref' in attributes:
            self._ref = self._makeStringAttribute(attributes['ref'])
        if 'repo' in attributes:
            self._repo = self._makeClassAttribute(github.Repository.Repository, attributes['repo'])
        if 'sha' in attributes:
            self._sha = self._makeStringAttribute(attributes['sha'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])