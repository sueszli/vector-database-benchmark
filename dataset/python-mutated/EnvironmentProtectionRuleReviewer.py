from __future__ import annotations
from typing import Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class EnvironmentProtectionRuleReviewer(NonCompletableGithubObject):
    """
    This class represents a reviewer for an EnvironmentProtectionRule. The reference can be found here https://docs.github.com/en/rest/reference/deployments#environments
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._type: Attribute[str] = NotSet
        self._reviewer: Attribute[github.NamedUser.NamedUser | github.Team.Team] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'type': self._type.value})

    @property
    def type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._type.value

    @property
    def reviewer(self) -> github.NamedUser.NamedUser | github.Team.Team:
        if False:
            print('Hello World!')
        return self._reviewer.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'reviewer' in attributes:
            assert self._type.value in ('User', 'Team')
            if self._type.value == 'User':
                self._reviewer = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['reviewer'])
            elif self._type.value == 'Team':
                self._reviewer = self._makeClassAttribute(github.Team.Team, attributes['reviewer'])

class ReviewerParams:
    """
    This class presents reviewers as can be configured for an Environment.
    """

    def __init__(self, type_: str, id_: int):
        if False:
            i = 10
            return i + 15
        assert isinstance(type_, str) and type_ in ('User', 'Team')
        assert isinstance(id_, int)
        self.type = type_
        self.id = id_

    def _asdict(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'type': self.type, 'id': self.id}