from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.EnvironmentProtectionRuleReviewer
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
if TYPE_CHECKING:
    from github.EnvironmentProtectionRuleReviewer import EnvironmentProtectionRuleReviewer

class EnvironmentProtectionRule(NonCompletableGithubObject):
    """
    This class represents a protection rule for an environment. The reference can be found here https://docs.github.com/en/rest/reference/deployments#environments
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._id: Attribute[int] = NotSet
        self._node_id: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet
        self._reviewers: Attribute[list[EnvironmentProtectionRuleReviewer]] = NotSet
        self._wait_timer: Attribute[int] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'id': self._id.value})

    @property
    def id(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._id.value

    @property
    def node_id(self) -> str:
        if False:
            return 10
        return self._node_id.value

    @property
    def type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._type.value

    @property
    def reviewers(self) -> list[EnvironmentProtectionRuleReviewer]:
        if False:
            print('Hello World!')
        return self._reviewers.value

    @property
    def wait_timer(self) -> int:
        if False:
            while True:
                i = 10
        return self._wait_timer.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            return 10
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'node_id' in attributes:
            self._node_id = self._makeStringAttribute(attributes['node_id'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'reviewers' in attributes:
            self._reviewers = self._makeListOfClassesAttribute(github.EnvironmentProtectionRuleReviewer.EnvironmentProtectionRuleReviewer, attributes['reviewers'])
        if 'wait_timer' in attributes:
            self._wait_timer = self._makeIntAttribute(attributes['wait_timer'])