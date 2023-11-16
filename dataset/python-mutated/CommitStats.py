from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class CommitStats(NonCompletableGithubObject):
    """
    This class represents CommitStats.
    """

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._total: Attribute[int] = NotSet
        self._deletions: Attribute[int] = NotSet
        self._additions: Attribute[int] = NotSet

    @property
    def additions(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._additions.value

    @property
    def deletions(self) -> int:
        if False:
            while True:
                i = 10
        return self._deletions.value

    @property
    def total(self) -> int:
        if False:
            while True:
                i = 10
        return self._total.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'additions' in attributes:
            self._additions = self._makeIntAttribute(attributes['additions'])
        if 'deletions' in attributes:
            self._deletions = self._makeIntAttribute(attributes['deletions'])
        if 'total' in attributes:
            self._total = self._makeIntAttribute(attributes['total'])