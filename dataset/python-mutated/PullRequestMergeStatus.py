from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class PullRequestMergeStatus(NonCompletableGithubObject):
    """
    This class represents PullRequestMergeStatuses. The reference can be found here https://docs.github.com/en/rest/reference/pulls#check-if-a-pull-request-has-been-merged
    """

    def _initAttributes(self) -> None:
        if False:
            return 10
        self._merged: Attribute[bool] = NotSet
        self._message: Attribute[str] = NotSet
        self._sha: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.get__repr__({'sha': self._sha.value, 'merged': self._merged.value})

    @property
    def merged(self) -> bool:
        if False:
            return 10
        return self._merged.value

    @property
    def message(self) -> str:
        if False:
            while True:
                i = 10
        return self._message.value

    @property
    def sha(self) -> str:
        if False:
            print('Hello World!')
        return self._sha.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'merged' in attributes:
            self._merged = self._makeBoolAttribute(attributes['merged'])
        if 'message' in attributes:
            self._message = self._makeStringAttribute(attributes['message'])
        if 'sha' in attributes:
            self._sha = self._makeStringAttribute(attributes['sha'])