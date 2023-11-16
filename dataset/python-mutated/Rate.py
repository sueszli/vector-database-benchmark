from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class Rate(NonCompletableGithubObject):
    """
    This class represents Rates. The reference can be found here https://docs.github.com/en/rest/reference/rate-limit
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._limit: Attribute[int] = NotSet
        self._remaining: Attribute[int] = NotSet
        self._reset: Attribute[datetime] = NotSet
        self._used: Attribute[int] = NotSet

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return self.get__repr__({'limit': self._limit.value, 'remaining': self._remaining.value, 'reset': self._reset.value})

    @property
    def limit(self) -> int:
        if False:
            return 10
        return self._limit.value

    @property
    def remaining(self) -> int:
        if False:
            return 10
        return self._remaining.value

    @property
    def reset(self) -> datetime:
        if False:
            return 10
        return self._reset.value

    @property
    def used(self) -> int:
        if False:
            while True:
                i = 10
        return self._used.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'limit' in attributes:
            self._limit = self._makeIntAttribute(attributes['limit'])
        if 'remaining' in attributes:
            self._remaining = self._makeIntAttribute(attributes['remaining'])
        if 'reset' in attributes:
            self._reset = self._makeTimestampAttribute(attributes['reset'])
        if 'used' in attributes:
            self._used = self._makeIntAttribute(attributes['used'])