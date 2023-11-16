from datetime import datetime
from typing import Any, Dict
import github.GithubObject
from github.GithubObject import Attribute

class StatsCommitActivity(github.GithubObject.NonCompletableGithubObject):
    """
    This class represents StatsCommitActivities. The reference can be found here https://docs.github.com/en/rest/reference/repos#get-the-last-year-of-commit-activity
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._week: Attribute[datetime] = github.GithubObject.NotSet
        self._total: Attribute[int] = github.GithubObject.NotSet
        self._days: Attribute[int] = github.GithubObject.NotSet

    @property
    def week(self) -> datetime:
        if False:
            print('Hello World!')
        return self._week.value

    @property
    def total(self) -> int:
        if False:
            return 10
        return self._total.value

    @property
    def days(self) -> int:
        if False:
            while True:
                i = 10
        return self._days.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'week' in attributes:
            self._week = self._makeTimestampAttribute(attributes['week'])
        if 'total' in attributes:
            self._total = self._makeIntAttribute(attributes['total'])
        if 'days' in attributes:
            self._days = self._makeListOfIntsAttribute(attributes['days'])