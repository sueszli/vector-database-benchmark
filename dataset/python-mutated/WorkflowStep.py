from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

class WorkflowStep(CompletableGithubObject):
    """
    This class represents steps in a Workflow Job. The reference can be found here https://docs.github.com/en/rest/reference/actions#workflow-jobs
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._completed_at: Attribute[datetime] = NotSet
        self._conclusion: Attribute[str] = NotSet
        self._name: Attribute[str] = NotSet
        self._number: Attribute[int] = NotSet
        self._started_at: Attribute[datetime] = NotSet
        self._status: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.get__repr__({'number': self._number.value, 'name': self._name.value})

    @property
    def completed_at(self) -> datetime:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._completed_at)
        return self._completed_at.value

    @property
    def conclusion(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._conclusion)
        return self._conclusion.value

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._name)
        return self._name.value

    @property
    def number(self) -> int:
        if False:
            while True:
                i = 10
        self._completeIfNotSet(self._number)
        return self._number.value

    @property
    def started_at(self) -> datetime:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._started_at)
        return self._started_at.value

    @property
    def status(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._status)
        return self._status.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'completed_at' in attributes:
            self._completed_at = self._makeDatetimeAttribute(attributes['completed_at'])
        if 'conclusion' in attributes:
            self._conclusion = self._makeStringAttribute(attributes['conclusion'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'number' in attributes:
            self._number = self._makeIntAttribute(attributes['number'])
        if 'started_at' in attributes:
            self._started_at = self._makeDatetimeAttribute(attributes['started_at'])
        if 'status' in attributes:
            self._status = self._makeStringAttribute(attributes['status'])