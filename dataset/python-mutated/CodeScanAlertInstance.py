from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.CodeScanAlertInstanceLocation
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
if TYPE_CHECKING:
    from github.CodeScanAlertInstanceLocation import CodeScanAlertInstanceLocation

class CodeScanAlertInstance(NonCompletableGithubObject):
    """
    This class represents code scanning alert instances.
    The reference can be found here https://docs.github.com/en/rest/reference/code-scanning.
    """

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._ref: Attribute[str] = NotSet
        self._analysis_key: Attribute[str] = NotSet
        self._environment: Attribute[str] = NotSet
        self._state: Attribute[str] = NotSet
        self._commit_sha: Attribute[str] = NotSet
        self._message: Attribute[dict[str, Any]] = NotSet
        self._location: Attribute[CodeScanAlertInstanceLocation] = NotSet
        self._classifications: Attribute[list[str]] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'ref': self.ref, 'analysis_key': self.analysis_key})

    @property
    def ref(self) -> str:
        if False:
            return 10
        return self._ref.value

    @property
    def analysis_key(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._analysis_key.value

    @property
    def environment(self) -> str:
        if False:
            while True:
                i = 10
        return self._environment.value

    @property
    def state(self) -> str:
        if False:
            while True:
                i = 10
        return self._state.value

    @property
    def commit_sha(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._commit_sha.value

    @property
    def message(self) -> dict[str, Any]:
        if False:
            return 10
        return self._message.value

    @property
    def location(self) -> CodeScanAlertInstanceLocation:
        if False:
            while True:
                i = 10
        return self._location.value

    @property
    def classifications(self) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._classifications.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'ref' in attributes:
            self._ref = self._makeStringAttribute(attributes['ref'])
        if 'analysis_key' in attributes:
            self._analysis_key = self._makeStringAttribute(attributes['analysis_key'])
        if 'environment' in attributes:
            self._environment = self._makeStringAttribute(attributes['environment'])
        if 'state' in attributes:
            self._state = self._makeStringAttribute(attributes['state'])
        if 'environment' in attributes:
            self._environment = self._makeStringAttribute(attributes['environment'])
        if 'commit_sha' in attributes:
            self._commit_sha = self._makeStringAttribute(attributes['commit_sha'])
        if 'message' in attributes:
            self._message = self._makeDictAttribute(attributes['message'])
        if 'location' in attributes:
            self._location = self._makeClassAttribute(github.CodeScanAlertInstanceLocation.CodeScanAlertInstanceLocation, attributes['location'])
        if 'classifications' in attributes:
            self._classifications = self._makeListOfStringsAttribute(attributes['classifications'])