from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

class AuthorizationApplication(CompletableGithubObject):
    """
    This class represents AuthorizationApplications
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._name: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return self.get__repr__({'name': self._name.value})

    @property
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        self._completeIfNotSet(self._name)
        return self._name.value

    @property
    def url(self) -> str:
        if False:
            print('Hello World!')
        self._completeIfNotSet(self._url)
        return self._url.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])