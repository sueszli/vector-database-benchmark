from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class CodeScanTool(NonCompletableGithubObject):
    """
    This class represents code scanning tools.
    The reference can be found here https://docs.github.com/en/rest/reference/code-scanning.
    """

    def _initAttributes(self) -> None:
        if False:
            while True:
                i = 10
        self._name: Attribute[str] = NotSet
        self._version: Attribute[str] = NotSet
        self._guid: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.get__repr__({'guid': self.guid, 'name': self.name, 'version': self.version})

    @property
    def name(self) -> str:
        if False:
            return 10
        return self._name.value

    @property
    def version(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._version.value

    @property
    def guid(self) -> str:
        if False:
            print('Hello World!')
        return self._guid.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'version' in attributes:
            self._version = self._makeStringAttribute(attributes['version'])
        if 'guid' in attributes:
            self._guid = self._makeStringAttribute(attributes['guid'])