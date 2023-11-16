from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class GitignoreTemplate(NonCompletableGithubObject):
    """
    This class represents GitignoreTemplates. The reference can be found here https://docs.github.com/en/rest/reference/gitignore
    """

    def _initAttributes(self) -> None:
        if False:
            while True:
                i = 10
        self._source: Attribute[str] = NotSet
        self._name: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'name': self._name.value})

    @property
    def source(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._source.value

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        return self._name.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'source' in attributes:
            self._source = self._makeStringAttribute(attributes['source'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])