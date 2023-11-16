from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class GistFile(NonCompletableGithubObject):
    """
    This class represents GistFiles
    """

    def _initAttributes(self) -> None:
        if False:
            return 10
        self._content: Attribute[str] = NotSet
        self._filename: Attribute[str] = NotSet
        self._language: Attribute[str] = NotSet
        self._raw_url: Attribute[str] = NotSet
        self._size: Attribute[int] = NotSet
        self._type: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'filename': self._filename.value})

    @property
    def content(self) -> str:
        if False:
            return 10
        return self._content.value

    @property
    def filename(self) -> str:
        if False:
            return 10
        return self._filename.value

    @property
    def language(self) -> str:
        if False:
            while True:
                i = 10
        return self._language.value

    @property
    def raw_url(self) -> str:
        if False:
            print('Hello World!')
        return self._raw_url.value

    @property
    def size(self) -> int:
        if False:
            while True:
                i = 10
        return self._size.value

    @property
    def type(self) -> str:
        if False:
            while True:
                i = 10
        return self._type.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'content' in attributes:
            self._content = self._makeStringAttribute(attributes['content'])
        if 'filename' in attributes:
            self._filename = self._makeStringAttribute(attributes['filename'])
        if 'language' in attributes:
            self._language = self._makeStringAttribute(attributes['language'])
        if 'raw_url' in attributes:
            self._raw_url = self._makeStringAttribute(attributes['raw_url'])
        if 'size' in attributes:
            self._size = self._makeIntAttribute(attributes['size'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])