from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class CheckRunOutput(NonCompletableGithubObject):
    """This class represents the output of check run."""

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._annotations_count: Attribute[int] = NotSet
        self._annotations_url: Attribute[str] = NotSet
        self._summary: Attribute[str] = NotSet
        self._text: Attribute[str] = NotSet
        self._title: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return self.get__repr__({'title': self._title.value})

    @property
    def annotations_count(self) -> int:
        if False:
            while True:
                i = 10
        return self._annotations_count.value

    @property
    def annotations_url(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._annotations_url.value

    @property
    def summary(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._summary.value

    @property
    def text(self) -> str:
        if False:
            while True:
                i = 10
        return self._text.value

    @property
    def title(self) -> str:
        if False:
            print('Hello World!')
        return self._title.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'annotations_count' in attributes:
            self._annotations_count = self._makeIntAttribute(attributes['annotations_count'])
        if 'annotations_url' in attributes:
            self._annotations_url = self._makeStringAttribute(attributes['annotations_url'])
        if 'summary' in attributes:
            self._summary = self._makeStringAttribute(attributes['summary'])
        if 'text' in attributes:
            self._text = self._makeStringAttribute(attributes['text'])
        if 'title' in attributes:
            self._title = self._makeStringAttribute(attributes['title'])