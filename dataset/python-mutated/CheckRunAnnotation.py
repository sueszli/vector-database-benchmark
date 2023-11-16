from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class CheckRunAnnotation(NonCompletableGithubObject):
    """
    This class represents check run annotations.
    The reference can be found here: https://docs.github.com/en/rest/reference/checks#list-check-run-annotations
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._annotation_level: Attribute[str] = NotSet
        self._end_column: Attribute[int] = NotSet
        self._end_line: Attribute[int] = NotSet
        self._message: Attribute[str] = NotSet
        self._path: Attribute[str] = NotSet
        self._raw_details: Attribute[str] = NotSet
        self._start_column: Attribute[int] = NotSet
        self._start_line: Attribute[int] = NotSet
        self._title: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return self.get__repr__({'title': self._title.value})

    @property
    def annotation_level(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._annotation_level.value

    @property
    def end_column(self) -> int:
        if False:
            print('Hello World!')
        return self._end_column.value

    @property
    def end_line(self) -> int:
        if False:
            return 10
        return self._end_line.value

    @property
    def message(self) -> str:
        if False:
            print('Hello World!')
        return self._message.value

    @property
    def path(self) -> str:
        if False:
            return 10
        return self._path.value

    @property
    def raw_details(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._raw_details.value

    @property
    def start_column(self) -> int:
        if False:
            return 10
        return self._start_column.value

    @property
    def start_line(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._start_line.value

    @property
    def title(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._title.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'annotation_level' in attributes:
            self._annotation_level = self._makeStringAttribute(attributes['annotation_level'])
        if 'end_column' in attributes:
            self._end_column = self._makeIntAttribute(attributes['end_column'])
        if 'end_line' in attributes:
            self._end_line = self._makeIntAttribute(attributes['end_line'])
        if 'message' in attributes:
            self._message = self._makeStringAttribute(attributes['message'])
        if 'path' in attributes:
            self._path = self._makeStringAttribute(attributes['path'])
        if 'raw_details' in attributes:
            self._raw_details = self._makeStringAttribute(attributes['raw_details'])
        if 'start_column' in attributes:
            self._start_column = self._makeIntAttribute(attributes['start_column'])
        if 'start_line' in attributes:
            self._start_line = self._makeIntAttribute(attributes['start_line'])
        if 'title' in attributes:
            self._title = self._makeStringAttribute(attributes['title'])