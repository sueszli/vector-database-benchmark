from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class File(NonCompletableGithubObject):
    """
    This class represents Files
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._additions: Attribute[int] = NotSet
        self._blob_url: Attribute[str] = NotSet
        self._changes: Attribute[int] = NotSet
        self._contents_url: Attribute[str] = NotSet
        self._deletions: Attribute[int] = NotSet
        self._filename: Attribute[str] = NotSet
        self._patch: Attribute[str] = NotSet
        self._previous_filename: Attribute[str] = NotSet
        self._raw_url: Attribute[str] = NotSet
        self._sha: Attribute[str] = NotSet
        self._status: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return self.get__repr__({'sha': self._sha.value, 'filename': self._filename.value})

    @property
    def additions(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._additions.value

    @property
    def blob_url(self) -> str:
        if False:
            return 10
        return self._blob_url.value

    @property
    def changes(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._changes.value

    @property
    def contents_url(self) -> str:
        if False:
            while True:
                i = 10
        return self._contents_url.value

    @property
    def deletions(self) -> int:
        if False:
            print('Hello World!')
        return self._deletions.value

    @property
    def filename(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._filename.value

    @property
    def patch(self) -> str:
        if False:
            return 10
        return self._patch.value

    @property
    def previous_filename(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._previous_filename.value

    @property
    def raw_url(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._raw_url.value

    @property
    def sha(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._sha.value

    @property
    def status(self) -> str:
        if False:
            while True:
                i = 10
        return self._status.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'additions' in attributes:
            self._additions = self._makeIntAttribute(attributes['additions'])
        if 'blob_url' in attributes:
            self._blob_url = self._makeStringAttribute(attributes['blob_url'])
        if 'changes' in attributes:
            self._changes = self._makeIntAttribute(attributes['changes'])
        if 'contents_url' in attributes:
            self._contents_url = self._makeStringAttribute(attributes['contents_url'])
        if 'deletions' in attributes:
            self._deletions = self._makeIntAttribute(attributes['deletions'])
        if 'filename' in attributes:
            self._filename = self._makeStringAttribute(attributes['filename'])
        if 'patch' in attributes:
            self._patch = self._makeStringAttribute(attributes['patch'])
        if 'previous_filename' in attributes:
            self._previous_filename = self._makeStringAttribute(attributes['previous_filename'])
        if 'raw_url' in attributes:
            self._raw_url = self._makeStringAttribute(attributes['raw_url'])
        if 'sha' in attributes:
            self._sha = self._makeStringAttribute(attributes['sha'])
        if 'status' in attributes:
            self._status = self._makeStringAttribute(attributes['status'])