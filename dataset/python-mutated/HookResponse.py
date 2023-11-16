from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class HookResponse(NonCompletableGithubObject):
    """
    This class represents HookResponses
    """

    def _initAttributes(self) -> None:
        if False:
            return 10
        self._code: Attribute[int] = NotSet
        self._message: Attribute[str] = NotSet
        self._status: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return self.get__repr__({'status': self._status.value})

    @property
    def code(self) -> int:
        if False:
            while True:
                i = 10
        return self._code.value

    @property
    def message(self) -> str:
        if False:
            print('Hello World!')
        return self._message.value

    @property
    def status(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._status.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'code' in attributes:
            self._code = self._makeIntAttribute(attributes['code'])
        if 'message' in attributes:
            self._message = self._makeStringAttribute(attributes['message'])
        if 'status' in attributes:
            self._status = self._makeStringAttribute(attributes['status'])