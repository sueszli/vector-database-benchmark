from __future__ import annotations
from typing import Any
from github.GithubObject import NotSet, Opt, is_defined, is_optional

class InputFileContent:
    """
    This class represents InputFileContents
    """

    def __init__(self, content: str, new_name: Opt[str]=NotSet):
        if False:
            print('Hello World!')
        assert isinstance(content, str), content
        assert is_optional(new_name, str), new_name
        self.__newName: Opt[str] = new_name
        self.__content: str = content

    @property
    def _identity(self) -> dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        identity: dict[str, Any] = {'content': self.__content}
        if is_defined(self.__newName):
            identity['filename'] = self.__newName
        return identity