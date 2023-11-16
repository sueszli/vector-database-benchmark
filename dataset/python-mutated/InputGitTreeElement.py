from __future__ import annotations
from typing import Any
from github.GithubObject import NotSet, Opt, is_defined, is_optional

class InputGitTreeElement:
    """
    This class represents InputGitTreeElements
    """

    def __init__(self, path: str, mode: str, type: str, content: Opt[str]=NotSet, sha: Opt[str | None]=NotSet):
        if False:
            i = 10
            return i + 15
        assert isinstance(path, str), path
        assert isinstance(mode, str), mode
        assert isinstance(type, str), type
        assert is_optional(content, str), content
        assert sha is None or is_optional(sha, str), sha
        self.__path = path
        self.__mode = mode
        self.__type = type
        self.__content = content
        self.__sha: Opt[str] | None = sha

    @property
    def _identity(self) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        identity: dict[str, Any] = {'path': self.__path, 'mode': self.__mode, 'type': self.__type}
        if is_defined(self.__sha):
            identity['sha'] = self.__sha
        if is_defined(self.__content):
            identity['content'] = self.__content
        return identity