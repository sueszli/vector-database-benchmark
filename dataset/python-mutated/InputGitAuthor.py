from __future__ import annotations
from typing import Any
from github.GithubObject import NotSet, Opt, is_defined, is_optional

class InputGitAuthor:
    """
    This class represents InputGitAuthors
    """

    def __init__(self, name: str, email: str, date: Opt[str]=NotSet):
        if False:
            while True:
                i = 10
        assert isinstance(name, str), name
        assert isinstance(email, str), email
        assert is_optional(date, str), date
        self.__name: str = name
        self.__email: str = email
        self.__date: Opt[str] = date

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'InputGitAuthor(name="{self.__name}")'

    @property
    def _identity(self) -> dict[str, str]:
        if False:
            print('Hello World!')
        identity: dict[str, Any] = {'name': self.__name, 'email': self.__email}
        if is_defined(self.__date):
            identity['date'] = self.__date
        return identity