from __future__ import annotations
from typing import Any, Union
from typing_extensions import TypedDict
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class SimpleCredit(TypedDict):
    """
    A simple credit for a security advisory.
    """
    login: str | github.NamedUser.NamedUser
    type: str
Credit = Union[SimpleCredit, 'RepositoryAdvisoryCredit']

class RepositoryAdvisoryCredit(NonCompletableGithubObject):
    """
    This class represents a credit that is assigned to a SecurityAdvisory.
    The reference can be found here https://docs.github.com/en/rest/security-advisories/repository-advisories
    """

    @property
    def login(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        :type: string\n        '
        return self._login.value

    @property
    def type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        :type: string\n        '
        return self._type.value

    def _initAttributes(self) -> None:
        if False:
            while True:
                i = 10
        self._login: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        if 'login' in attributes:
            self._login = self._makeStringAttribute(attributes['login'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])

    @staticmethod
    def _validate_credit(credit: Credit) -> None:
        if False:
            print('Hello World!')
        assert isinstance(credit, (dict, RepositoryAdvisoryCredit)), credit
        if isinstance(credit, dict):
            assert 'login' in credit, credit
            assert 'type' in credit, credit
            assert isinstance(credit['login'], (str, github.NamedUser.NamedUser)), credit['login']
            assert isinstance(credit['type'], str), credit['type']
        else:
            assert isinstance(credit.login, str), credit.login
            assert isinstance(credit.type, str), credit.type

    @staticmethod
    def _to_github_dict(credit: Credit) -> SimpleCredit:
        if False:
            while True:
                i = 10
        assert isinstance(credit, (dict, RepositoryAdvisoryCredit)), credit
        if isinstance(credit, dict):
            assert 'login' in credit, credit
            assert 'type' in credit, credit
            assert isinstance(credit['login'], (str, github.NamedUser.NamedUser)), credit['login']
            login = credit['login']
            if isinstance(login, github.NamedUser.NamedUser):
                login = login.login
            return {'login': login, 'type': credit['type']}
        else:
            return {'login': credit.login, 'type': credit.type}