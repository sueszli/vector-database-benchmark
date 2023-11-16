from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

class Variable(CompletableGithubObject):
    """
    This class represents a GitHub variable. The reference can be found here https://docs.github.com/en/rest/actions/variables
    """

    def _initAttributes(self) -> None:
        if False:
            while True:
                i = 10
        self._name: Attribute[str] = NotSet
        self._value: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'name': self.name})

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        '\n        :type: string\n        '
        self._completeIfNotSet(self._name)
        return self._name.value

    @property
    def value(self) -> str:
        if False:
            print('Hello World!')
        '\n        :type: string\n        '
        self._completeIfNotSet(self._value)
        return self._value.value

    @property
    def created_at(self) -> datetime:
        if False:
            i = 10
            return i + 15
        '\n        :type: datetime.datetime\n        '
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def updated_at(self) -> datetime:
        if False:
            print('Hello World!')
        '\n        :type: datetime.datetime\n        '
        self._completeIfNotSet(self._updated_at)
        return self._updated_at.value

    @property
    def url(self) -> str:
        if False:
            print('Hello World!')
        '\n        :type: string\n        '
        return self._url.value

    def edit(self, value: str) -> bool:
        if False:
            print('Hello World!')
        '\n        :calls: `PATCH /repos/{owner}/{repo}/actions/variables/{variable_name} <https://docs.github.com/en/rest/reference/actions/variables#update-a-repository-variable>`_\n        :param variable_name: string\n        :param value: string\n        :rtype: bool\n        '
        assert isinstance(value, str), value
        patch_parameters = {'name': self.name, 'value': value}
        (status, _, _) = self._requester.requestJson('PATCH', f'{self.url}/actions/variables/{self.name}', input=patch_parameters)
        return status == 204

    def delete(self) -> None:
        if False:
            while True:
                i = 10
        '\n        :calls: `DELETE {variable_url} <https://docs.github.com/en/rest/actions/variables>`_\n        :rtype: None\n        '
        self._requester.requestJsonAndCheck('DELETE', f'{self.url}/actions/variables/{self.name}')

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'value' in attributes:
            self._value = self._makeStringAttribute(attributes['value'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])