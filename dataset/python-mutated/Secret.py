from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

class Secret(CompletableGithubObject):
    """
    This class represents a GitHub secret. The reference can be found here https://docs.github.com/en/rest/actions/secrets
    """

    def _initAttributes(self) -> None:
        if False:
            while True:
                i = 10
        self._name: Attribute[str] = NotSet
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
            i = 10
            return i + 15
        '\n        :type: string\n        '
        self._completeIfNotSet(self._name)
        return self._name.value

    @property
    def created_at(self) -> datetime:
        if False:
            return 10
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
            return 10
        '\n        :type: string\n        '
        return self._url.value

    def delete(self) -> None:
        if False:
            print('Hello World!')
        '\n        :calls: `DELETE {secret_url} <https://docs.github.com/en/rest/actions/secrets>`_\n        :rtype: None\n        '
        self._requester.requestJsonAndCheck('DELETE', self.url)

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])