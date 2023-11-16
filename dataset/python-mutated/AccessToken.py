from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class AccessToken(NonCompletableGithubObject):
    """
    This class represents access tokens.
    """
    _created: datetime

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._token: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet
        self._scope: Attribute[str] = NotSet
        self._expires_in: Attribute[int | None] = NotSet
        self._refresh_token: Attribute[str] = NotSet
        self._refresh_expires_in: Attribute[int | None] = NotSet

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return self.get__repr__({'token': f'{self.token[:5]}...', 'scope': self.scope, 'type': self.type, 'expires_in': self.expires_in, 'refresh_token': f'{self.refresh_token[:5]}...' if self.refresh_token else None, 'refresh_token_expires_in': self.refresh_expires_in})

    @property
    def token(self) -> str:
        if False:
            print('Hello World!')
        '\n        :type: string\n        '
        return self._token.value

    @property
    def type(self) -> str:
        if False:
            return 10
        '\n        :type: string\n        '
        return self._type.value

    @property
    def scope(self) -> str:
        if False:
            return 10
        '\n        :type: string\n        '
        return self._scope.value

    @property
    def created(self) -> datetime:
        if False:
            print('Hello World!')
        '\n        :type: datetime\n        '
        return self._created

    @property
    def expires_in(self) -> int | None:
        if False:
            return 10
        '\n        :type: Optional[int]\n        '
        return self._expires_in.value

    @property
    def expires_at(self) -> datetime | None:
        if False:
            while True:
                i = 10
        '\n        :type: Optional[datetime]\n        '
        seconds = self.expires_in
        if seconds is not None:
            return self._created + timedelta(seconds=seconds)
        return None

    @property
    def refresh_token(self) -> str | None:
        if False:
            i = 10
            return i + 15
        '\n        :type: Optional[string]\n        '
        return self._refresh_token.value

    @property
    def refresh_expires_in(self) -> int | None:
        if False:
            print('Hello World!')
        '\n        :type: Optional[int]\n        '
        return self._refresh_expires_in.value

    @property
    def refresh_expires_at(self) -> datetime | None:
        if False:
            print('Hello World!')
        '\n        :type: Optional[datetime]\n        '
        seconds = self.refresh_expires_in
        if seconds is not None:
            return self._created + timedelta(seconds=seconds)
        return None

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        self._created = datetime.now(timezone.utc)
        if 'access_token' in attributes:
            self._token = self._makeStringAttribute(attributes['access_token'])
        if 'token_type' in attributes:
            self._type = self._makeStringAttribute(attributes['token_type'])
        if 'scope' in attributes:
            self._scope = self._makeStringAttribute(attributes['scope'])
        if 'expires_in' in attributes:
            self._expires_in = self._makeIntAttribute(attributes['expires_in'])
        if 'refresh_token' in attributes:
            self._refresh_token = self._makeStringAttribute(attributes['refresh_token'])
        if 'refresh_token_expires_in' in attributes:
            self._refresh_expires_in = self._makeIntAttribute(attributes['refresh_token_expires_in'])