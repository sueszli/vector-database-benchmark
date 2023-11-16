from __future__ import annotations
from base64 import b64decode
from dataclasses import dataclass, field
from typing import Optional

@dataclass()
class Credentials:
    auth_type: Optional[str]
    token: Optional[str]
    _username: Optional[str] = field(default=None)
    _password: Optional[str] = field(default=None)

    def __post_init__(self):
        if False:
            print('Hello World!')
        if self._auth_is_basic:
            (self._username, self._password) = b64decode(self.token.encode('utf-8')).decode().split(':')

    @property
    def username(self):
        if False:
            return 10
        if not self._auth_is_basic:
            raise AttributeError('Username is available for Basic Auth only')
        return self._username

    @property
    def password(self):
        if False:
            print('Hello World!')
        if not self._auth_is_basic:
            raise AttributeError('Password is available for Basic Auth only')
        return self._password

    @property
    def _auth_is_basic(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.auth_type == 'Basic'