from abc import abstractmethod
from typing import Any, Mapping
from requests.auth import AuthBase

class AbstractHeaderAuthenticator(AuthBase):
    """Abstract class for an header-based authenticators that add a header to outgoing HTTP requests."""

    def __call__(self, request):
        if False:
            for i in range(10):
                print('nop')
        'Attach the HTTP headers required to authenticate on the HTTP request'
        request.headers.update(self.get_auth_header())
        return request

    def get_auth_header(self) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        'The header to set on outgoing HTTP requests'
        if self.auth_header:
            return {self.auth_header: self.token}
        return {}

    @property
    @abstractmethod
    def auth_header(self) -> str:
        if False:
            i = 10
            return i + 15
        'HTTP header to set on the requests'

    @property
    @abstractmethod
    def token(self) -> str:
        if False:
            i = 10
            return i + 15
        'The header value to set on outgoing HTTP requests'