from abc import ABC, abstractmethod
from typing import Any, Optional
from django.http import HttpRequest
from ninja.compatibility.request import get_headers
from ninja.errors import HttpError
from ninja.security.base import AuthBase
from ninja.utils import check_csrf
__all__ = ['APIKeyBase', 'APIKeyQuery', 'APIKeyCookie', 'APIKeyHeader']

class APIKeyBase(AuthBase, ABC):
    openapi_type: str = 'apiKey'
    param_name: str = 'key'

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.openapi_name = self.param_name
        super().__init__()

    def __call__(self, request: HttpRequest) -> Optional[Any]:
        if False:
            print('Hello World!')
        key = self._get_key(request)
        return self.authenticate(request, key)

    @abstractmethod
    def _get_key(self, request: HttpRequest) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def authenticate(self, request: HttpRequest, key: Optional[str]) -> Optional[Any]:
        if False:
            for i in range(10):
                print('nop')
        pass

class APIKeyQuery(APIKeyBase, ABC):
    openapi_in: str = 'query'

    def _get_key(self, request: HttpRequest) -> Optional[str]:
        if False:
            return 10
        return request.GET.get(self.param_name)

class APIKeyCookie(APIKeyBase, ABC):
    openapi_in: str = 'cookie'

    def __init__(self, csrf: bool=True) -> None:
        if False:
            print('Hello World!')
        self.csrf = csrf
        super().__init__()

    def _get_key(self, request: HttpRequest) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        if self.csrf:
            error_response = check_csrf(request)
            if error_response:
                raise HttpError(403, 'CSRF check Failed')
        return request.COOKIES.get(self.param_name)

class APIKeyHeader(APIKeyBase, ABC):
    openapi_in: str = 'header'

    def _get_key(self, request: HttpRequest) -> Optional[str]:
        if False:
            print('Hello World!')
        headers = get_headers(request)
        return headers.get(self.param_name)