from typing import Any, Optional
from superset.exceptions import SupersetException

class NotAuthorizedObject:

    def __init__(self, what_not_authorized: str):
        if False:
            i = 10
            return i + 15
        self._what_not_authorized = what_not_authorized

    def __getattr__(self, item: Any) -> None:
        if False:
            while True:
                i = 10
        raise NotAuthorizedException(self._what_not_authorized)

    def __getitem__(self, item: Any) -> None:
        if False:
            print('Hello World!')
        raise NotAuthorizedException(self._what_not_authorized)

class NotAuthorizedException(SupersetException):

    def __init__(self, what_not_authorized: str='', exception: Optional[Exception]=None) -> None:
        if False:
            return 10
        super().__init__('The user is not authorized to ' + what_not_authorized, exception)