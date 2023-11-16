""" Provide a hook for supplying authorization mechanisms to a Bokeh server.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import importlib.util
from os.path import isfile
from types import ModuleType
from typing import TYPE_CHECKING, Awaitable, Callable, NewType
from tornado.httputil import HTTPServerRequest
from tornado.web import RequestHandler
from ..util.serialization import make_globally_unique_id
if TYPE_CHECKING:
    from ..core.types import PathLike
__all__ = ('AuthModule', 'AuthProvider', 'NullAuth')
User = NewType('User', object)

class AuthProvider:
    """ Abstract base class for implementing authorization hooks.

    Subclasses must supply one of: ``get_user`` or ``get_user_async``.

    Subclasses must also supply one of ``login_url`` or ``get_login_url``.

    Optionally, if ``login_url`` provides a relative URL, then ``login_handler``
    may also be supplied.

    The properties ``logout_url`` and ``get_logout_handler`` are analogous to
    the corresponding login properties, and are optional.

    """

    def __init__(self) -> None:
        if False:
            return 10
        self._validate()

    @property
    def endpoints(self) -> list[tuple[str, type[RequestHandler]]]:
        if False:
            for i in range(10):
                print('nop')
        ' URL patterns for login/logout endpoints.\n\n        '
        endpoints: list[tuple[str, type[RequestHandler]]] = []
        if self.login_handler:
            assert self.login_url is not None
            endpoints.append((self.login_url, self.login_handler))
        if self.logout_handler:
            assert self.logout_url is not None
            endpoints.append((self.logout_url, self.logout_handler))
        return endpoints

    @property
    def get_login_url(self) -> Callable[[HTTPServerRequest], str] | None:
        if False:
            return 10
        ' A function that computes a URL to redirect unathenticated users\n        to for login.\n\n        This property may return None, if a ``login_url`` is supplied\n        instead.\n\n        If a function is returned, it should accept a ``RequestHandler``\n        and return a login URL for unathenticated users.\n\n        '
        pass

    @property
    def get_user(self) -> Callable[[HTTPServerRequest], User] | None:
        if False:
            i = 10
            return i + 15
        ' A function to get the current authenticated user.\n\n        This property may return None, if a ``get_user_async`` function is\n        supplied instead.\n\n        If a function is returned, it should accept a ``RequestHandler``\n        and return the current authenticated user.\n\n        '
        pass

    @property
    def get_user_async(self) -> Callable[[HTTPServerRequest], Awaitable[User]] | None:
        if False:
            return 10
        ' An async function to get the current authenticated user.\n\n        This property may return None, if a ``get_user`` function is supplied\n        instead.\n\n        If a function is returned, it should accept a ``RequestHandler``\n        and return the current authenticated user.\n\n        '
        pass

    @property
    def login_handler(self) -> type[RequestHandler] | None:
        if False:
            while True:
                i = 10
        ' A request handler class for a login page.\n\n        This property may return None, if ``login_url`` is supplied\n        instead.\n\n        If a class is returned, it must be a subclass of RequestHandler,\n        which will used for the endpoint specified by ``logout_url``\n\n        '
        pass

    @property
    def login_url(self) -> str | None:
        if False:
            print('Hello World!')
        ' A URL to redirect unauthenticated users to for login.\n\n        This proprty may return None, if a ``get_login_url`` function is\n        supplied instead.\n\n        '
        pass

    @property
    def logout_handler(self) -> type[RequestHandler] | None:
        if False:
            return 10
        ' A request handler class for a logout page.\n\n        This property may return None.\n\n        If a class is returned, it must be a subclass of RequestHandler,\n        which will used for the endpoint specified by ``logout_url``\n\n        '
        pass

    @property
    def logout_url(self) -> str | None:
        if False:
            while True:
                i = 10
        ' A URL to redirect authenticated users to for logout.\n\n        This proprty may return None.\n\n        '
        pass

    def _validate(self) -> None:
        if False:
            print('Hello World!')
        if self.get_user and self.get_user_async:
            raise ValueError('Only one of get_user or get_user_async should be supplied')
        if (self.get_user or self.get_user_async) and (not (self.login_url or self.get_login_url)):
            raise ValueError('When user authentication is enabled, one of login_url or get_login_url must be supplied')
        if self.login_url and self.get_login_url:
            raise ValueError('At most one of login_url or get_login_url should be supplied')
        if self.login_handler and self.get_login_url:
            raise ValueError('LoginHandler cannot be used with a get_login_url() function')
        if self.login_handler and (not issubclass(self.login_handler, RequestHandler)):
            raise ValueError('LoginHandler must be a Tornado RequestHandler')
        if self.login_url and (not probably_relative_url(self.login_url)):
            raise ValueError('LoginHandler can only be used with a relative login_url')
        if self.logout_handler and (not issubclass(self.logout_handler, RequestHandler)):
            raise ValueError('LogoutHandler must be a Tornado RequestHandler')
        if self.logout_url and (not probably_relative_url(self.logout_url)):
            raise ValueError('LogoutHandler can only be used with a relative logout_url')

class AuthModule(AuthProvider):
    """ An AuthProvider configured from a Python module.

    The following properties return the corresponding values from the module if
    they exist, or None otherwise:

    * ``get_login_url``,
    * ``get_user``
    * ``get_user_async``
    * ``login_url``
    * ``logout_url``

    The ``login_handler`` property will return a ``LoginHandler`` class from the
    module, or None otherwise.

    The ``logout_handler`` property will return a ``LogoutHandler`` class from
    the module, or None otherwise.

    """

    def __init__(self, module_path: PathLike) -> None:
        if False:
            while True:
                i = 10
        if not isfile(module_path):
            raise ValueError(f'no file exists at module_path: {module_path!r}')
        self._module = load_auth_module(module_path)
        super().__init__()

    @property
    def get_user(self):
        if False:
            i = 10
            return i + 15
        return getattr(self._module, 'get_user', None)

    @property
    def get_user_async(self):
        if False:
            return 10
        return getattr(self._module, 'get_user_async', None)

    @property
    def login_url(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._module, 'login_url', None)

    @property
    def get_login_url(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._module, 'get_login_url', None)

    @property
    def login_handler(self):
        if False:
            i = 10
            return i + 15
        return getattr(self._module, 'LoginHandler', None)

    @property
    def logout_url(self):
        if False:
            while True:
                i = 10
        return getattr(self._module, 'logout_url', None)

    @property
    def logout_handler(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._module, 'LogoutHandler', None)

class NullAuth(AuthProvider):
    """ A default no-auth AuthProvider.

    All of the properties of this provider return None.

    """

    @property
    def get_user(self):
        if False:
            return 10
        return None

    @property
    def get_user_async(self):
        if False:
            return 10
        return None

    @property
    def login_url(self):
        if False:
            while True:
                i = 10
        return None

    @property
    def get_login_url(self):
        if False:
            while True:
                i = 10
        return None

    @property
    def login_handler(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    @property
    def logout_url(self):
        if False:
            return 10
        return None

    @property
    def logout_handler(self):
        if False:
            print('Hello World!')
        return None

def load_auth_module(module_path: PathLike) -> ModuleType:
    if False:
        for i in range(10):
            print('nop')
    ' Load a Python source file at a given path as a module.\n\n    Arguments:\n        module_path (str): path to a Python source file\n\n    Returns\n        module\n\n    '
    module_name = 'bokeh.auth_' + make_globally_unique_id().replace('-', '')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def probably_relative_url(url: str) -> bool:
    if False:
        print('Hello World!')
    ' Return True if a URL is not one of the common absolute URL formats.\n\n    Arguments:\n        url (str): a URL string\n\n    Returns\n        bool\n\n    '
    return not url.startswith(('http://', 'https://', '//'))