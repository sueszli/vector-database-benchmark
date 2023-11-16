import abc
import base64
import time
from abc import ABC
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Optional, Union
import jwt
from requests import utils
from github import Consts
from github.InstallationAuthorization import InstallationAuthorization
from github.Requester import Requester, WithRequester
if TYPE_CHECKING:
    from github.GithubIntegration import GithubIntegration
ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS = 20
TOKEN_REFRESH_THRESHOLD_TIMEDELTA = timedelta(seconds=ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS)

class Auth(abc.ABC):
    """
    This class is the base class of all authentication methods for Requester.
    """

    @property
    @abc.abstractmethod
    def token_type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        The type of the auth token as used in the HTTP Authorization header, e.g. Bearer or Basic.\n        :return: token type\n        '

    @property
    @abc.abstractmethod
    def token(self) -> str:
        if False:
            while True:
                i = 10
        '\n        The auth token as used in the HTTP Authorization header.\n        :return: token\n        '

class HTTPBasicAuth(Auth, abc.ABC):

    @property
    @abc.abstractmethod
    def username(self) -> str:
        if False:
            print('Hello World!')
        'The username.'

    @property
    @abc.abstractmethod
    def password(self) -> str:
        if False:
            while True:
                i = 10
        'The password'

    @property
    def token_type(self) -> str:
        if False:
            print('Hello World!')
        return 'Basic'

    @property
    def token(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return base64.b64encode(f'{self.username}:{self.password}'.encode()).decode('utf-8').replace('\n', '')

class Login(HTTPBasicAuth):
    """
    This class is used to authenticate with login and password.
    """

    def __init__(self, login: str, password: str):
        if False:
            return 10
        assert isinstance(login, str)
        assert len(login) > 0
        assert isinstance(password, str)
        assert len(password) > 0
        self._login = login
        self._password = password

    @property
    def login(self) -> str:
        if False:
            while True:
                i = 10
        return self._login

    @property
    def username(self) -> str:
        if False:
            print('Hello World!')
        return self.login

    @property
    def password(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._password

class Token(Auth):
    """
    This class is used to authenticate with a single constant token.
    """

    def __init__(self, token: str):
        if False:
            print('Hello World!')
        assert isinstance(token, str)
        assert len(token) > 0
        self._token = token

    @property
    def token_type(self) -> str:
        if False:
            while True:
                i = 10
        return 'token'

    @property
    def token(self) -> str:
        if False:
            print('Hello World!')
        return self._token

class JWT(Auth, ABC):
    """
    This class is the base class to authenticate with a JSON Web Token (JWT).
    https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-json-web-token-jwt-for-a-github-app
    """

    @property
    def token_type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'Bearer'

class AppAuth(JWT):
    """
    This class is used to authenticate as a GitHub App.
    https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app
    """

    def __init__(self, app_id: Union[int, str], private_key: str, jwt_expiry: int=Consts.DEFAULT_JWT_EXPIRY, jwt_issued_at: int=Consts.DEFAULT_JWT_ISSUED_AT, jwt_algorithm: str=Consts.DEFAULT_JWT_ALGORITHM):
        if False:
            return 10
        assert isinstance(app_id, (int, str)), app_id
        if isinstance(app_id, str):
            assert len(app_id) > 0, 'app_id must not be empty'
        assert isinstance(private_key, str)
        assert len(private_key) > 0, 'private_key must not be empty'
        assert isinstance(jwt_expiry, int), jwt_expiry
        assert Consts.MIN_JWT_EXPIRY <= jwt_expiry <= Consts.MAX_JWT_EXPIRY, jwt_expiry
        self._app_id = app_id
        self._private_key = private_key
        self._jwt_expiry = jwt_expiry
        self._jwt_issued_at = jwt_issued_at
        self._jwt_algorithm = jwt_algorithm

    @property
    def app_id(self) -> Union[int, str]:
        if False:
            while True:
                i = 10
        return self._app_id

    @property
    def private_key(self) -> str:
        if False:
            print('Hello World!')
        return self._private_key

    @property
    def token(self) -> str:
        if False:
            while True:
                i = 10
        return self.create_jwt()

    def get_installation_auth(self, installation_id: int, token_permissions: Optional[Dict[str, str]]=None, requester: Optional[Requester]=None) -> 'AppInstallationAuth':
        if False:
            print('Hello World!')
        '\n        Creates a github.Auth.AppInstallationAuth instance for an installation.\n        :param installation_id: installation id\n        :param token_permissions: optional permissions\n        :param requester: optional requester with app authentication\n        :return:\n        '
        return AppInstallationAuth(self, installation_id, token_permissions, requester)

    def create_jwt(self, expiration: Optional[int]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a signed JWT\n        https://docs.github.com/en/developers/apps/building-github-apps/authenticating-with-github-apps#authenticating-as-a-github-app\n\n        :return string: jwt\n        '
        if expiration is not None:
            assert isinstance(expiration, int), expiration
            assert Consts.MIN_JWT_EXPIRY <= expiration <= Consts.MAX_JWT_EXPIRY, expiration
        now = int(time.time())
        payload = {'iat': now + self._jwt_issued_at, 'exp': now + (expiration if expiration is not None else self._jwt_expiry), 'iss': self._app_id}
        encrypted = jwt.encode(payload, key=self.private_key, algorithm=self._jwt_algorithm)
        if isinstance(encrypted, bytes):
            return encrypted.decode('utf-8')
        return encrypted

class AppAuthToken(JWT):
    """
    This class is used to authenticate as a GitHub App with a single constant JWT.
    https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app
    """

    def __init__(self, token: str):
        if False:
            i = 10
            return i + 15
        assert isinstance(token, str)
        assert len(token) > 0
        self._token = token

    @property
    def token(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._token

class AppInstallationAuth(Auth, WithRequester['AppInstallationAuth']):
    """
    This class is used to authenticate as a GitHub App Installation.
    https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app-installation
    """
    __integration: Optional['GithubIntegration'] = None
    __installation_authorization: Optional[InstallationAuthorization] = None

    def __init__(self, app_auth: AppAuth, installation_id: int, token_permissions: Optional[Dict[str, str]]=None, requester: Optional[Requester]=None):
        if False:
            while True:
                i = 10
        super().__init__()
        assert isinstance(app_auth, AppAuth), app_auth
        assert isinstance(installation_id, int), installation_id
        assert token_permissions is None or isinstance(token_permissions, dict), token_permissions
        self._app_auth = app_auth
        self._installation_id = installation_id
        self._token_permissions = token_permissions
        if requester is not None:
            self.withRequester(requester)

    def withRequester(self, requester: Requester) -> 'AppInstallationAuth':
        if False:
            for i in range(10):
                print('nop')
        super().withRequester(requester.withAuth(self._app_auth))
        from github.GithubIntegration import GithubIntegration
        self.__integration = GithubIntegration(**self.requester.kwargs)
        return self

    @property
    def app_id(self) -> Union[int, str]:
        if False:
            i = 10
            return i + 15
        return self._app_auth.app_id

    @property
    def private_key(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._app_auth.private_key

    @property
    def installation_id(self) -> int:
        if False:
            return 10
        return self._installation_id

    @property
    def token_permissions(self) -> Optional[Dict[str, str]]:
        if False:
            while True:
                i = 10
        return self._token_permissions

    @property
    def token_type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'token'

    @property
    def token(self) -> str:
        if False:
            return 10
        if self.__installation_authorization is None or self._is_expired:
            self.__installation_authorization = self._get_installation_authorization()
        return self.__installation_authorization.token

    @property
    def _is_expired(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        assert self.__installation_authorization is not None
        token_expires_at = self.__installation_authorization.expires_at - TOKEN_REFRESH_THRESHOLD_TIMEDELTA
        return token_expires_at < datetime.now(timezone.utc)

    def _get_installation_authorization(self) -> InstallationAuthorization:
        if False:
            return 10
        assert self.__integration is not None, 'Method withRequester(Requester) must be called first'
        return self.__integration.get_access_token(self._installation_id, permissions=self._token_permissions)

class AppUserAuth(Auth, WithRequester['AppUserAuth']):
    """
    This class is used to authenticate as a GitHub App on behalf of a user.
    https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-with-a-github-app-on-behalf-of-a-user
    """
    _client_id: str
    _client_secret: str
    _token: str
    _type: str
    _scope: Optional[str]
    _expires_at: Optional[datetime]
    _refresh_token: Optional[str]
    _refresh_expires_at: Optional[datetime]
    from github.ApplicationOAuth import ApplicationOAuth
    __app: ApplicationOAuth

    def __init__(self, client_id: str, client_secret: str, token: str, token_type: Optional[str]=None, expires_at: Optional[datetime]=None, refresh_token: Optional[str]=None, refresh_expires_at: Optional[datetime]=None, requester: Optional[Requester]=None) -> None:
        if False:
            return 10
        super().__init__()
        assert isinstance(client_id, str)
        assert len(client_id) > 0
        assert isinstance(client_secret, str)
        assert len(client_secret) > 0
        assert isinstance(token, str)
        assert len(token) > 0
        if token_type is not None:
            assert isinstance(token_type, str)
            assert len(token_type) > 0
        assert isinstance(token, str)
        if token_type is not None:
            assert isinstance(token_type, str)
            assert len(token_type) > 0
        if expires_at is not None:
            assert isinstance(expires_at, datetime)
        if refresh_token is not None:
            assert isinstance(refresh_token, str)
            assert len(refresh_token) > 0
        if refresh_expires_at is not None:
            assert isinstance(refresh_expires_at, datetime)
        self._client_id = client_id
        self._client_secret = client_secret
        self._token = token
        self._type = token_type or 'bearer'
        self._expires_at = expires_at
        self._refresh_token = refresh_token
        self._refresh_expires_at = refresh_expires_at
        if requester is not None:
            self.withRequester(requester)

    @property
    def token_type(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._type

    @property
    def token(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self._is_expired:
            self._refresh()
        return self._token

    def withRequester(self, requester: Requester) -> 'AppUserAuth':
        if False:
            for i in range(10):
                print('nop')
        super().withRequester(requester.withAuth(None))
        from github.ApplicationOAuth import ApplicationOAuth
        self.__app = ApplicationOAuth(super().requester, headers={}, attributes={'client_id': self._client_id, 'client_secret': self._client_secret}, completed=False)
        return self

    @property
    def _is_expired(self) -> bool:
        if False:
            return 10
        return self._expires_at is not None and self._expires_at < datetime.now(timezone.utc)

    def _refresh(self) -> None:
        if False:
            print('Hello World!')
        if self._refresh_token is None:
            raise RuntimeError('Cannot refresh expired token because no refresh token has been provided')
        if self._refresh_expires_at is not None and self._refresh_expires_at < datetime.now(timezone.utc):
            raise RuntimeError('Cannot refresh expired token because refresh token also expired')
        token = self.__app.refresh_access_token(self._refresh_token)
        self._token = token.token
        self._type = token.type
        self._scope = token.scope
        self._expires_at = token.expires_at
        self._refresh_token = token.refresh_token
        self._refresh_expires_at = token.refresh_expires_at

    @property
    def expires_at(self) -> Optional[datetime]:
        if False:
            return 10
        return self._expires_at

    @property
    def refresh_token(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._refresh_token

    @property
    def refresh_expires_at(self) -> Optional[datetime]:
        if False:
            print('Hello World!')
        return self._refresh_expires_at

class NetrcAuth(HTTPBasicAuth, WithRequester['NetrcAuth']):
    """
    This class is used to authenticate via .netrc.
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._login: Optional[str] = None
        self._password: Optional[str] = None

    @property
    def username(self) -> str:
        if False:
            return 10
        return self.login

    @property
    def login(self) -> str:
        if False:
            print('Hello World!')
        assert self._login is not None, 'Method withRequester(Requester) must be called first'
        return self._login

    @property
    def password(self) -> str:
        if False:
            print('Hello World!')
        assert self._password is not None, 'Method withRequester(Requester) must be called first'
        return self._password

    def withRequester(self, requester: Requester) -> 'NetrcAuth':
        if False:
            for i in range(10):
                print('nop')
        super().withRequester(requester)
        auth = utils.get_netrc_auth(requester.base_url, raise_errors=True)
        if auth is None:
            raise RuntimeError(f'Could not get credentials from netrc for host {requester.hostname}')
        (self._login, self._password) = auth
        return self