from typing import Generic, Optional, Sequence, Type
from fastapi import APIRouter
from fastapi_users import models, schemas
from fastapi_users.authentication import AuthenticationBackend, Authenticator
from fastapi_users.jwt import SecretType
from fastapi_users.manager import UserManagerDependency
from fastapi_users.router import get_auth_router, get_register_router, get_reset_password_router, get_users_router, get_verify_router
try:
    from httpx_oauth.oauth2 import BaseOAuth2
    from fastapi_users.router import get_oauth_router
    from fastapi_users.router.oauth import get_oauth_associate_router
except ModuleNotFoundError:
    BaseOAuth2 = Type

class FastAPIUsers(Generic[models.UP, models.ID]):
    """
    Main object that ties together the component for users authentication.

    :param get_user_manager: Dependency callable getter to inject the
    user manager class instance.
    :param auth_backends: List of authentication backends.

    :attribute current_user: Dependency callable getter to inject authenticated user
    with a specific set of parameters.
    """
    authenticator: Authenticator

    def __init__(self, get_user_manager: UserManagerDependency[models.UP, models.ID], auth_backends: Sequence[AuthenticationBackend]):
        if False:
            print('Hello World!')
        self.authenticator = Authenticator(auth_backends, get_user_manager)
        self.get_user_manager = get_user_manager
        self.current_user = self.authenticator.current_user

    def get_register_router(self, user_schema: Type[schemas.U], user_create_schema: Type[schemas.UC]) -> APIRouter:
        if False:
            print('Hello World!')
        '\n        Return a router with a register route.\n\n        :param user_schema: Pydantic schema of a public user.\n        :param user_create_schema: Pydantic schema for creating a user.\n        '
        return get_register_router(self.get_user_manager, user_schema, user_create_schema)

    def get_verify_router(self, user_schema: Type[schemas.U]) -> APIRouter:
        if False:
            while True:
                i = 10
        '\n        Return a router with e-mail verification routes.\n\n        :param user_schema: Pydantic schema of a public user.\n        '
        return get_verify_router(self.get_user_manager, user_schema)

    def get_reset_password_router(self) -> APIRouter:
        if False:
            print('Hello World!')
        'Return a reset password process router.'
        return get_reset_password_router(self.get_user_manager)

    def get_auth_router(self, backend: AuthenticationBackend, requires_verification: bool=False) -> APIRouter:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an auth router for a given authentication backend.\n\n        :param backend: The authentication backend instance.\n        :param requires_verification: Whether the authentication\n        require the user to be verified or not. Defaults to False.\n        '
        return get_auth_router(backend, self.get_user_manager, self.authenticator, requires_verification)

    def get_oauth_router(self, oauth_client: BaseOAuth2, backend: AuthenticationBackend, state_secret: SecretType, redirect_url: Optional[str]=None, associate_by_email: bool=False, is_verified_by_default: bool=False) -> APIRouter:
        if False:
            i = 10
            return i + 15
        "\n        Return an OAuth router for a given OAuth client and authentication backend.\n\n        :param oauth_client: The HTTPX OAuth client instance.\n        :param backend: The authentication backend instance.\n        :param state_secret: Secret used to encode the state JWT.\n        :param redirect_url: Optional arbitrary redirect URL for the OAuth2 flow.\n        If not given, the URL to the callback endpoint will be generated.\n        :param associate_by_email: If True, any existing user with the same\n        e-mail address will be associated to this user. Defaults to False.\n        :param is_verified_by_default: If True, the `is_verified` flag will be\n        set to `True` on newly created user. Make sure the OAuth Provider you're\n        using does verify the email address before enabling this flag.\n        "
        return get_oauth_router(oauth_client, backend, self.get_user_manager, state_secret, redirect_url, associate_by_email, is_verified_by_default)

    def get_oauth_associate_router(self, oauth_client: BaseOAuth2, user_schema: Type[schemas.U], state_secret: SecretType, redirect_url: Optional[str]=None, requires_verification: bool=False) -> APIRouter:
        if False:
            return 10
        '\n        Return an OAuth association router for a given OAuth client.\n\n        :param oauth_client: The HTTPX OAuth client instance.\n        :param user_schema: Pydantic schema of a public user.\n        :param state_secret: Secret used to encode the state JWT.\n        :param redirect_url: Optional arbitrary redirect URL for the OAuth2 flow.\n        If not given, the URL to the callback endpoint will be generated.\n        :param requires_verification: Whether the endpoints\n        require the users to be verified or not. Defaults to False.\n        '
        return get_oauth_associate_router(oauth_client, self.authenticator, self.get_user_manager, user_schema, state_secret, redirect_url, requires_verification)

    def get_users_router(self, user_schema: Type[schemas.U], user_update_schema: Type[schemas.UU], requires_verification: bool=False) -> APIRouter:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a router with routes to manage users.\n\n        :param user_schema: Pydantic schema of a public user.\n        :param user_update_schema: Pydantic schema for updating a user.\n        :param requires_verification: Whether the endpoints\n        require the users to be verified or not. Defaults to False.\n        '
        return get_users_router(self.get_user_manager, user_schema, user_update_schema, self.authenticator, requires_verification)