from enum import Enum
from typing import Optional, TypedDict, Union
from flask_appbuilder.security.sqla.models import Role
from flask_login import AnonymousUserMixin

class GuestTokenUser(TypedDict, total=False):
    username: str
    first_name: str
    last_name: str

class GuestTokenResourceType(Enum):
    DASHBOARD = 'dashboard'

class GuestTokenResource(TypedDict):
    type: GuestTokenResourceType
    id: Union[str, int]
GuestTokenResources = list[GuestTokenResource]

class GuestTokenRlsRule(TypedDict):
    dataset: Optional[str]
    clause: str

class GuestToken(TypedDict):
    iat: float
    exp: float
    user: GuestTokenUser
    resources: GuestTokenResources
    rls_rules: list[GuestTokenRlsRule]

class GuestUser(AnonymousUserMixin):
    """
    Used as the "anonymous" user in case of guest authentication (embedded)
    """
    is_guest_user = True

    @property
    def is_authenticated(self) -> bool:
        if False:
            print('Hello World!')
        '\n        This is set to true because guest users should be considered authenticated,\n        at least in most places. The treatment of this flag is kind of inconsistent.\n        '
        return True

    @property
    def is_anonymous(self) -> bool:
        if False:
            return 10
        '\n        This is set to false because lots of code assumes that\n        if user.is_anonymous, then role = Public\n        But guest users need to have their own role independent of Public.\n        '
        return False

    def __init__(self, token: GuestToken, roles: list[Role]):
        if False:
            for i in range(10):
                print('nop')
        user = token['user']
        self.guest_token = token
        self.username = user.get('username', 'guest_user')
        self.first_name = user.get('first_name', 'Guest')
        self.last_name = user.get('last_name', 'User')
        self.roles = roles
        self.resources = token['resources']
        self.rls = token.get('rls_rules', [])