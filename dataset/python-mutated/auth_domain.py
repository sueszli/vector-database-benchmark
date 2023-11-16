"""Domain objects for authentication."""
from __future__ import annotations
import collections
from core import utils
from typing import Any, Optional, TypedDict
AuthIdUserIdPair = collections.namedtuple('AuthIdUserIdPair', ['auth_id', 'user_id'])

class InvalidAuthSessionError(Exception):
    """Error raised when an invalid auth session is detected."""
    pass

class StaleAuthSessionError(Exception):
    """Error raised when an auth session needs to be refreshed."""
    pass

class UserDisabledError(Exception):
    """Error raised when the user whose details are requested is disabled."""
    pass

class AuthClaimsDict(TypedDict):
    """Dictionary representing the AuthClaims object."""
    sub: str
    email: str
    role: str

class AuthClaims:
    """Domain object for holding onto essential Claims about an authorized user.

    A Claim is a piece of information about a user (e.g. name, mailing address,
    phone number).

    Attributes:
        auth_id: str. A unique identifier provided by an identity provider that
            is associated with the user. The ID is only unique with respect to
            the identity provider that produced it.
        email: str|None. The email address associated with the user, if any.
        role_is_super_admin: bool. Whether the user has super admin privileges.
    """

    def __init__(self, auth_id: str, email: Optional[str], role_is_super_admin: bool) -> None:
        if False:
            return 10
        if not auth_id:
            raise Exception('auth_id must not be empty')
        self.auth_id = auth_id
        self.email = email
        self.role_is_super_admin = role_is_super_admin

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'AuthClaims(auth_id=%r, email=%r, role_is_super_admin=%r)' % (self.auth_id, self.email, self.role_is_super_admin)

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash((self.auth_id, self.email, self.role_is_super_admin))

    def __eq__(self, other: Any) -> Any:
        if False:
            i = 10
            return i + 15
        return NotImplemented if not isinstance(other, AuthClaims) else (self.auth_id, self.email, self.role_is_super_admin) == (other.auth_id, other.email, other.role_is_super_admin)

class UserAuthDetailsDict(TypedDict):
    """Dictionary representing the UserAuthDetails object."""
    gae_id: Optional[str]
    firebase_auth_id: Optional[str]
    parent_user_id: Optional[str]
    deleted: bool

class UserAuthDetails:
    """Domain object representing a user's authentication details.

    There are two distinct types of user accounts: "full" and "profile".
        full: An account that is directly associated with an identity provider.
            The provider's auth_id value will be kept in its corresponding
            property (e.g. gae_id for Google AppEngine authentication and
            firebase_auth_id for Firebase authentication).
        profile: An account that depends on its parent user for authentication.
            These accounts are not directly associated with an identity
            provider.

    The distinction between profile and full user accounts are enforced through
    invariants on the properties: auth_id and parent_user_id (where auth_id is:
    gae_id or firebase_auth_id).
    Specifically: (parent_user_id is not None) if and only if (auth_id is None).
    """

    def __init__(self, user_id: str, gae_id: Optional[str], firebase_auth_id: Optional[str], parent_user_id: Optional[str], deleted: bool=False) -> None:
        if False:
            print('Hello World!')
        self.user_id = user_id
        self.gae_id = gae_id
        self.firebase_auth_id = firebase_auth_id
        self.parent_user_id = parent_user_id
        self.deleted = deleted

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return 'UserAuthDetails(user_id=%r, gae_id=%r, firebase_auth_id=%r, parent_user_id=%r, deleted=%r)' % (self.user_id, self.gae_id, self.firebase_auth_id, self.parent_user_id, self.deleted)

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Checks whether user_id, gae_id, firebase_auth_id, and parent_user_id\n        are valid.\n\n        Raises:\n            ValidationError. The user_id is not specified.\n            ValidationError. The user_id is not a string.\n            ValidationError. The user_id has the wrong format.\n            ValidationError. The gae_id is not a string.\n            ValidationError. The firebase_auth_id is not a string.\n            ValidationError. The parent_user_id has the wrong format.\n            ValidationError. The parent_user_id is set for a full user.\n            ValidationError. The parent_user_id is not set for a profile user.\n        '
        if not self.user_id:
            raise utils.ValidationError('No user_id specified')
        if not isinstance(self.user_id, str):
            raise utils.ValidationError('user_id must be a string, but got %r' % self.user_id)
        if not utils.is_user_id_valid(self.user_id):
            raise utils.ValidationError('user_id=%r has the wrong format' % self.user_id)
        if self.gae_id is not None and (not isinstance(self.gae_id, str)):
            raise utils.ValidationError('gae_id must be a string, but got %r' % self.gae_id)
        if self.firebase_auth_id is not None and (not isinstance(self.firebase_auth_id, str)):
            raise utils.ValidationError('firebase_auth_id must be a string, but got %r' % self.firebase_auth_id)
        if self.parent_user_id is not None and (not utils.is_user_id_valid(self.parent_user_id)):
            raise utils.ValidationError('parent_user_id=%r has the wrong format' % self.parent_user_id)
        if self.is_full_user() and self.parent_user_id is not None:
            raise utils.ValidationError('parent_user_id must not be set for a full user, but got gae_id=%r, firebase_auth_id=%r, parent_user_id=%r' % (self.gae_id, self.firebase_auth_id, self.parent_user_id))
        if not self.is_full_user() and self.parent_user_id is None:
            raise utils.ValidationError('parent_user_id must be set for a profile user, but got gae_id=%r, firebase_auth_id=%r, parent_user_id=%r' % (self.gae_id, self.firebase_auth_id, self.parent_user_id))

    @property
    def auth_id(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        "Returns the auth ID corresponding to the user account, if any.\n\n        This method is a utility for simplifying code that doesn't care about\n        which identity provider the auth ID came from.\n\n        Returns:\n            str. Returns firebase_auth_id if it is not None, otherwise gae_id.\n        "
        return self.firebase_auth_id or self.gae_id

    def is_full_user(self) -> bool:
        if False:
            print('Hello World!')
        'Returns whether self refers to a full user account.'
        return self.auth_id is not None

    def to_dict(self) -> UserAuthDetailsDict:
        if False:
            return 10
        "Returns values corresponding to UserAuthDetailsModel's properties.\n\n        This method is a utility for assigning values to UserAuthDetailsModel:\n            user_auth_details.validate()\n            user_auth_details_model.populate(**user_auth_details.to_dict())\n\n        NOTE: The dict returned does not include user_id because that value is\n        UserAuthDetailsModel's key. Keys are distinct from normal properties,\n        and cannot be re-assigned using the `populate()` method; trying to\n        assign to it will raise an exception.\n\n        Returns:\n            dict(str:*). A dict of values from self using UserAuthDetailsModel\n            property names as keys.\n        "
        return {'gae_id': self.gae_id, 'firebase_auth_id': self.firebase_auth_id, 'parent_user_id': self.parent_user_id, 'deleted': self.deleted}