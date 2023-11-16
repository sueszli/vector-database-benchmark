"""Services for managing user authentication."""
from __future__ import annotations
import base64
import os
from core.domain import auth_domain
from core.domain import caching_services
from core.platform import models
from core.platform.auth import firebase_auth_services
from typing import Final, List, Optional
import webapp2
MYPY = False
if MYPY:
    from mypy_imports import auth_models
    from mypy_imports import platform_auth_services
(auth_models,) = models.Registry.import_models([models.Names.AUTH])
platform_auth_services = models.Registry.import_auth_services()
CSRF_SECRET_INSTANCE_ID: Final = 'csrf_secret'

def create_profile_user_auth_details(user_id: str, parent_user_id: str) -> auth_domain.UserAuthDetails:
    if False:
        i = 10
        return i + 15
    "Returns a domain object for a new profile user.\n\n    Args:\n        user_id: str. A user ID produced by Oppia for the new profile user.\n        parent_user_id: str. The user ID of the full user account which will own\n            the new profile account.\n\n    Returns:\n        UserAuthDetails. Auth details for the new user.\n\n    Raises:\n        ValueError. The new user's parent is itself.\n    "
    if user_id == parent_user_id:
        raise ValueError('user cannot be its own parent')
    return auth_domain.UserAuthDetails(user_id, None, None, parent_user_id)

def get_all_profiles_by_parent_user_id(parent_user_id: str) -> List[auth_models.UserAuthDetailsModel]:
    if False:
        for i in range(10):
            print('nop')
    'Fetch the auth details of all profile users with the given parent user.\n\n    Args:\n        parent_user_id: str. The user ID of the parent user.\n\n    Returns:\n        list(UserAuthDetailsModel). List of UserAuthDetailsModel instances\n        with the given parent user.\n    '
    return list(auth_models.UserAuthDetailsModel.query(auth_models.UserAuthDetailsModel.parent_user_id == parent_user_id).fetch())

def establish_auth_session(request: webapp2.Request, response: webapp2.Response) -> None:
    if False:
        i = 10
        return i + 15
    "Sets login cookies to maintain a user's sign-in session.\n\n    Args:\n        request: webapp2.Request. The request with the authorization to begin a\n            new session.\n        response: webapp2.Response. The response to establish the new session\n            upon.\n    "
    platform_auth_services.establish_auth_session(request, response)

def destroy_auth_session(response: webapp2.Response) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Clears login cookies from the given response headers.\n\n    Args:\n        response: webapp2.Response. Response to clear the cookies from.\n    '
    platform_auth_services.destroy_auth_session(response)

def get_user_auth_details_from_model(user_auth_details_model: auth_models.UserAuthDetailsModel) -> auth_domain.UserAuthDetails:
    if False:
        print('Hello World!')
    'Returns a UserAuthDetails domain object from the given model.\n\n    Args:\n        user_auth_details_model: UserAuthDetailsModel. The source model.\n\n    Returns:\n        UserAuthDetails. The domain object with values taken from the model.\n    '
    return auth_domain.UserAuthDetails(user_auth_details_model.id, user_auth_details_model.gae_id, user_auth_details_model.firebase_auth_id, user_auth_details_model.parent_user_id, deleted=user_auth_details_model.deleted)

def get_auth_claims_from_request(request: webapp2.Request) -> Optional[auth_domain.AuthClaims]:
    if False:
        i = 10
        return i + 15
    'Authenticates the request and returns claims about its authorizer.\n\n    Args:\n        request: webapp2.Request. The HTTP request to authenticate.\n\n    Returns:\n        AuthClaims|None. Claims about the currently signed in user. If no user\n        is signed in, then returns None.\n\n    Raises:\n        InvalidAuthSessionError. The request contains an invalid session.\n        StaleAuthSessionError. The cookie has lost its authority.\n    '
    return platform_auth_services.get_auth_claims_from_request(request)

def mark_user_for_deletion(user_id: str) -> None:
    if False:
        print('Hello World!')
    'Marks the user, and all of their auth associations, as deleted.\n\n    Args:\n        user_id: str. The unique ID of the user whose associations should be\n            deleted.\n    '
    platform_auth_services.mark_user_for_deletion(user_id)

def delete_external_auth_associations(user_id: str) -> None:
    if False:
        while True:
            i = 10
    'Deletes all associations that refer to the user outside of Oppia.\n\n    Args:\n        user_id: str. The unique ID of the user whose associations should be\n            deleted.\n    '
    platform_auth_services.delete_external_auth_associations(user_id)

def verify_external_auth_associations_are_deleted(user_id: str) -> bool:
    if False:
        while True:
            i = 10
    'Returns true if and only if we have successfully verified that all\n    external associations have been deleted.\n\n    Args:\n        user_id: str. The unique ID of the user whose associations should be\n            checked.\n\n    Returns:\n        bool. True if and only if we have successfully verified that all\n        external associations have been deleted.\n    '
    return platform_auth_services.verify_external_auth_associations_are_deleted(user_id)

def get_auth_id_from_user_id(user_id: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    'Returns the auth ID associated with the given user ID.\n\n    Args:\n        user_id: str. The auth ID.\n\n    Returns:\n        str|None. The user ID associated with the given auth ID, or None if no\n        association exists.\n    '
    return platform_auth_services.get_auth_id_from_user_id(user_id)

def get_multi_auth_ids_from_user_ids(user_ids: List[str]) -> List[Optional[str]]:
    if False:
        while True:
            i = 10
    "Returns the auth IDs associated with the given user IDs.\n\n    Args:\n        user_ids: list(str). The user IDs.\n\n    Returns:\n        list(str|None). The auth IDs associated with each of the given user IDs,\n        or None for associations which don't exist.\n    "
    return platform_auth_services.get_multi_auth_ids_from_user_ids(user_ids)

def get_user_id_from_auth_id(auth_id: str, include_deleted: bool=False) -> Optional[str]:
    if False:
        print('Hello World!')
    'Returns the user ID associated with the given auth ID.\n\n    Args:\n        auth_id: str. The auth ID.\n        include_deleted: bool. Whether to return the ID of models marked for\n            deletion.\n\n    Returns:\n        str|None. The user ID associated with the given auth ID, or None if no\n        association exists.\n    '
    return platform_auth_services.get_user_id_from_auth_id(auth_id, include_deleted=include_deleted)

def get_multi_user_ids_from_auth_ids(auth_ids: List[str]) -> List[Optional[str]]:
    if False:
        while True:
            i = 10
    "Returns the user IDs associated with the given auth IDs.\n\n    Args:\n        auth_ids: list(str). The auth IDs.\n\n    Returns:\n        list(str|None). The user IDs associated with each of the given auth IDs,\n        or None for associations which don't exist.\n    "
    return platform_auth_services.get_multi_user_ids_from_auth_ids(auth_ids)

def associate_auth_id_with_user_id(auth_id_user_id_pair: auth_domain.AuthIdUserIdPair) -> None:
    if False:
        while True:
            i = 10
    'Commits the association between auth ID and user ID.\n\n    Args:\n        auth_id_user_id_pair: auth_domain.AuthIdUserIdPair. The association to\n            commit.\n\n    Raises:\n        Exception. The IDs are already associated with a value.\n    '
    platform_auth_services.associate_auth_id_with_user_id(auth_id_user_id_pair)

def associate_multi_auth_ids_with_user_ids(auth_id_user_id_pairs: List[auth_domain.AuthIdUserIdPair]) -> None:
    if False:
        i = 10
        return i + 15
    'Commits the associations between auth IDs and user IDs.\n\n    Args:\n        auth_id_user_id_pairs: list(auth_domain.AuthIdUserIdPair). The\n            associations to commit.\n\n    Raises:\n        Exception. One or more auth associations already exist.\n    '
    platform_auth_services.associate_multi_auth_ids_with_user_ids(auth_id_user_id_pairs)

def grant_super_admin_privileges(user_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Grants the user super admin privileges.\n\n    Args:\n        user_id: str. The Oppia user ID to promote to super admin.\n    '
    firebase_auth_services.grant_super_admin_privileges(user_id)

def revoke_super_admin_privileges(user_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Revokes the user's super admin privileges.\n\n    Args:\n        user_id: str. The Oppia user ID to revoke privileges from.\n    "
    firebase_auth_services.revoke_super_admin_privileges(user_id)

def get_csrf_secret_value() -> str:
    if False:
        return 10
    'Returns the CSRF secret value. If this value does not exist,\n    creates a new secret value and returns it.\n\n    Returns:\n        str. Returns the csrf secret value.\n    '
    memcached_items = caching_services.get_multi(caching_services.CACHE_NAMESPACE_DEFAULT, None, [CSRF_SECRET_INSTANCE_ID])
    if CSRF_SECRET_INSTANCE_ID in memcached_items:
        csrf_value = memcached_items[CSRF_SECRET_INSTANCE_ID]
        assert isinstance(csrf_value, str)
        return csrf_value
    csrf_secret_model = auth_models.CsrfSecretModel.get(CSRF_SECRET_INSTANCE_ID, strict=False)
    if csrf_secret_model is None:
        csrf_secret_value = base64.urlsafe_b64encode(os.urandom(20)).decode()
        auth_models.CsrfSecretModel(id=CSRF_SECRET_INSTANCE_ID, oppia_csrf_secret=csrf_secret_value).put()
        caching_services.set_multi(caching_services.CACHE_NAMESPACE_DEFAULT, None, {CSRF_SECRET_INSTANCE_ID: csrf_secret_value})
        csrf_secret_model = auth_models.CsrfSecretModel.get(CSRF_SECRET_INSTANCE_ID, strict=False)
    assert csrf_secret_model is not None
    csrf_secret_value = csrf_secret_model.oppia_csrf_secret
    assert isinstance(csrf_secret_value, str)
    return csrf_secret_value