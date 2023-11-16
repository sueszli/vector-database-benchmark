from .enums import BasePermissionEnum

def is_app(context):
    if False:
        return 10
    return bool(context.app)

def is_user(context):
    if False:
        while True:
            i = 10
    user = context.user
    return user and user.is_active

def is_staff_user(context):
    if False:
        return 10
    return is_user(context) and context.user.is_staff

class AuthorizationFilters(BasePermissionEnum):
    AUTHENTICATED_APP = 'authorization_filters.authenticated_app'
    AUTHENTICATED_STAFF_USER = 'authorization_filters.authenticated_staff_user'
    AUTHENTICATED_USER = 'authorization_filters.authenticated_user'
    OWNER = 'authorization_filters.owner'
AUTHORIZATION_FILTER_MAP = {AuthorizationFilters.AUTHENTICATED_APP: is_app, AuthorizationFilters.AUTHENTICATED_USER: is_user, AuthorizationFilters.AUTHENTICATED_STAFF_USER: is_staff_user}

def resolve_authorization_filter_fn(perm):
    if False:
        i = 10
        return i + 15
    return AUTHORIZATION_FILTER_MAP.get(perm)