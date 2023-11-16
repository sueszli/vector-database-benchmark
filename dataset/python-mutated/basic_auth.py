"""Basic authentication backend."""
from __future__ import annotations
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast
from flask import Response, request
from flask_appbuilder.const import AUTH_LDAP
from flask_login import login_user
from airflow.utils.airflow_flask_app import get_airflow_app
if TYPE_CHECKING:
    from airflow.auth.managers.fab.models import User
CLIENT_AUTH: tuple[str, str] | Any | None = None
T = TypeVar('T', bound=Callable)

def init_app(_):
    if False:
        i = 10
        return i + 15
    'Initialize authentication backend.'

def auth_current_user() -> User | None:
    if False:
        i = 10
        return i + 15
    'Authenticate and set current user if Authorization header exists.'
    auth = request.authorization
    if auth is None or not auth.username or (not auth.password):
        return None
    ab_security_manager = get_airflow_app().appbuilder.sm
    user = None
    if ab_security_manager.auth_type == AUTH_LDAP:
        user = ab_security_manager.auth_user_ldap(auth.username, auth.password)
    if user is None:
        user = ab_security_manager.auth_user_db(auth.username, auth.password)
    if user is not None:
        login_user(user, remember=False)
    return user

def requires_authentication(function: T):
    if False:
        i = 10
        return i + 15
    'Decorate functions that require authentication.'

    @wraps(function)
    def decorated(*args, **kwargs):
        if False:
            while True:
                i = 10
        if auth_current_user() is not None:
            return function(*args, **kwargs)
        else:
            return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic'})
    return cast(T, decorated)