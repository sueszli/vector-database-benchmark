"""Default authentication backend - everything is allowed"""
from __future__ import annotations
import logging
from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar, cast
from flask import Response, request
from flask_login import login_user
from airflow.utils.airflow_flask_app import get_airflow_app
if TYPE_CHECKING:
    from requests.auth import AuthBase
log = logging.getLogger(__name__)
CLIENT_AUTH: tuple[str, str] | AuthBase | None = None

def init_app(_):
    if False:
        i = 10
        return i + 15
    'Initializes authentication backend'
T = TypeVar('T', bound=Callable)

def _lookup_user(user_email_or_username: str):
    if False:
        for i in range(10):
            print('nop')
    security_manager = get_airflow_app().appbuilder.sm
    user = security_manager.find_user(email=user_email_or_username) or security_manager.find_user(username=user_email_or_username)
    if not user:
        return None
    if not user.is_active:
        return None
    return user

def requires_authentication(function: T):
    if False:
        while True:
            i = 10
    'Decorator for functions that require authentication'

    @wraps(function)
    def decorated(*args, **kwargs):
        if False:
            while True:
                i = 10
        user_id = request.remote_user
        if not user_id:
            log.debug('Missing REMOTE_USER.')
            return Response('Forbidden', 403)
        log.debug('Looking for user: %s', user_id)
        user = _lookup_user(user_id)
        if not user:
            return Response('Forbidden', 403)
        log.debug('Found user: %s', user)
        login_user(user, remember=False)
        return function(*args, **kwargs)
    return cast(T, decorated)