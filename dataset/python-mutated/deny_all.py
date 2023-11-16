"""Authentication backend that denies all requests."""
from __future__ import annotations
from functools import wraps
from typing import Any, Callable, TypeVar, cast
from flask import Response
CLIENT_AUTH: tuple[str, str] | Any | None = None

def init_app(_):
    if False:
        i = 10
        return i + 15
    'Initialize authentication.'
T = TypeVar('T', bound=Callable)

def requires_authentication(function: T):
    if False:
        for i in range(10):
            print('nop')
    'Decorate functions that require authentication.'

    @wraps(function)
    def decorated(*args, **kwargs):
        if False:
            print('Hello World!')
        return Response('Forbidden', 403)
    return cast(T, decorated)