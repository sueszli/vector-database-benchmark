from __future__ import annotations
from collections.abc import Callable
from functools import wraps
from typing import TypeVar
import firebase_admin
from firebase_admin import auth
from flask import request, Response
import structlog
a = TypeVar('a')
default_app = firebase_admin.initialize_app()

def jwt_authenticated(func: Callable[..., int]) -> Callable[..., int]:
    if False:
        i = 10
        return i + 15
    'Use the Firebase Admin SDK to parse Authorization header to verify the\n    user ID token.\n\n    The server extracts the Identity Platform uid for that user.\n    '

    @wraps(func)
    def decorated_function(*args: a, **kwargs: a) -> a:
        if False:
            for i in range(10):
                print('nop')
        header = request.headers.get('Authorization', None)
        if header:
            token = header.split(' ')[1]
            try:
                decoded_token = firebase_admin.auth.verify_id_token(token)
            except Exception as e:
                logger.exception(e)
                return Response(status=403, response=f'Error with authentication: {e}')
        else:
            return Response(status=401)
        request.uid = decoded_token['uid']
        return func(*args, **kwargs)
    return decorated_function

def field_name_modifier(logger: structlog._loggers.PrintLogger, log_method: str, event_dict: dict) -> dict:
    if False:
        print('Hello World!')
    'A structlog processor for mapping fields to Cloud Logging.\n    Learn more at https://www.structlog.org/en/stable/processors.html\n\n    Args:\n        logger: A logger object.\n        log_method: The name of the wrapped method.\n        event_dict:Current context together with the current event.\n\n    Returns:\n        A structlog processor.\n    '
    event_dict['severity'] = event_dict['level']
    del event_dict['level']
    event_dict['message'] = event_dict['event']
    del event_dict['event']
    return event_dict

def getJSONLogger() -> structlog._config.BoundLoggerLazyProxy:
    if False:
        while True:
            i = 10
    'Initialize a logger configured for JSON structured logs.\n\n    Returns:\n        A configured logger object.\n    '
    structlog.configure(processors=[structlog.stdlib.add_log_level, structlog.stdlib.PositionalArgumentsFormatter(), field_name_modifier, structlog.processors.TimeStamper('iso'), structlog.processors.JSONRenderer()], wrapper_class=structlog.stdlib.BoundLogger)
    return structlog.get_logger()
logger = getJSONLogger()

def logging_flush() -> None:
    if False:
        i = 10
        return i + 15
    pass