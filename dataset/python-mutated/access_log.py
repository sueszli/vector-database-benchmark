from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable
from django.conf import settings
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.services.hybrid_cloud.auth import AuthenticatedToken
from sentry.utils import metrics
from . import is_frontend_request
api_access_logger = logging.getLogger('sentry.access.api')

@dataclass
class _AccessLogMetaData:
    request_start_time: float

    def get_request_duration(self) -> float:
        if False:
            print('Hello World!')
        return time.time() - self.request_start_time
RequestAuth = Any

def _get_request_auth(request: Request) -> RequestAuth | None:
    if False:
        for i in range(10):
            print('nop')
    if request.path_info.startswith(settings.ANONYMOUS_STATIC_PREFIXES):
        return None
    return getattr(request, 'auth', None)

def _get_token_name(auth: RequestAuth) -> str | None:
    if False:
        for i in range(10):
            print('nop')
    if not auth:
        return None
    if isinstance(auth, AuthenticatedToken):
        return auth.kind
    token_class = getattr(auth, '__class__', None)
    return token_class.__name__ if token_class else None

def _get_rate_limit_stats_dict(request: Request) -> dict[str, str]:
    if False:
        while True:
            i = 10
    default = {'rate_limit_type': 'DNE', 'concurrent_limit': str(None), 'concurrent_requests': str(None), 'reset_time': str(None), 'group': str(None), 'limit': str(None), 'remaining': str(None)}
    rate_limit_metadata = getattr(request, 'rate_limit_metadata', None)
    if not rate_limit_metadata:
        return default
    res = {}
    for field in default:
        res[field] = str(getattr(rate_limit_metadata, field, None))
    return res

def _create_api_access_log(request: Request, response: Response | None, access_log_metadata: _AccessLogMetaData) -> None:
    if False:
        while True:
            i = 10
    '\n    Create a log entry to be used for api metrics gathering\n    '
    try:
        try:
            view = request.resolver_match._func_path
        except AttributeError:
            view = 'Unknown'
        request_user = getattr(request, 'user', None)
        user_id = getattr(request_user, 'id', None)
        is_app = getattr(request_user, 'is_sentry_app', None)
        org_id = getattr(getattr(request, 'organization', None), 'id', None)
        request_auth = _get_request_auth(request)
        auth_id = getattr(request_auth, 'id', None)
        status_code = getattr(response, 'status_code', 500)
        log_metrics = dict(method=str(request.method), view=view, response=status_code, user_id=str(user_id), is_app=str(is_app), token_type=str(_get_token_name(request_auth)), is_frontend_request=str(is_frontend_request(request)), organization_id=str(org_id), auth_id=str(auth_id), path=str(request.path), caller_ip=str(request.META.get('REMOTE_ADDR')), user_agent=str(request.META.get('HTTP_USER_AGENT')), rate_limited=str(getattr(request, 'will_be_rate_limited', False)), rate_limit_category=str(getattr(request, 'rate_limit_category', None)), request_duration_seconds=access_log_metadata.get_request_duration(), **_get_rate_limit_stats_dict(request))
        api_access_logger.info('api.access', extra=log_metrics)
        metrics.incr('middleware.access_log.created')
    except Exception:
        api_access_logger.exception('api.access: Error capturing API access logs')

def access_log_middleware(get_response: Callable[[Request], Response]) -> Callable[[Request], Response]:
    if False:
        print('Hello World!')

    def middleware(request: Request) -> Response:
        if False:
            while True:
                i = 10
        if not settings.LOG_API_ACCESS:
            return get_response(request)
        if request.path_info.startswith(settings.ANONYMOUS_STATIC_PREFIXES):
            return get_response(request)
        access_log_metadata = _AccessLogMetaData(request_start_time=time.time())
        response = get_response(request)
        _create_api_access_log(request, response, access_log_metadata)
        return response
    return middleware