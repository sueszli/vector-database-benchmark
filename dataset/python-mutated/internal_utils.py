import logging
from typing import Optional, Dict, Any
from slack_sdk.web.internal_utils import _parse_web_class_objects, get_user_agent
from .webhook_response import WebhookResponse

def _build_body(original_body: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if False:
        return 10
    if original_body:
        body = {k: v for (k, v) in original_body.items() if v is not None}
        _parse_web_class_objects(body)
        return body
    return None

def _build_request_headers(default_headers: Dict[str, str], additional_headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    if default_headers is None and additional_headers is None:
        return {}
    request_headers = {'Content-Type': 'application/json;charset=utf-8'}
    if default_headers is None or 'User-Agent' not in default_headers:
        request_headers['User-Agent'] = get_user_agent()
    request_headers.update(default_headers)
    if additional_headers:
        request_headers.update(additional_headers)
    return request_headers

def _debug_log_response(logger, resp: WebhookResponse) -> None:
    if False:
        for i in range(10):
            print('nop')
    if logger.level <= logging.DEBUG:
        logger.debug(f'Received the following response - status: {resp.status_code}, headers: {dict(resp.headers)}, body: {resp.body}')