""" Abstract request handler that handles bokeh-session-id

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
from tornado.httputil import HTTPServerRequest
from tornado.web import HTTPError, authenticated
from bokeh.util.token import check_token_signature, generate_jwt_token, generate_session_id, get_session_id
from .auth_request_handler import AuthRequestHandler
if TYPE_CHECKING:
    from ...core.types import ID
    from ..contexts import ApplicationContext
    from ..session import ServerSession
    from ..tornado import BokehTornado
__all__ = ('SessionHandler',)

class SessionHandler(AuthRequestHandler):
    """ Implements a custom Tornado handler for document display page

    """
    application: BokehTornado
    request: HTTPServerRequest
    application_context: ApplicationContext
    bokeh_websocket_path: str

    def __init__(self, tornado_app: BokehTornado, *args, **kw) -> None:
        if False:
            while True:
                i = 10
        self.application_context = kw['application_context']
        self.bokeh_websocket_path = kw['bokeh_websocket_path']
        super().__init__(tornado_app, *args, **kw)

    def initialize(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        pass

    @authenticated
    async def get_session(self) -> ServerSession:
        app = self.application
        token = self.get_argument('bokeh-token', default=None)
        session_id: ID | None = self.get_argument('bokeh-session-id', default=None)
        if 'Bokeh-Session-Id' in self.request.headers:
            if session_id is not None:
                log.debug('Server received session ID in request argument and header, expected only one')
                raise HTTPError(status_code=403, reason='session ID was provided as an argument and header')
            session_id = self.request.headers.get('Bokeh-Session-Id')
        if token is not None:
            if session_id is not None:
                log.debug('Server received both token and session ID, expected only one')
                raise HTTPError(status_code=403, reason='Both token and session ID were provided')
            session_id = get_session_id(token)
        elif session_id is None:
            if app.generate_session_ids:
                session_id = generate_session_id(secret_key=app.secret_key, signed=app.sign_sessions)
            else:
                log.debug('Server configured not to generate session IDs and none was provided')
                raise HTTPError(status_code=403, reason='No bokeh-session-id provided')
        if token is None:
            if app.include_headers is None:
                excluded_headers = app.exclude_headers or []
                allowed_headers = [header for header in self.request.headers if header not in excluded_headers]
            else:
                allowed_headers = app.include_headers
            headers = {k: v for (k, v) in self.request.headers.items() if k in allowed_headers}
            if app.include_cookies is None:
                excluded_cookies = app.exclude_cookies or []
                allowed_cookies = [cookie for cookie in self.request.cookies if cookie not in excluded_cookies]
            else:
                allowed_cookies = app.include_cookies
            cookies = {k: v.value for (k, v) in self.request.cookies.items() if k in allowed_cookies}
            if cookies and 'Cookie' in headers and ('Cookie' not in (app.include_headers or [])):
                del headers['Cookie']
            arguments = {} if self.request.arguments is None else self.request.arguments
            payload = {'headers': headers, 'cookies': cookies, 'arguments': arguments}
            payload.update(self.application_context.application.process_request(self.request))
            token = generate_jwt_token(session_id, secret_key=app.secret_key, signed=app.sign_sessions, expiration=app.session_token_expiration, extra_payload=payload)
        if not check_token_signature(token, secret_key=app.secret_key, signed=app.sign_sessions):
            log.error('Session id had invalid signature: %r', session_id)
            raise HTTPError(status_code=403, reason='Invalid token or session ID')
        session = await self.application_context.create_session_if_needed(session_id, self.request, token)
        return session