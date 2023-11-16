from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
from urllib.parse import urlsplit
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.proxy_fix import ProxyFix
from airflow.configuration import conf
from airflow.exceptions import AirflowConfigException
if TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse, WSGIEnvironment
    from flask import Flask

def _root_app(env: WSGIEnvironment, resp: StartResponse) -> Iterable[bytes]:
    if False:
        i = 10
        return i + 15
    resp('404 Not Found', [('Content-Type', 'text/plain')])
    return [b'Apache Airflow is not at this location']

def init_wsgi_middleware(flask_app: Flask) -> None:
    if False:
        return 10
    'Handle X-Forwarded-* headers and base_url support.'
    webserver_base_url = conf.get_mandatory_value('webserver', 'BASE_URL', fallback='')
    if webserver_base_url.endswith('/'):
        raise AirflowConfigException('webserver.base_url conf cannot have a trailing slash.')
    base_url = urlsplit(webserver_base_url)[2]
    if not base_url or base_url == '/':
        base_url = ''
    if base_url:
        wsgi_app = DispatcherMiddleware(_root_app, mounts={base_url: flask_app.wsgi_app})
        flask_app.wsgi_app = wsgi_app
    if conf.getboolean('webserver', 'ENABLE_PROXY_FIX'):
        flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_for=conf.getint('webserver', 'PROXY_FIX_X_FOR', fallback=1), x_proto=conf.getint('webserver', 'PROXY_FIX_X_PROTO', fallback=1), x_host=conf.getint('webserver', 'PROXY_FIX_X_HOST', fallback=1), x_port=conf.getint('webserver', 'PROXY_FIX_X_PORT', fallback=1), x_prefix=conf.getint('webserver', 'PROXY_FIX_X_PREFIX', fallback=1))