from __future__ import annotations
import logging
from importlib import import_module
from flask import g, redirect, request
from airflow.configuration import conf
from airflow.exceptions import AirflowConfigException, AirflowException
from airflow.www.extensions.init_auth_manager import get_auth_manager
log = logging.getLogger(__name__)

def init_xframe_protection(app):
    if False:
        return 10
    '\n    Add X-Frame-Options header.\n\n    Use it to avoid click-jacking attacks, by ensuring that their content is not embedded into other sites.\n\n    See also: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/X-Frame-Options\n    '
    x_frame_enabled = conf.getboolean('webserver', 'X_FRAME_ENABLED', fallback=True)
    if x_frame_enabled:
        return

    def apply_caching(response):
        if False:
            i = 10
            return i + 15
        response.headers['X-Frame-Options'] = 'DENY'
        return response
    app.after_request(apply_caching)

def init_api_experimental_auth(app):
    if False:
        while True:
            i = 10
    'Load authentication backends.'
    auth_backends = 'airflow.api.auth.backend.default'
    try:
        auth_backends = conf.get('api', 'auth_backends')
    except AirflowConfigException:
        pass
    app.api_auth = []
    try:
        for backend in auth_backends.split(','):
            auth = import_module(backend.strip())
            auth.init_app(app)
            app.api_auth.append(auth)
    except ImportError as err:
        log.critical('Cannot import %s for API authentication due to: %s', backend, err)
        raise AirflowException(err)

def init_check_user_active(app):
    if False:
        while True:
            i = 10

    @app.before_request
    def check_user_active():
        if False:
            i = 10
            return i + 15
        url_logout = get_auth_manager().get_url_logout()
        if request.path == url_logout:
            return
        if get_auth_manager().is_logged_in() and (not g.user.is_active):
            return redirect(url_logout)