"""
.. module: security_monkey.sso.header_auth
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Jordan Milne <jordan.milne@reddit.com>

"""
import functools
from flask import request
from flask_login import current_user
from flask_principal import Identity, identity_changed
from flask_security.utils import login_user
from security_monkey.sso.service import setup_user
from security_monkey import db

class HeaderAuthExtension(object):
    """
    Extension for handling login via authn headers set by a trusted reverse proxy
    """

    def __init__(self, app=None):
        if False:
            print('Hello World!')
        if app:
            self.init_app(app)

    def init_app(self, app):
        if False:
            return 10
        orig_login = app.view_functions['security.login']

        @functools.wraps(orig_login)
        def _wrapped_login_view():
            if False:
                print('Hello World!')
            if app.config.get('USE_HEADER_AUTH'):
                username_header_name = app.config['HEADER_AUTH_USERNAME_HEADER']
                groups_header_name = app.config.get('HEADER_AUTH_GROUPS_HEADER')
                authed_user = request.headers.get(username_header_name)
                if not current_user.is_authenticated and authed_user:
                    groups = []
                    if groups_header_name and groups_header_name in request.headers:
                        groups = request.headers[groups_header_name].split(',')
                    user = setup_user(authed_user, groups=groups, default_role=app.config.get('HEADER_AUTH_DEFAULT_ROLE', 'View'))
                    identity_changed.send(app, identity=Identity(user.id))
                    login_user(user)
                    db.session.commit()
                    db.session.refresh(user)
            return orig_login()
        rbac = app.extensions['rbac'].rbac
        rbac.exempt(_wrapped_login_view)
        app.view_functions['security.login'] = _wrapped_login_view