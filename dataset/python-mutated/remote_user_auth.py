import logging
from flask import Blueprint, redirect, request, url_for
from redash import settings
from redash.authentication import create_and_login_user, get_next_path, logout_and_redirect_to_index
from redash.authentication.org_resolving import current_org
from redash.handlers.base import org_scoped_rule
logger = logging.getLogger('remote_user_auth')
blueprint = Blueprint('remote_user_auth', __name__)

@blueprint.route(org_scoped_rule('/remote_user/login'))
def login(org_slug=None):
    if False:
        print('Hello World!')
    unsafe_next_path = request.args.get('next')
    next_path = get_next_path(unsafe_next_path)
    if not settings.REMOTE_USER_LOGIN_ENABLED:
        logger.error('Cannot use remote user for login without being enabled in settings')
        return redirect(url_for('redash.index', next=next_path, org_slug=org_slug))
    email = request.headers.get(settings.REMOTE_USER_HEADER)
    if email == '(null)':
        email = None
    if not email:
        logger.error("Cannot use remote user for login when it's not provided in the request (looked in headers['" + settings.REMOTE_USER_HEADER + "'])")
        return redirect(url_for('redash.index', next=next_path, org_slug=org_slug))
    logger.info('Logging in ' + email + ' via remote user')
    user = create_and_login_user(current_org, email, email)
    if user is None:
        return logout_and_redirect_to_index()
    return redirect(next_path or url_for('redash.index', org_slug=org_slug), code=302)