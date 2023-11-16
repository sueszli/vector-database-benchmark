import traceback
from ast import literal_eval
from itertools import chain
from logging import getLogger
from urllib.parse import unquote
import flask
from flask.json import jsonify
from pyload import APPID
from ..helpers import clear_session, set_session
bp = flask.Blueprint('api', __name__)
log = getLogger(APPID)

@bp.route('/api/<func>', methods=['GET', 'POST'], endpoint='rpc')
@bp.route('/api/<func>/<args>', methods=['GET', 'POST'], endpoint='rpc')
def rpc(func, args=''):
    if False:
        return 10
    api = flask.current_app.config['PYLOAD_API']
    if flask.request.authorization:
        user = flask.request.authorization.get('username', '')
        password = flask.request.authorization.get('password', '')
    else:
        user = flask.request.form.get('u', '')
        password = flask.request.form.get('p', '')
    if user:
        user_info = api.check_auth(user, password)
        s = set_session(user_info)
    else:
        s = flask.session
    if 'role' not in s or 'perms' not in s or (not api.is_authorized(func, {'role': s['role'], 'permission': s['perms']})):
        return (jsonify({'error': 'Unauthorized'}), 401)
    args = args.split(',')
    if len(args) == 1 and (not args[0]):
        args = []
    kwargs = {}
    for (x, y) in chain(flask.request.args.items(), flask.request.form.items()):
        if x not in ('u', 'p'):
            kwargs[x] = unquote(y)
    try:
        response = call_api(func, *args, **kwargs)
    except Exception as exc:
        response = (jsonify(error=str(exc), traceback=traceback.format_exc()), 500)
    return response

def call_api(func, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    api = flask.current_app.config['PYLOAD_API']
    if func.startswith('_'):
        flask.flash(f"Invalid API call '{func}'")
        return (jsonify({'error': 'Forbidden'}), 403)
    result = getattr(api, func)(*[literal_eval(x) for x in args], **{x: literal_eval(y) for (x, y) in kwargs.items()})
    return jsonify(result)

@bp.route('/api/login', methods=['POST'], endpoint='login')
def login():
    if False:
        print('Hello World!')
    user = flask.request.form['username']
    password = flask.request.form['password']
    api = flask.current_app.config['PYLOAD_API']
    user_info = api.check_auth(user, password)
    if not user_info:
        log.error(f"Login failed for user '{user}'")
        return jsonify(False)
    s = set_session(user_info)
    log.info(f"User '{user}' successfully logged in")
    flask.flash('Logged in successfully')
    return jsonify(s)

@bp.route('/api/logout', endpoint='logout')
def logout():
    if False:
        return 10
    s = flask.session
    user = s.get('name')
    clear_session(s)
    if user:
        log.info(f"User '{user}' logged out")
    return jsonify(True)