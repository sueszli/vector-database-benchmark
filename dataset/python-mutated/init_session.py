from __future__ import annotations
from flask import session as builtin_flask_session
from airflow.configuration import conf
from airflow.exceptions import AirflowConfigException
from airflow.www.session import AirflowDatabaseSessionInterface, AirflowSecureCookieSessionInterface

def init_airflow_session_interface(app):
    if False:
        print('Hello World!')
    'Set airflow session interface.'
    config = app.config.copy()
    selected_backend = conf.get('webserver', 'SESSION_BACKEND')
    permanent_cookie = config.get('SESSION_PERMANENT', True)
    if selected_backend == 'securecookie':
        app.session_interface = AirflowSecureCookieSessionInterface()
        if permanent_cookie:

            def make_session_permanent():
                if False:
                    while True:
                        i = 10
                builtin_flask_session.permanent = True
            app.before_request(make_session_permanent)
    elif selected_backend == 'database':
        app.session_interface = AirflowDatabaseSessionInterface(app=app, db=None, permanent=permanent_cookie, table='session', key_prefix='', use_signer=True)
    else:
        raise AirflowConfigException(f"Unrecognized session backend specified in web_server_session_backend: '{selected_backend}'. Please set this to either 'database' or 'securecookie'.")