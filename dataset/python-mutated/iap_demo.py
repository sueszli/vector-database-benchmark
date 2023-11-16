"""Demo App Engine standard environment app for Identity-Aware Proxy.

The / handler returns the contents of the JWT header, for use in
iap_test.py.

The /identity handler demonstrates how to use the Google App Engine
standard environment's Users API to obtain the identity of users
authenticated by Identity-Aware Proxy.

To deploy this app, follow the instructions in
https://cloud.google.com/appengine/docs/python/tools/using-libraries-python-27#installing_a_third-party_library
to install the flask library into your application.
"""
import flask
from google.appengine.api import users
app = flask.Flask(__name__)

@app.route('/')
def echo_jwt():
    if False:
        for i in range(10):
            print('nop')
    return 'x-goog-authenticated-user-jwt: {}'.format(flask.request.headers.get('x-goog-iap-jwt-assertion'))

@app.route('/identity')
def show_identity():
    if False:
        while True:
            i = 10
    user = users.get_current_user()
    return 'Authenticated as {} ({})'.format(user.email(), user.user_id())