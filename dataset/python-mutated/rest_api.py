"""Demonstration of the Firebase REST API in Python"""
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache
import json
from google.auth.transport.requests import AuthorizedSession
import google.auth
_FIREBASE_SCOPES = ['https://www.googleapis.com/auth/firebase.database', 'https://www.googleapis.com/auth/userinfo.email']

@lru_cache()
def _get_session():
    if False:
        i = 10
        return i + 15
    'Provides an authed requests session object.'
    (creds, _) = google.auth.default(scopes=[_FIREBASE_SCOPES])
    authed_session = AuthorizedSession(creds)
    return authed_session

def firebase_put(path, value=None):
    if False:
        for i in range(10):
            print('nop')
    'Writes data to Firebase.\n\n    An HTTP PUT writes an entire object at the given database path. Updates to\n    fields cannot be performed without overwriting the entire object\n\n    Args:\n        path - the url to the Firebase object to write.\n        value - a json string.\n    '
    (response, content) = _get_session().put(path, body=value)
    return json.loads(content)

def firebase_patch(path, value=None):
    if False:
        while True:
            i = 10
    'Update specific children or fields\n\n    An HTTP PATCH allows specific children or fields to be updated without\n    overwriting the entire object.\n\n    Args:\n        path - the url to the Firebase object to write.\n        value - a json string.\n    '
    (response, content) = _get_session().patch(path, body=value)
    return json.loads(content)

def firebase_post(path, value=None):
    if False:
        print('Hello World!')
    'Add an object to an existing list of data.\n\n    An HTTP POST allows an object to be added to an existing list of data.\n    A successful request will be indicated by a 200 OK HTTP status code. The\n    response content will contain a new attribute "name" which is the key for\n    the child added.\n\n    Args:\n        path - the url to the Firebase list to append to.\n        value - a json string.\n    '
    (response, content) = _get_session().post(path, body=value)
    return json.loads(content)

def firebase_get(path):
    if False:
        i = 10
        return i + 15
    'Read the data at the given path.\n\n    An HTTP GET request allows reading of data at a particular path.\n    A successful request will be indicated by a 200 OK HTTP status code.\n    The response will contain the data being retrieved.\n\n    Args:\n        path - the url to the Firebase object to read.\n    '
    (response, content) = _get_session().get(path)
    return json.loads(content)

def firebase_delete(path):
    if False:
        print('Hello World!')
    'Removes the data at a particular path.\n\n    An HTTP DELETE removes the data at a particular path.  A successful request\n    will be indicated by a 200 OK HTTP status code with a response containing\n    JSON null.\n\n    Args:\n        path - the url to the Firebase object to delete.\n    '
    (response, content) = _get_session().delete(path)