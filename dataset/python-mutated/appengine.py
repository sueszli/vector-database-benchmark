"""Provides HTTP functions for gdata.service to use on Google App Engine

AppEngineHttpClient: Provides an HTTP request method which uses App Engine's
   urlfetch API. Set the http_client member of a GDataService object to an
   instance of an AppEngineHttpClient to allow the gdata library to run on
   Google App Engine.

run_on_appengine: Function which will modify an existing GDataService object
   to allow it to run on App Engine. It works by creating a new instance of
   the AppEngineHttpClient and replacing the GDataService object's
   http_client.
"""
__author__ = 'api.jscudder (Jeff Scudder)'
import StringIO
import pickle
import atom.http_interface
import atom.token_store
from google.appengine.api import urlfetch
from google.appengine.ext import db
from google.appengine.api import users
from google.appengine.api import memcache

def run_on_appengine(gdata_service, store_tokens=True, single_user_mode=False, deadline=None):
    if False:
        i = 10
        return i + 15
    "Modifies a GDataService object to allow it to run on App Engine.\n\n  Args:\n    gdata_service: An instance of AtomService, GDataService, or any\n        of their subclasses which has an http_client member and a \n        token_store member.\n    store_tokens: Boolean, defaults to True. If True, the gdata_service\n                  will attempt to add each token to it's token_store when\n                  SetClientLoginToken or SetAuthSubToken is called. If False\n                  the tokens will not automatically be added to the \n                  token_store.\n    single_user_mode: Boolean, defaults to False. If True, the current_token\n                      member of gdata_service will be set when \n                      SetClientLoginToken or SetAuthTubToken is called. If set\n                      to True, the current_token is set in the gdata_service\n                      and anyone who accesses the object will use the same \n                      token. \n                      \n                      Note: If store_tokens is set to False and \n                      single_user_mode is set to False, all tokens will be \n                      ignored, since the library assumes: the tokens should not\n                      be stored in the datastore and they should not be stored\n                      in the gdata_service object. This will make it \n                      impossible to make requests which require authorization.\n    deadline: int (optional) The number of seconds to wait for a response\n              before timing out on the HTTP request. If no deadline is\n              specified, the deafault deadline for HTTP requests from App\n              Engine is used. The maximum is currently 10 (for 10 seconds).\n              The default deadline for App Engine is 5 seconds.\n  "
    gdata_service.http_client = AppEngineHttpClient(deadline=deadline)
    gdata_service.token_store = AppEngineTokenStore()
    gdata_service.auto_store_tokens = store_tokens
    gdata_service.auto_set_current_token = single_user_mode
    return gdata_service

class AppEngineHttpClient(atom.http_interface.GenericHttpClient):

    def __init__(self, headers=None, deadline=None):
        if False:
            while True:
                i = 10
        self.debug = False
        self.headers = headers or {}
        self.deadline = deadline

    def request(self, operation, url, data=None, headers=None):
        if False:
            return 10
        "Performs an HTTP call to the server, supports GET, POST, PUT, and\n    DELETE.\n\n    Usage example, perform and HTTP GET on http://www.google.com/:\n      import atom.http\n      client = atom.http.HttpClient()\n      http_response = client.request('GET', 'http://www.google.com/')\n\n    Args:\n      operation: str The HTTP operation to be performed. This is usually one\n          of 'GET', 'POST', 'PUT', or 'DELETE'\n      data: filestream, list of parts, or other object which can be converted\n          to a string. Should be set to None when performing a GET or DELETE.\n          If data is a file-like object which can be read, this method will\n          read a chunk of 100K bytes at a time and send them.\n          If the data is a list of parts to be sent, each part will be\n          evaluated and sent.\n      url: The full URL to which the request should be sent. Can be a string\n          or atom.url.Url.\n      headers: dict of strings. HTTP headers which should be sent\n          in the request.\n    "
        all_headers = self.headers.copy()
        if headers:
            all_headers.update(headers)
        data_str = data
        if data:
            if isinstance(data, list):
                converted_parts = [_convert_data_part(x) for x in data]
                data_str = ''.join(converted_parts)
            else:
                data_str = _convert_data_part(data)
        if data and 'Content-Length' not in all_headers:
            all_headers['Content-Length'] = str(len(data_str))
        if 'Content-Type' not in all_headers:
            all_headers['Content-Type'] = 'application/atom+xml'
        if operation == 'GET':
            method = urlfetch.GET
        elif operation == 'POST':
            method = urlfetch.POST
        elif operation == 'PUT':
            method = urlfetch.PUT
        elif operation == 'DELETE':
            method = urlfetch.DELETE
        else:
            method = None
        if self.deadline is None:
            return HttpResponse(urlfetch.Fetch(url=str(url), payload=data_str, method=method, headers=all_headers, follow_redirects=False))
        return HttpResponse(urlfetch.Fetch(url=str(url), payload=data_str, method=method, headers=all_headers, follow_redirects=False, deadline=self.deadline))

def _convert_data_part(data):
    if False:
        return 10
    if not data or isinstance(data, str):
        return data
    elif hasattr(data, 'read'):
        return data.read()
    return str(data)

class HttpResponse(object):
    """Translates a urlfetch resoinse to look like an hhtplib resoinse.

  Used to allow the resoinse from HttpRequest to be usable by gdata.service
  methods.
  """

    def __init__(self, urlfetch_response):
        if False:
            print('Hello World!')
        self.body = StringIO.StringIO(urlfetch_response.content)
        self.headers = urlfetch_response.headers
        self.status = urlfetch_response.status_code
        self.reason = ''

    def read(self, length=None):
        if False:
            print('Hello World!')
        if not length:
            return self.body.read()
        else:
            return self.body.read(length)

    def getheader(self, name):
        if False:
            while True:
                i = 10
        if not self.headers.has_key(name):
            return self.headers[name.lower()]
        return self.headers[name]

class TokenCollection(db.Model):
    """Datastore Model which associates auth tokens with the current user."""
    user = db.UserProperty()
    pickled_tokens = db.BlobProperty()

class AppEngineTokenStore(atom.token_store.TokenStore):
    """Stores the user's auth tokens in the App Engine datastore.

  Tokens are only written to the datastore if a user is signed in (if 
  users.get_current_user() returns a user object).
  """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.user = None

    def add_token(self, token):
        if False:
            i = 10
            return i + 15
        'Associates the token with the current user and stores it.\n    \n    If there is no current user, the token will not be stored.\n\n    Returns:\n      False if the token was not stored. \n    '
        tokens = load_auth_tokens(self.user)
        if not hasattr(token, 'scopes') or not token.scopes:
            return False
        for scope in token.scopes:
            tokens[str(scope)] = token
        key = save_auth_tokens(tokens, self.user)
        if key:
            return True
        return False

    def find_token(self, url):
        if False:
            while True:
                i = 10
        "Searches the current user's collection of token for a token which can\n    be used for a request to the url.\n\n    Returns:\n      The stored token which belongs to the current user and is valid for the\n      desired URL. If there is no current user, or there is no valid user \n      token in the datastore, a atom.http_interface.GenericToken is returned.\n    "
        if url is None:
            return None
        if isinstance(url, (str, unicode)):
            url = atom.url.parse_url(url)
        tokens = load_auth_tokens(self.user)
        if url in tokens:
            token = tokens[url]
            if token.valid_for_scope(url):
                return token
            else:
                del tokens[url]
                save_auth_tokens(tokens, self.user)
        for (scope, token) in tokens.iteritems():
            if token.valid_for_scope(url):
                return token
        return atom.http_interface.GenericToken()

    def remove_token(self, token):
        if False:
            return 10
        "Removes the token from the current user's collection in the datastore.\n    \n    Returns:\n      False if the token was not removed, this could be because the token was\n      not in the datastore, or because there is no current user.\n    "
        token_found = False
        scopes_to_delete = []
        tokens = load_auth_tokens(self.user)
        for (scope, stored_token) in tokens.iteritems():
            if stored_token == token:
                scopes_to_delete.append(scope)
                token_found = True
        for scope in scopes_to_delete:
            del tokens[scope]
        if token_found:
            save_auth_tokens(tokens, self.user)
        return token_found

    def remove_all_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        "Removes all of the current user's tokens from the datastore."
        save_auth_tokens({}, self.user)

def save_auth_tokens(token_dict, user=None):
    if False:
        print('Hello World!')
    "Associates the tokens with the current user and writes to the datastore.\n  \n  If there us no current user, the tokens are not written and this function\n  returns None.\n\n  Returns:\n    The key of the datastore entity containing the user's tokens, or None if\n    there was no current user.\n  "
    if user is None:
        user = users.get_current_user()
    if user is None:
        return None
    memcache.set('gdata_pickled_tokens:%s' % user, pickle.dumps(token_dict))
    user_tokens = TokenCollection.all().filter('user =', user).get()
    if user_tokens:
        user_tokens.pickled_tokens = pickle.dumps(token_dict)
        return user_tokens.put()
    else:
        user_tokens = TokenCollection(user=user, pickled_tokens=pickle.dumps(token_dict))
        return user_tokens.put()

def load_auth_tokens(user=None):
    if False:
        while True:
            i = 10
    "Reads a dictionary of the current user's tokens from the datastore.\n  \n  If there is no current user (a user is not signed in to the app) or the user\n  does not have any tokens, an empty dictionary is returned.\n  "
    if user is None:
        user = users.get_current_user()
    if user is None:
        return {}
    pickled_tokens = memcache.get('gdata_pickled_tokens:%s' % user)
    if pickled_tokens:
        return pickle.loads(pickled_tokens)
    user_tokens = TokenCollection.all().filter('user =', user).get()
    if user_tokens:
        memcache.set('gdata_pickled_tokens:%s' % user, user_tokens.pickled_tokens)
        return pickle.loads(user_tokens.pickled_tokens)
    return {}