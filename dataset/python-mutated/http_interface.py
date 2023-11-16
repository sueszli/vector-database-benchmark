"""This module provides a common interface for all HTTP requests.

  HttpResponse: Represents the server's response to an HTTP request. Provides
      an interface identical to httplib.HTTPResponse which is the response
      expected from higher level classes which use HttpClient.request.

  GenericHttpClient: Provides an interface (superclass) for an object 
      responsible for making HTTP requests. Subclasses of this object are
      used in AtomService and GDataService to make requests to the server. By
      changing the http_client member object, the AtomService is able to make
      HTTP requests using different logic (for example, when running on 
      Google App Engine, the http_client makes requests using the App Engine
      urlfetch API). 
"""
import io
USER_AGENT = '%s GData-Python/2.0.18'

class Error(Exception):
    pass

class UnparsableUrlObject(Error):
    pass

class ContentLengthRequired(Error):
    pass

class HttpResponse(object):

    def __init__(self, body=None, status=None, reason=None, headers=None):
        if False:
            return 10
        "Constructor for an HttpResponse object.\n\n        HttpResponse represents the server's response to an HTTP request from\n        the client. The HttpClient.request method returns a httplib.HTTPResponse\n        object and this HttpResponse class is designed to mirror the interface\n        exposed by httplib.HTTPResponse.\n\n        Args:\n          body: A file like object, with a read() method. The body could also\n              be a string, and the constructor will wrap it so that\n              HttpResponse.read(self) will return the full string.\n          status: The HTTP status code as an int. Example: 200, 201, 404.\n          reason: The HTTP status message which follows the code. Example:\n              OK, Created, Not Found\n          headers: A dictionary containing the HTTP headers in the server's\n              response. A common header in the response is Content-Length.\n        "
        if body:
            if hasattr(body, 'read'):
                self._body = body
            else:
                self._body = io.StringIO(body)
        else:
            self._body = None
        if status is not None:
            self.status = int(status)
        else:
            self.status = None
        self.reason = reason
        self._headers = headers or {}

    def getheader(self, name, default=None):
        if False:
            while True:
                i = 10
        if name in self._headers:
            return self._headers[name]
        else:
            return default

    def read(self, amt=None):
        if False:
            while True:
                i = 10
        if not amt:
            return self._body.read()
        else:
            return self._body.read(amt)

class GenericHttpClient(object):
    debug = False

    def __init__(self, http_client, headers=None):
        if False:
            return 10
        "\n\n        Args:\n          http_client: An object which provides a request method to make an HTTP\n              request. The request method in GenericHttpClient performs a\n              call-through to the contained HTTP client object.\n          headers: A dictionary containing HTTP headers which should be included\n              in every HTTP request. Common persistent headers include\n              'User-Agent'.\n        "
        self.http_client = http_client
        self.headers = headers or {}

    def request(self, operation, url, data=None, headers=None):
        if False:
            for i in range(10):
                print('nop')
        all_headers = self.headers.copy()
        if headers:
            all_headers.update(headers)
        return self.http_client.request(operation, url, data=data, headers=all_headers)

    def get(self, url, headers=None):
        if False:
            i = 10
            return i + 15
        return self.request('GET', url, headers=headers)

    def post(self, url, data, headers=None):
        if False:
            return 10
        return self.request('POST', url, data=data, headers=headers)

    def put(self, url, data, headers=None):
        if False:
            while True:
                i = 10
        return self.request('PUT', url, data=data, headers=headers)

    def delete(self, url, headers=None):
        if False:
            for i in range(10):
                print('nop')
        return self.request('DELETE', url, headers=headers)

class GenericToken(object):
    """Represents an Authorization token to be added to HTTP requests.

    Some Authorization headers included calculated fields (digital
    signatures for example) which are based on the parameters of the HTTP
    request. Therefore the token is responsible for signing the request
    and adding the Authorization header.
    """

    def perform_request(self, http_client, operation, url, data=None, headers=None):
        if False:
            return 10
        'For the GenericToken, no Authorization token is set.'
        return http_client.request(operation, url, data=data, headers=headers)

    def valid_for_scope(self, url):
        if False:
            i = 10
            return i + 15
        "Tells the caller if the token authorizes access to the desired URL.\n\n        Since the generic token doesn't add an auth header, it is not valid for\n        any scope.\n        "
        return False