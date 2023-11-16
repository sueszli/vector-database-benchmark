"""MockService provides CRUD ops. for mocking calls to AtomPub services.

  MockService: Exposes the publicly used methods of AtomService to provide
      a mock interface which can be used in unit tests.
"""
import pickle
import atom.service
recordings = []
real_request_handler = None

def ConcealValueWithSha(source):
    if False:
        return 10
    import sha
    return sha.new(source[:-5]).hexdigest()

def DumpRecordings(conceal_func=ConcealValueWithSha):
    if False:
        for i in range(10):
            print('nop')
    if conceal_func:
        for recording_pair in recordings:
            recording_pair[0].ConcealSecrets(conceal_func)
    return pickle.dumps(recordings)

def LoadRecordings(recordings_file_or_string):
    if False:
        i = 10
        return i + 15
    if isinstance(recordings_file_or_string, str):
        atom.mock_service.recordings = pickle.loads(recordings_file_or_string)
    elif hasattr(recordings_file_or_string, 'read'):
        atom.mock_service.recordings = pickle.loads(recordings_file_or_string.read())

def HttpRequest(service, operation, data, uri, extra_headers=None, url_params=None, escape_params=True, content_type='application/atom+xml'):
    if False:
        print('Hello World!')
    "Simulates an HTTP call to the server, makes an actual HTTP request if\n    real_request_handler is set.\n\n    This function operates in two different modes depending on if\n    real_request_handler is set or not. If real_request_handler is not set,\n    HttpRequest will look in this module's recordings list to find a response\n    which matches the parameters in the function call. If real_request_handler\n    is set, this function will call real_request_handler.HttpRequest, add the\n    response to the recordings list, and respond with the actual response.\n\n    Args:\n      service: atom.AtomService object which contains some of the parameters\n          needed to make the request. The following members are used to\n          construct the HTTP call: server (str), additional_headers (dict),\n          port (int), and ssl (bool).\n      operation: str The HTTP operation to be performed. This is usually one of\n          'GET', 'POST', 'PUT', or 'DELETE'\n      data: ElementTree, filestream, list of parts, or other object which can be\n          converted to a string.\n          Should be set to None when performing a GET or PUT.\n          If data is a file-like object which can be read, this method will read\n          a chunk of 100K bytes at a time and send them.\n          If the data is a list of parts to be sent, each part will be evaluated\n          and sent.\n      uri: The beginning of the URL to which the request should be sent.\n          Examples: '/', '/base/feeds/snippets',\n          '/m8/feeds/contacts/default/base'\n      extra_headers: dict of strings. HTTP headers which should be sent\n          in the request. These headers are in addition to those stored in\n          service.additional_headers.\n      url_params: dict of strings. Key value pairs to be added to the URL as\n          URL parameters. For example {'foo':'bar', 'test':'param'} will\n          become ?foo=bar&test=param.\n      escape_params: bool default True. If true, the keys and values in\n          url_params will be URL escaped when the form is constructed\n          (Special characters converted to %XX form.)\n      content_type: str The MIME type for the data being sent. Defaults to\n          'application/atom+xml', this is only used if data is set.\n    "
    full_uri = atom.service.BuildUri(uri, url_params, escape_params)
    (server, port, ssl, uri) = atom.service.ProcessUrl(service, uri)
    current_request = MockRequest(operation, full_uri, host=server, ssl=ssl, data=data, extra_headers=extra_headers, url_params=url_params, escape_params=escape_params, content_type=content_type)
    if real_request_handler:
        response = real_request_handler.HttpRequest(service, operation, data, uri, extra_headers=extra_headers, url_params=url_params, escape_params=escape_params, content_type=content_type)
        recorded_response = MockHttpResponse(body=response.read(), status=response.status, reason=response.reason)
        recordings.append((current_request, recorded_response))
        return recorded_response
    else:
        for request_response_pair in recordings:
            if request_response_pair[0].IsMatch(current_request):
                return request_response_pair[1]
    return None

class MockRequest(object):
    """Represents a request made to an AtomPub server.

    These objects are used to determine if a client request matches a recorded
    HTTP request to determine what the mock server's response will be.
    """

    def __init__(self, operation, uri, host=None, ssl=False, port=None, data=None, extra_headers=None, url_params=None, escape_params=True, content_type='application/atom+xml'):
        if False:
            for i in range(10):
                print('nop')
        "Constructor for a MockRequest\n\n        Args:\n          operation: str One of 'GET', 'POST', 'PUT', or 'DELETE' this is the\n              HTTP operation requested on the resource.\n          uri: str The URL describing the resource to be modified or feed to be\n              retrieved. This should include the protocol (http/https) and the host\n              (aka domain). For example, these are some valud full_uris:\n              'http://example.com', 'https://www.google.com/accounts/ClientLogin'\n          host: str (optional) The server name which will be placed at the\n              beginning of the URL if the uri parameter does not begin with 'http'.\n              Examples include 'example.com', 'www.google.com', 'www.blogger.com'.\n          ssl: boolean (optional) If true, the request URL will begin with https\n              instead of http.\n          data: ElementTree, filestream, list of parts, or other object which can be\n              converted to a string. (optional)\n              Should be set to None when performing a GET or PUT.\n              If data is a file-like object which can be read, the constructor\n              will read the entire file into memory. If the data is a list of\n              parts to be sent, each part will be evaluated and stored.\n          extra_headers: dict (optional) HTTP headers included in the request.\n          url_params: dict (optional) Key value pairs which should be added to\n              the URL as URL parameters in the request. For example uri='/',\n              url_parameters={'foo':'1','bar':'2'} could become '/?foo=1&bar=2'.\n          escape_params: boolean (optional) Perform URL escaping on the keys and\n              values specified in url_params. Defaults to True.\n          content_type: str (optional) Provides the MIME type of the data being\n              sent.\n        "
        self.operation = operation
        self.uri = _ConstructFullUrlBase(uri, host=host, ssl=ssl)
        self.data = data
        self.extra_headers = extra_headers
        self.url_params = url_params or {}
        self.escape_params = escape_params
        self.content_type = content_type

    def ConcealSecrets(self, conceal_func):
        if False:
            for i in range(10):
                print('nop')
        'Conceal secret data in this request.'
        if 'Authorization' in self.extra_headers:
            self.extra_headers['Authorization'] = conceal_func(self.extra_headers['Authorization'])

    def IsMatch(self, other_request):
        if False:
            while True:
                i = 10
        'Check to see if the other_request is equivalent to this request.\n\n        Used to determine if a recording matches an incoming request so that a\n        recorded response should be sent to the client.\n\n        The matching is not exact, only the operation and URL are examined\n        currently.\n\n        Args:\n          other_request: MockRequest The request which we want to check this\n              (self) MockRequest against to see if they are equivalent.\n        '
        return self.operation == other_request.operation and self.uri == other_request.uri

def _ConstructFullUrlBase(uri, host=None, ssl=False):
    if False:
        for i in range(10):
            print('nop')
    "Puts URL components into the form http(s)://full.host.strinf/uri/path\n\n    Used to construct a roughly canonical URL so that URLs which begin with\n    'http://example.com/' can be compared to a uri of '/' when the host is\n    set to 'example.com'\n\n    If the uri contains 'http://host' already, the host and ssl parameters\n    are ignored.\n\n    Args:\n      uri: str The path component of the URL, examples include '/'\n      host: str (optional) The host name which should prepend the URL. Example:\n          'example.com'\n      ssl: boolean (optional) If true, the returned URL will begin with https\n          instead of http.\n\n    Returns:\n      String which has the form http(s)://example.com/uri/string/contents\n    "
    if uri.startswith('http'):
        return uri
    if ssl:
        return 'https://%s%s' % (host, uri)
    else:
        return 'http://%s%s' % (host, uri)

class MockHttpResponse(object):
    """Returned from MockService crud methods as the server's response."""

    def __init__(self, body=None, status=None, reason=None, headers=None):
        if False:
            while True:
                i = 10
        "Construct a mock HTTPResponse and set members.\n\n        Args:\n          body: str (optional) The HTTP body of the server's response.\n          status: int (optional)\n          reason: str (optional)\n          headers: dict (optional)\n        "
        self.body = body
        self.status = status
        self.reason = reason
        self.headers = headers or {}

    def read(self):
        if False:
            while True:
                i = 10
        return self.body

    def getheader(self, header_name):
        if False:
            for i in range(10):
                print('nop')
        return self.headers[header_name]