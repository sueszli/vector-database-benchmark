import atom.http_interface
import atom.url

class Error(Exception):
    pass

class NoRecordingFound(Error):
    pass

class MockRequest(object):
    """Holds parameters of an HTTP request for matching against future requests.
    """

    def __init__(self, operation, url, data=None, headers=None):
        if False:
            return 10
        self.operation = operation
        if isinstance(url, str):
            url = atom.url.parse_url(url)
        self.url = url
        self.data = data
        self.headers = headers

class MockResponse(atom.http_interface.HttpResponse):
    """Simulates an httplib.HTTPResponse object."""

    def __init__(self, body=None, status=None, reason=None, headers=None):
        if False:
            i = 10
            return i + 15
        if body and hasattr(body, 'read'):
            self.body = body.read()
        else:
            self.body = body
        if status is not None:
            self.status = int(status)
        else:
            self.status = None
        self.reason = reason
        self._headers = headers or {}

    def read(self):
        if False:
            while True:
                i = 10
        return self.body

class MockHttpClient(atom.http_interface.GenericHttpClient):

    def __init__(self, headers=None, recordings=None, real_client=None):
        if False:
            print('Hello World!')
        "An HttpClient which responds to request with stored data.\n\n        The request-response pairs are stored as tuples in a member list named\n        recordings.\n\n        The MockHttpClient can be switched from replay mode to record mode by\n        setting the real_client member to an instance of an HttpClient which will\n        make real HTTP requests and store the server's response in list of\n        recordings.\n\n        Args:\n          headers: dict containing HTTP headers which should be included in all\n              HTTP requests.\n          recordings: The initial recordings to be used for responses. This list\n              contains tuples in the form: (MockRequest, MockResponse)\n          real_client: An HttpClient which will make a real HTTP request. The\n              response will be converted into a MockResponse and stored in\n              recordings.\n        "
        self.recordings = recordings or []
        self.real_client = real_client
        self.headers = headers or {}

    def add_response(self, response, operation, url, data=None, headers=None):
        if False:
            while True:
                i = 10
        'Adds a request-response pair to the recordings list.\n\n        After the recording is added, future matching requests will receive the\n        response.\n\n        Args:\n          response: MockResponse\n          operation: str\n          url: str\n          data: str, Currently the data is ignored when looking for matching\n              requests.\n          headers: dict of strings: Currently the headers are ignored when\n              looking for matching requests.\n        '
        request = MockRequest(operation, url, data=data, headers=headers)
        self.recordings.append((request, response))

    def request(self, operation, url, data=None, headers=None):
        if False:
            print('Hello World!')
        "Returns a matching MockResponse from the recordings.\n\n        If the real_client is set, the request will be passed along and the\n        server's response will be added to the recordings and also returned.\n\n        If there is no match, a NoRecordingFound error will be raised.\n        "
        if self.real_client is None:
            if isinstance(url, str):
                url = atom.url.parse_url(url)
            for recording in self.recordings:
                if recording[0].operation == operation and recording[0].url == url:
                    return recording[1]
            raise NoRecordingFound('No recodings found for %s %s' % (operation, url))
        else:
            response = self.real_client.request(operation, url, data=data, headers=headers)
            stored_response = MockResponse(body=response, status=response.status, reason=response.reason)
            self.add_response(stored_response, operation, url, data=data, headers=headers)
            return stored_response