import io
import os.path
import pickle
import tempfile
import atom.http_core

class Error(Exception):
    pass

class NoRecordingFound(Error):
    pass

class MockHttpClient(object):
    debug = None
    real_client = None
    last_request_was_live = False
    cache_name_prefix = 'gdata_live_test'
    cache_case_name = ''
    cache_test_name = ''

    def __init__(self, recordings=None, real_client=None):
        if False:
            i = 10
            return i + 15
        self._recordings = recordings or []
        if real_client is not None:
            self.real_client = real_client

    def add_response(self, http_request, status, reason, headers=None, body=None):
        if False:
            return 10
        response = MockHttpResponse(status, reason, headers, body)
        self._recordings.append((http_request._copy(), response))
    AddResponse = add_response

    def request(self, http_request):
        if False:
            while True:
                i = 10
        'Provide a recorded response, or record a response for replay.\n\n        If the real_client is set, the request will be made using the\n        real_client, and the response from the server will be recorded.\n        If the real_client is None (the default), this method will examine\n        the recordings and find the first which matches.\n        '
        request = http_request._copy()
        _scrub_request(request)
        if self.real_client is None:
            self.last_request_was_live = False
            for recording in self._recordings:
                if _match_request(recording[0], request):
                    return recording[1]
        else:
            self.real_client.debug = self.debug
            self.last_request_was_live = True
            response = self.real_client.request(http_request)
            scrubbed_response = _scrub_response(response)
            self.add_response(request, scrubbed_response.status, scrubbed_response.reason, dict(atom.http_core.get_headers(scrubbed_response)), scrubbed_response.read())
            return self._recordings[-1][1]
        raise NoRecordingFound('No recoding was found for request: %s %s' % (request.method, str(request.uri)))
    Request = request

    def _save_recordings(self, filename):
        if False:
            return 10
        recording_file = open(os.path.join(tempfile.gettempdir(), filename), 'wb')
        pickle.dump(self._recordings, recording_file)
        recording_file.close()

    def _load_recordings(self, filename):
        if False:
            while True:
                i = 10
        recording_file = open(os.path.join(tempfile.gettempdir(), filename), 'rb')
        self._recordings = pickle.load(recording_file)
        recording_file.close()

    def _delete_recordings(self, filename):
        if False:
            i = 10
            return i + 15
        full_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(full_path):
            os.remove(full_path)

    def _load_or_use_client(self, filename, http_client):
        if False:
            i = 10
            return i + 15
        if os.path.exists(os.path.join(tempfile.gettempdir(), filename)):
            self._load_recordings(filename)
        else:
            self.real_client = http_client

    def use_cached_session(self, name=None, real_http_client=None):
        if False:
            return 10
        'Attempts to load recordings from a previous live request.\n\n        If a temp file with the recordings exists, then it is used to fulfill\n        requests. If the file does not exist, then a real client is used to\n        actually make the desired HTTP requests. Requests and responses are\n        recorded and will be written to the desired temprary cache file when\n        close_session is called.\n\n        Args:\n          name: str (optional) The file name of session file to be used. The file\n                is loaded from the temporary directory of this machine. If no name\n                is passed in, a default name will be constructed using the\n                cache_name_prefix, cache_case_name, and cache_test_name of this\n                object.\n          real_http_client: atom.http_core.HttpClient the real client to be used\n                            if the cached recordings are not found. If the default\n                            value is used, this will be an\n                            atom.http_core.HttpClient.\n        '
        if real_http_client is None:
            real_http_client = atom.http_core.HttpClient()
        if name is None:
            self._recordings_cache_name = self.get_cache_file_name()
        else:
            self._recordings_cache_name = name
        self._load_or_use_client(self._recordings_cache_name, real_http_client)

    def close_session(self):
        if False:
            return 10
        'Saves recordings in the temporary file named in use_cached_session.'
        if self.real_client is not None:
            self._save_recordings(self._recordings_cache_name)

    def delete_session(self, name=None):
        if False:
            while True:
                i = 10
        'Removes recordings from a previous live request.'
        if name is None:
            self._delete_recordings(self._recordings_cache_name)
        else:
            self._delete_recordings(name)

    def get_cache_file_name(self):
        if False:
            i = 10
            return i + 15
        return '%s.%s.%s' % (self.cache_name_prefix, self.cache_case_name, self.cache_test_name)

    def _dump(self):
        if False:
            while True:
                i = 10
        'Provides debug information in a string.'
        output = 'MockHttpClient\n  real_client: %s\n  cache file name: %s\n' % (self.real_client, self.get_cache_file_name())
        output += '  recordings:\n'
        i = 0
        for recording in self._recordings:
            output += '    recording %i is for: %s %s\n' % (i, recording[0].method, str(recording[0].uri))
            i += 1
        return output

def _match_request(http_request, stored_request):
    if False:
        i = 10
        return i + 15
    'Determines whether a request is similar enough to a stored request\n       to cause the stored response to be returned.'
    if http_request.uri.host is not None and http_request.uri.host != stored_request.uri.host:
        return False
    elif http_request.uri.path != stored_request.uri.path:
        return False
    elif http_request.method != stored_request.method:
        return False
    elif 'gsessionid' in http_request.uri.query or 'gsessionid' in stored_request.uri.query:
        if 'gsessionid' not in stored_request.uri.query:
            return False
        elif 'gsessionid' not in http_request.uri.query:
            return False
        elif http_request.uri.query['gsessionid'] != stored_request.uri.query['gsessionid']:
            return False
    return True

def _scrub_request(http_request):
    if False:
        for i in range(10):
            print('nop')
    ' Removes email address and password from a client login request.\n\n    Since the mock server saves the request and response in plantext, sensitive\n    information like the password should be removed before saving the\n    recordings. At the moment only requests sent to a ClientLogin url are\n    scrubbed.\n    '
    if http_request and http_request.uri and http_request.uri.path and http_request.uri.path.endswith('ClientLogin'):
        http_request._body_parts = []
        http_request.add_form_inputs({'form_data': 'client login request has been scrubbed'})
    else:
        http_request._body_parts = []
    return http_request

def _scrub_response(http_response):
    if False:
        for i in range(10):
            print('nop')
    return http_response

class EchoHttpClient(object):
    """Sends the request data back in the response.

    Used to check the formatting of the request as it was sent. Always responds
    with a 200 OK, and some information from the HTTP request is returned in
    special Echo-X headers in the response. The following headers are added
    in the response:
    'Echo-Host': The host name and port number to which the HTTP connection is
                 made. If no port was passed in, the header will contain
                 host:None.
    'Echo-Uri': The path portion of the URL being requested. /example?x=1&y=2
    'Echo-Scheme': The beginning of the URL, usually 'http' or 'https'
    'Echo-Method': The HTTP method being used, 'GET', 'POST', 'PUT', etc.
    """

    def request(self, http_request):
        if False:
            return 10
        return self._http_request(http_request.uri, http_request.method, http_request.headers, http_request._body_parts)

    def _http_request(self, uri, method, headers=None, body_parts=None):
        if False:
            i = 10
            return i + 15
        body = io.StringIO()
        response = atom.http_core.HttpResponse(status=200, reason='OK', body=body)
        if headers is None:
            response._headers = {}
        else:
            for (header, value) in headers.items():
                response._headers[header] = str(value)
        response._headers['Echo-Host'] = '%s:%s' % (uri.host, str(uri.port))
        response._headers['Echo-Uri'] = uri._get_relative_path()
        response._headers['Echo-Scheme'] = uri.scheme
        response._headers['Echo-Method'] = method
        for part in body_parts:
            if isinstance(part, str):
                body.write(part)
            elif hasattr(part, 'read'):
                body.write(part.read())
        body.seek(0)
        return response

class SettableHttpClient(object):
    """An HTTP Client which responds with the data given in set_response."""

    def __init__(self, status, reason, body, headers):
        if False:
            return 10
        'Configures the response for the server.\n\n        See set_response for details on the arguments to the constructor.\n        '
        self.set_response(status, reason, body, headers)
        self.last_request = None

    def set_response(self, status, reason, body, headers):
        if False:
            i = 10
            return i + 15
        'Determines the response which will be sent for each request.\n\n        Args:\n          status: An int for the HTTP status code, example: 200, 404, etc.\n          reason: String for the HTTP reason, example: OK, NOT FOUND, etc.\n          body: The body of the HTTP response as a string or a file-like\n                object (something with a read method).\n          headers: dict of strings containing the HTTP headers in the response.\n        '
        self.response = atom.http_core.HttpResponse(status=status, reason=reason, body=body)
        self.response._headers = headers.copy()

    def request(self, http_request):
        if False:
            return 10
        self.last_request = http_request
        return self.response

class MockHttpResponse(atom.http_core.HttpResponse):

    def __init__(self, status=None, reason=None, headers=None, body=None):
        if False:
            print('Hello World!')
        self._headers = headers or {}
        if status is not None:
            self.status = status
        if reason is not None:
            self.reason = reason
        if body is not None:
            if hasattr(body, 'read'):
                self._body = body.read()
            else:
                self._body = body

    def read(self):
        if False:
            i = 10
            return i + 15
        return self._body