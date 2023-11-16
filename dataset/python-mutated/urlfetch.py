"""Provides HTTP functions for gdata.service to use on Google App Engine

AppEngineHttpClient: Provides an HTTP request method which uses App Engine's
   urlfetch API. Set the http_client member of a GDataService object to an
   instance of an AppEngineHttpClient to allow the gdata library to run on
   Google App Engine.

run_on_appengine: Function which will modify an existing GDataService object
   to allow it to run on App Engine. It works by creating a new instance of
   the AppEngineHttpClient and replacing the GDataService object's 
   http_client.

HttpRequest: Function that wraps google.appengine.api.urlfetch.Fetch in a 
    common interface which is used by gdata.service.GDataService. In other 
    words, this module can be used as the gdata service request handler so 
    that all HTTP requests will be performed by the hosting Google App Engine
    server. 
"""
__author__ = 'api.jscudder (Jeff Scudder)'
import io
import atom.service
import atom.http_interface
from google.appengine.api import urlfetch

def run_on_appengine(gdata_service):
    if False:
        for i in range(10):
            print('nop')
    'Modifies a GDataService object to allow it to run on App Engine.\n\n  Args:\n    gdata_service: An instance of AtomService, GDataService, or any\n        of their subclasses which has an http_client member.\n  '
    gdata_service.http_client = AppEngineHttpClient()

class AppEngineHttpClient(atom.http_interface.GenericHttpClient):

    def __init__(self, headers=None):
        if False:
            for i in range(10):
                print('nop')
        self.debug = False
        self.headers = headers or {}

    def request(self, operation, url, data=None, headers=None):
        if False:
            i = 10
            return i + 15
        "Performs an HTTP call to the server, supports GET, POST, PUT, and\n    DELETE.\n\n    Usage example, perform and HTTP GET on http://www.google.com/:\n      import atom.http\n      client = atom.http.HttpClient()\n      http_response = client.request('GET', 'http://www.google.com/')\n\n    Args:\n      operation: str The HTTP operation to be performed. This is usually one\n          of 'GET', 'POST', 'PUT', or 'DELETE'\n      data: filestream, list of parts, or other object which can be converted\n          to a string. Should be set to None when performing a GET or DELETE.\n          If data is a file-like object which can be read, this method will\n          read a chunk of 100K bytes at a time and send them.\n          If the data is a list of parts to be sent, each part will be\n          evaluated and sent.\n      url: The full URL to which the request should be sent. Can be a string\n          or atom.url.Url.\n      headers: dict of strings. HTTP headers which should be sent\n          in the request.\n    "
        all_headers = self.headers.copy()
        if headers:
            all_headers.update(headers)
        data_str = data
        if data:
            if isinstance(data, list):
                converted_parts = [__ConvertDataPart(x) for x in data]
                data_str = ''.join(converted_parts)
            else:
                data_str = __ConvertDataPart(data)
        if data and 'Content-Length' not in all_headers:
            all_headers['Content-Length'] = len(data_str)
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
        return HttpResponse(urlfetch.Fetch(url=str(url), payload=data_str, method=method, headers=all_headers))

def HttpRequest(service, operation, data, uri, extra_headers=None, url_params=None, escape_params=True, content_type='application/atom+xml'):
    if False:
        return 10
    "Performs an HTTP call to the server, supports GET, POST, PUT, and DELETE.\n\n  This function is deprecated, use AppEngineHttpClient.request instead.\n\n  To use this module with gdata.service, you can set this module to be the\n  http_request_handler so that HTTP requests use Google App Engine's urlfetch.\n  import gdata.service\n  import gdata.urlfetch\n  gdata.service.http_request_handler = gdata.urlfetch\n\n  Args:\n    service: atom.AtomService object which contains some of the parameters\n        needed to make the request. The following members are used to\n        construct the HTTP call: server (str), additional_headers (dict),\n        port (int), and ssl (bool).\n    operation: str The HTTP operation to be performed. This is usually one of\n        'GET', 'POST', 'PUT', or 'DELETE'\n    data: filestream, list of parts, or other object which can be\n        converted to a string.\n        Should be set to None when performing a GET or PUT.\n        If data is a file-like object which can be read, this method will read\n        a chunk of 100K bytes at a time and send them.\n        If the data is a list of parts to be sent, each part will be evaluated\n        and sent.\n    uri: The beginning of the URL to which the request should be sent.\n        Examples: '/', '/base/feeds/snippets',\n        '/m8/feeds/contacts/default/base'\n    extra_headers: dict of strings. HTTP headers which should be sent\n        in the request. These headers are in addition to those stored in\n        service.additional_headers.\n    url_params: dict of strings. Key value pairs to be added to the URL as\n        URL parameters. For example {'foo':'bar', 'test':'param'} will\n        become ?foo=bar&test=param.\n    escape_params: bool default True. If true, the keys and values in\n        url_params will be URL escaped when the form is constructed\n        (Special characters converted to %XX form.)\n    content_type: str The MIME type for the data being sent. Defaults to\n        'application/atom+xml', this is only used if data is set.\n  "
    full_uri = atom.service.BuildUri(uri, url_params, escape_params)
    (server, port, ssl, partial_uri) = atom.service.ProcessUrl(service, full_uri)
    if ssl:
        full_url = 'https://%s%s' % (server, partial_uri)
    else:
        full_url = 'http://%s%s' % (server, partial_uri)
    data_str = data
    if data:
        if isinstance(data, list):
            converted_parts = [__ConvertDataPart(x) for x in data]
            data_str = ''.join(converted_parts)
        else:
            data_str = __ConvertDataPart(data)
    headers = {}
    if isinstance(service.additional_headers, dict):
        headers = service.additional_headers.copy()
    if isinstance(extra_headers, dict):
        for (header, value) in extra_headers.items():
            headers[header] = value
    if content_type:
        headers['Content-Type'] = content_type
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
    return HttpResponse(urlfetch.Fetch(url=full_url, payload=data_str, method=method, headers=headers))

def __ConvertDataPart(data):
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
            return 10
        self.body = io.StringIO(urlfetch_response.content)
        self.headers = urlfetch_response.headers
        self.status = urlfetch_response.status_code
        self.reason = ''

    def read(self, length=None):
        if False:
            return 10
        if not length:
            return self.body.read()
        else:
            return self.body.read(length)

    def getheader(self, name):
        if False:
            print('Hello World!')
        if name not in self.headers:
            return self.headers[name.lower()]
        return self.headers[name]