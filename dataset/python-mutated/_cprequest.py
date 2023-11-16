import sys
import time
import collections
import operator
from http.cookies import SimpleCookie, CookieError
import uuid
from more_itertools import consume
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy import _cpreqbody
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil, reprconf, encoding

class Hook(object):
    """A callback and its metadata: failsafe, priority, and kwargs."""
    callback = None
    '\n    The bare callable that this Hook object is wrapping, which will\n    be called when the Hook is called.'
    failsafe = False
    '\n    If True, the callback is guaranteed to run even if other callbacks\n    from the same call point raise exceptions.'
    priority = 50
    '\n    Defines the order of execution for a list of Hooks. Priority numbers\n    should be limited to the closed interval [0, 100], but values outside\n    this range are acceptable, as are fractional values.'
    kwargs = {}
    '\n    A set of keyword arguments that will be passed to the\n    callable on each call.'

    def __init__(self, callback, failsafe=None, priority=None, **kwargs):
        if False:
            while True:
                i = 10
        self.callback = callback
        if failsafe is None:
            failsafe = getattr(callback, 'failsafe', False)
        self.failsafe = failsafe
        if priority is None:
            priority = getattr(callback, 'priority', 50)
        self.priority = priority
        self.kwargs = kwargs

    def __lt__(self, other):
        if False:
            return 10
        '\n        Hooks sort by priority, ascending, such that\n        hooks of lower priority are run first.\n        '
        return self.priority < other.priority

    def __call__(self):
        if False:
            while True:
                i = 10
        'Run self.callback(**self.kwargs).'
        return self.callback(**self.kwargs)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        cls = self.__class__
        return '%s.%s(callback=%r, failsafe=%r, priority=%r, %s)' % (cls.__module__, cls.__name__, self.callback, self.failsafe, self.priority, ', '.join(['%s=%r' % (k, v) for (k, v) in self.kwargs.items()]))

class HookMap(dict):
    """A map of call points to lists of callbacks (Hook objects)."""

    def __new__(cls, points=None):
        if False:
            return 10
        d = dict.__new__(cls)
        for p in points or []:
            d[p] = []
        return d

    def __init__(self, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        pass

    def attach(self, point, callback, failsafe=None, priority=None, **kwargs):
        if False:
            while True:
                i = 10
        'Append a new Hook made from the supplied arguments.'
        self[point].append(Hook(callback, failsafe, priority, **kwargs))

    def run(self, point):
        if False:
            while True:
                i = 10
        'Execute all registered Hooks (callbacks) for the given point.'
        self.run_hooks(iter(sorted(self[point])))

    @classmethod
    def run_hooks(cls, hooks):
        if False:
            for i in range(10):
                print('nop')
        'Execute the indicated hooks, trapping errors.\n\n        Hooks with ``.failsafe == True`` are guaranteed to run\n        even if others at the same hookpoint fail. In this case,\n        log the failure and proceed on to the next hook. The only\n        way to stop all processing from one of these hooks is\n        to raise a BaseException like SystemExit or\n        KeyboardInterrupt and stop the whole server.\n        '
        assert isinstance(hooks, collections.abc.Iterator)
        quiet_errors = (cherrypy.HTTPError, cherrypy.HTTPRedirect, cherrypy.InternalRedirect)
        safe = filter(operator.attrgetter('failsafe'), hooks)
        for hook in hooks:
            try:
                hook()
            except quiet_errors:
                cls.run_hooks(safe)
                raise
            except Exception:
                cherrypy.log(traceback=True, severity=40)
                cls.run_hooks(safe)
                raise

    def __copy__(self):
        if False:
            while True:
                i = 10
        newmap = self.__class__()
        for (k, v) in self.items():
            newmap[k] = v[:]
        return newmap
    copy = __copy__

    def __repr__(self):
        if False:
            return 10
        cls = self.__class__
        return '%s.%s(points=%r)' % (cls.__module__, cls.__name__, list(self))

def hooks_namespace(k, v):
    if False:
        for i in range(10):
            print('nop')
    'Attach bare hooks declared in config.'
    hookpoint = k.split('.', 1)[0]
    if isinstance(v, str):
        v = cherrypy.lib.reprconf.attributes(v)
    if not isinstance(v, Hook):
        v = Hook(v)
    cherrypy.serving.request.hooks[hookpoint].append(v)

def request_namespace(k, v):
    if False:
        return 10
    'Attach request attributes declared in config.'
    if k[:5] == 'body.':
        setattr(cherrypy.serving.request.body, k[5:], v)
    else:
        setattr(cherrypy.serving.request, k, v)

def response_namespace(k, v):
    if False:
        while True:
            i = 10
    'Attach response attributes declared in config.'
    if k[:8] == 'headers.':
        cherrypy.serving.response.headers[k.split('.', 1)[1]] = v
    else:
        setattr(cherrypy.serving.response, k, v)

def error_page_namespace(k, v):
    if False:
        print('Hello World!')
    'Attach error pages declared in config.'
    if k != 'default':
        k = int(k)
    cherrypy.serving.request.error_page[k] = v
hookpoints = ['on_start_resource', 'before_request_body', 'before_handler', 'before_finalize', 'on_end_resource', 'on_end_request', 'before_error_response', 'after_error_response']

class Request(object):
    """An HTTP request.

    This object represents the metadata of an HTTP request message;
    that is, it contains attributes which describe the environment
    in which the request URL, headers, and body were sent (if you
    want tools to interpret the headers and body, those are elsewhere,
    mostly in Tools). This 'metadata' consists of socket data,
    transport characteristics, and the Request-Line. This object
    also contains data regarding the configuration in effect for
    the given URL, and the execution plan for generating a response.
    """
    prev = None
    '\n    The previous Request object (if any). This should be None\n    unless we are processing an InternalRedirect.'
    local = httputil.Host('127.0.0.1', 80)
    'An httputil.Host(ip, port, hostname) object for the server socket.'
    remote = httputil.Host('127.0.0.1', 1111)
    'An httputil.Host(ip, port, hostname) object for the client socket.'
    scheme = 'http'
    "\n    The protocol used between client and server. In most cases,\n    this will be either 'http' or 'https'."
    server_protocol = 'HTTP/1.1'
    '\n    The HTTP version for which the HTTP server is at least\n    conditionally compliant.'
    base = ''
    "The (scheme://host) portion of the requested URL.\n    In some cases (e.g. when proxying via mod_rewrite), this may contain\n    path segments which cherrypy.url uses when constructing url's, but\n    which otherwise are ignored by CherryPy. Regardless, this value\n    MUST NOT end in a slash."
    request_line = ''
    '\n    The complete Request-Line received from the client. This is a\n    single string consisting of the request method, URI, and protocol\n    version (joined by spaces). Any final CRLF is removed.'
    method = 'GET'
    '\n    Indicates the HTTP method to be performed on the resource identified\n    by the Request-URI. Common methods include GET, HEAD, POST, PUT, and\n    DELETE. CherryPy allows any extension method; however, various HTTP\n    servers and gateways may restrict the set of allowable methods.\n    CherryPy applications SHOULD restrict the set (on a per-URI basis).'
    query_string = ''
    "\n    The query component of the Request-URI, a string of information to be\n    interpreted by the resource. The query portion of a URI follows the\n    path component, and is separated by a '?'. For example, the URI\n    'http://www.cherrypy.dev/wiki?a=3&b=4' has the query component,\n    'a=3&b=4'."
    query_string_encoding = 'utf8'
    "\n    The encoding expected for query string arguments after % HEX HEX decoding).\n    If a query string is provided that cannot be decoded with this encoding,\n    404 is raised (since technically it's a different URI). If you want\n    arbitrary encodings to not error, set this to 'Latin-1'; you can then\n    encode back to bytes and re-decode to whatever encoding you like later.\n    "
    protocol = (1, 1)
    "The HTTP protocol version corresponding to the set\n    of features which should be allowed in the response. If BOTH\n    the client's request message AND the server's level of HTTP\n    compliance is HTTP/1.1, this attribute will be the tuple (1, 1).\n    If either is 1.0, this attribute will be the tuple (1, 0).\n    Lower HTTP protocol versions are not explicitly supported."
    params = {}
    "\n    A dict which combines query string (GET) and request entity (POST)\n    variables. This is populated in two stages: GET params are added\n    before the 'on_start_resource' hook, and POST params are added\n    between the 'before_request_body' and 'before_handler' hooks."
    header_list = []
    '\n    A list of the HTTP request headers as (name, value) tuples.\n    In general, you should use request.headers (a dict) instead.'
    headers = httputil.HeaderMap()
    "\n    A dict-like object containing the request headers. Keys are header\n    names (in Title-Case format); however, you may get and set them in\n    a case-insensitive manner. That is, headers['Content-Type'] and\n    headers['content-type'] refer to the same value. Values are header\n    values (decoded according to :rfc:`2047` if necessary). See also:\n    httputil.HeaderMap, httputil.HeaderElement."
    cookie = SimpleCookie()
    'See help(Cookie).'
    rfile = None
    "\n    If the request included an entity (body), it will be available\n    as a stream in this attribute. However, the rfile will normally\n    be read for you between the 'before_request_body' hook and the\n    'before_handler' hook, and the resulting string is placed into\n    either request.params or the request.body attribute.\n\n    You may disable the automatic consumption of the rfile by setting\n    request.process_request_body to False, either in config for the desired\n    path, or in an 'on_start_resource' or 'before_request_body' hook.\n\n    WARNING: In almost every case, you should not attempt to read from the\n    rfile stream after CherryPy's automatic mechanism has read it. If you\n    turn off the automatic parsing of rfile, you should read exactly the\n    number of bytes specified in request.headers['Content-Length'].\n    Ignoring either of these warnings may result in a hung request thread\n    or in corruption of the next (pipelined) request.\n    "
    process_request_body = True
    '\n    If True, the rfile (if any) is automatically read and parsed,\n    and the result placed into request.params or request.body.'
    methods_with_bodies = ('POST', 'PUT', 'PATCH')
    '\n    A sequence of HTTP methods for which CherryPy will automatically\n    attempt to read a body from the rfile. If you are going to change\n    this property, modify it on the configuration (recommended)\n    or on the "hook point" `on_start_resource`.\n    '
    body = None
    "\n    If the request Content-Type is 'application/x-www-form-urlencoded'\n    or multipart, this will be None. Otherwise, this will be an instance\n    of :class:`RequestBody<cherrypy._cpreqbody.RequestBody>` (which you\n    can .read()); this value is set between the 'before_request_body' and\n    'before_handler' hooks (assuming that process_request_body is True)."
    dispatch = cherrypy.dispatch.Dispatcher()
    "\n    The object which looks up the 'page handler' callable and collects\n    config for the current request based on the path_info, other\n    request attributes, and the application architecture. The core\n    calls the dispatcher as early as possible, passing it a 'path_info'\n    argument.\n\n    The default dispatcher discovers the page handler by matching path_info\n    to a hierarchical arrangement of objects, starting at request.app.root.\n    See help(cherrypy.dispatch) for more information."
    script_name = ''
    '\n    The \'mount point\' of the application which is handling this request.\n\n    This attribute MUST NOT end in a slash. If the script_name refers to\n    the root of the URI, it MUST be an empty string (not "/").\n    '
    path_info = '/'
    "\n    The 'relative path' portion of the Request-URI. This is relative\n    to the script_name ('mount point') of the application which is\n    handling this request."
    login = None
    "\n    When authentication is used during the request processing this is\n    set to 'False' if it failed and to the 'username' value if it succeeded.\n    The default 'None' implies that no authentication happened."
    app = None
    'The cherrypy.Application object which is handling this request.'
    handler = None
    '\n    The function, method, or other callable which CherryPy will call to\n    produce the response. The discovery of the handler and the arguments\n    it will receive are determined by the request.dispatch object.\n    By default, the handler is discovered by walking a tree of objects\n    starting at request.app.root, and is then passed all HTTP params\n    (from the query string and POST body) as keyword arguments.'
    toolmaps = {}
    '\n    A nested dict of all Toolboxes and Tools in effect for this request,\n    of the form: {Toolbox.namespace: {Tool.name: config dict}}.'
    config = None
    '\n    A flat dict of all configuration entries which apply to the\n    current request. These entries are collected from global config,\n    application config (based on request.path_info), and from handler\n    config (exactly how is governed by the request.dispatch object in\n    effect for this request; by default, handler config can be attached\n    anywhere in the tree between request.app.root and the final handler,\n    and inherits downward).'
    is_index = None
    "\n    This will be True if the current request is mapped to an 'index'\n    resource handler (also, a 'default' handler if path_info ends with\n    a slash). The value may be used to automatically redirect the\n    user-agent to a 'more canonical' URL which either adds or removes\n    the trailing slash. See cherrypy.tools.trailing_slash."
    hooks = HookMap(hookpoints)
    '\n    A HookMap (dict-like object) of the form: {hookpoint: [hook, ...]}.\n    Each key is a str naming the hook point, and each value is a list\n    of hooks which will be called at that hook point during this request.\n    The list of hooks is generally populated as early as possible (mostly\n    from Tools specified in config), but may be extended at any time.\n    See also: _cprequest.Hook, _cprequest.HookMap, and cherrypy.tools.'
    error_response = cherrypy.HTTPError(500).set_response
    '\n    The no-arg callable which will handle unexpected, untrapped errors\n    during request processing. This is not used for expected exceptions\n    (like NotFound, HTTPError, or HTTPRedirect) which are raised in\n    response to expected conditions (those should be customized either\n    via request.error_page or by overriding HTTPError.set_response).\n    By default, error_response uses HTTPError(500) to return a generic\n    error response to the user-agent.'
    error_page = {}
    "\n    A dict of {error code: response filename or callable} pairs.\n\n    The error code must be an int representing a given HTTP error code,\n    or the string 'default', which will be used if no matching entry\n    is found for a given numeric code.\n\n    If a filename is provided, the file should contain a Python string-\n    formatting template, and can expect by default to receive format\n    values with the mapping keys %(status)s, %(message)s, %(traceback)s,\n    and %(version)s. The set of format mappings can be extended by\n    overriding HTTPError.set_response.\n\n    If a callable is provided, it will be called by default with keyword\n    arguments 'status', 'message', 'traceback', and 'version', as for a\n    string-formatting template. The callable must return a string or\n    iterable of strings which will be set to response.body. It may also\n    override headers or perform any other processing.\n\n    If no entry is given for an error code, and no 'default' entry exists,\n    a default template will be used.\n    "
    show_tracebacks = True
    '\n    If True, unexpected errors encountered during request processing will\n    include a traceback in the response body.'
    show_mismatched_params = True
    '\n    If True, mismatched parameters encountered during PageHandler invocation\n    processing will be included in the response body.'
    throws = (KeyboardInterrupt, SystemExit, cherrypy.InternalRedirect)
    'The sequence of exceptions which Request.run does not trap.'
    throw_errors = False
    "\n    If True, Request.run will not trap any errors (except HTTPRedirect and\n    HTTPError, which are more properly called 'exceptions', not errors)."
    closed = False
    'True once the close method has been called, False otherwise.'
    stage = None
    '\n    A string containing the stage reached in the request-handling process.\n    This is useful when debugging a live server with hung requests.'
    unique_id = None
    'A lazy object generating and memorizing UUID4 on ``str()`` render.'
    namespaces = reprconf.NamespaceSet(**{'hooks': hooks_namespace, 'request': request_namespace, 'response': response_namespace, 'error_page': error_page_namespace, 'tools': cherrypy.tools})

    def __init__(self, local_host, remote_host, scheme='http', server_protocol='HTTP/1.1'):
        if False:
            for i in range(10):
                print('nop')
        'Populate a new Request object.\n\n        local_host should be an httputil.Host object with the server info.\n        remote_host should be an httputil.Host object with the client info.\n        scheme should be a string, either "http" or "https".\n        '
        self.local = local_host
        self.remote = remote_host
        self.scheme = scheme
        self.server_protocol = server_protocol
        self.closed = False
        self.error_page = self.error_page.copy()
        self.namespaces = self.namespaces.copy()
        self.stage = None
        self.unique_id = LazyUUID4()

    def close(self):
        if False:
            return 10
        'Run cleanup code. (Core)'
        if not self.closed:
            self.closed = True
            self.stage = 'on_end_request'
            self.hooks.run('on_end_request')
            self.stage = 'close'

    def run(self, method, path, query_string, req_protocol, headers, rfile):
        if False:
            print('Hello World!')
        'Process the Request. (Core)\n\n        method, path, query_string, and req_protocol should be pulled directly\n        from the Request-Line (e.g. "GET /path?key=val HTTP/1.0").\n\n        path\n            This should be %XX-unquoted, but query_string should not be.\n\n            When using Python 2, they both MUST be byte strings,\n            not unicode strings.\n\n            When using Python 3, they both MUST be unicode strings,\n            not byte strings, and preferably not bytes \\x00-\\xFF\n            disguised as unicode.\n\n        headers\n            A list of (name, value) tuples.\n\n        rfile\n            A file-like object containing the HTTP request entity.\n\n        When run() is done, the returned object should have 3 attributes:\n\n          * status, e.g. "200 OK"\n          * header_list, a list of (name, value) tuples\n          * body, an iterable yielding strings\n\n        Consumer code (HTTP servers) should then access these response\n        attributes to build the outbound stream.\n\n        '
        response = cherrypy.serving.response
        self.stage = 'run'
        try:
            self.error_response = cherrypy.HTTPError(500).set_response
            self.method = method
            path = path or '/'
            self.query_string = query_string or ''
            self.params = {}
            rp = (int(req_protocol[5]), int(req_protocol[7]))
            sp = (int(self.server_protocol[5]), int(self.server_protocol[7]))
            self.protocol = min(rp, sp)
            response.headers.protocol = self.protocol
            url = path
            if query_string:
                url += '?' + query_string
            self.request_line = '%s %s %s' % (method, url, req_protocol)
            self.header_list = list(headers)
            self.headers = httputil.HeaderMap()
            self.rfile = rfile
            self.body = None
            self.cookie = SimpleCookie()
            self.handler = None
            self.script_name = self.app.script_name
            self.path_info = pi = path[len(self.script_name):]
            self.stage = 'respond'
            self.respond(pi)
        except self.throws:
            raise
        except Exception:
            if self.throw_errors:
                raise
            else:
                cherrypy.log(traceback=True, severity=40)
                if self.show_tracebacks:
                    body = format_exc()
                else:
                    body = ''
                r = bare_error(body)
                (response.output_status, response.header_list, response.body) = r
        if self.method == 'HEAD':
            response.body = []
        try:
            cherrypy.log.access()
        except Exception:
            cherrypy.log.error(traceback=True)
        return response

    def respond(self, path_info):
        if False:
            return 10
        'Generate a response for the resource at self.path_info. (Core)'
        try:
            try:
                try:
                    self._do_respond(path_info)
                except (cherrypy.HTTPRedirect, cherrypy.HTTPError):
                    inst = sys.exc_info()[1]
                    inst.set_response()
                    self.stage = 'before_finalize (HTTPError)'
                    self.hooks.run('before_finalize')
                    cherrypy.serving.response.finalize()
            finally:
                self.stage = 'on_end_resource'
                self.hooks.run('on_end_resource')
        except self.throws:
            raise
        except Exception:
            if self.throw_errors:
                raise
            self.handle_error()

    def _do_respond(self, path_info):
        if False:
            while True:
                i = 10
        response = cherrypy.serving.response
        if self.app is None:
            raise cherrypy.NotFound()
        self.hooks = self.__class__.hooks.copy()
        self.toolmaps = {}
        self.stage = 'process_headers'
        self.process_headers()
        self.stage = 'get_resource'
        self.get_resource(path_info)
        self.body = _cpreqbody.RequestBody(self.rfile, self.headers, request_params=self.params)
        self.namespaces(self.config)
        self.stage = 'on_start_resource'
        self.hooks.run('on_start_resource')
        self.stage = 'process_query_string'
        self.process_query_string()
        if self.process_request_body:
            if self.method not in self.methods_with_bodies:
                self.process_request_body = False
        self.stage = 'before_request_body'
        self.hooks.run('before_request_body')
        if self.process_request_body:
            self.body.process()
        self.stage = 'before_handler'
        self.hooks.run('before_handler')
        if self.handler:
            self.stage = 'handler'
            response.body = self.handler()
        self.stage = 'before_finalize'
        self.hooks.run('before_finalize')
        response.finalize()

    def process_query_string(self):
        if False:
            print('Hello World!')
        'Parse the query string into Python structures. (Core)'
        try:
            p = httputil.parse_query_string(self.query_string, encoding=self.query_string_encoding)
        except UnicodeDecodeError:
            raise cherrypy.HTTPError(404, 'The given query string could not be processed. Query strings for this resource must be encoded with %r.' % self.query_string_encoding)
        self.params.update(p)

    def process_headers(self):
        if False:
            while True:
                i = 10
        'Parse HTTP header data into Python structures. (Core)'
        headers = self.headers
        for (name, value) in self.header_list:
            name = name.title()
            value = value.strip()
            headers[name] = httputil.decode_TEXT_maybe(value)
            if name == 'Cookie':
                try:
                    self.cookie.load(value)
                except CookieError as exc:
                    raise cherrypy.HTTPError(400, str(exc))
        if not dict.__contains__(headers, 'Host'):
            if self.protocol >= (1, 1):
                msg = "HTTP/1.1 requires a 'Host' request header."
                raise cherrypy.HTTPError(400, msg)
        else:
            headers['Host'] = httputil.SanitizedHost(dict.get(headers, 'Host'))
        host = dict.get(headers, 'Host')
        if not host:
            host = self.local.name or self.local.ip
        self.base = '%s://%s' % (self.scheme, host)

    def get_resource(self, path):
        if False:
            i = 10
            return i + 15
        'Call a dispatcher (which sets self.handler and .config). (Core)'
        dispatch = self.app.find_config(path, 'request.dispatch', self.dispatch)
        dispatch(path)

    def handle_error(self):
        if False:
            print('Hello World!')
        'Handle the last unanticipated exception. (Core)'
        try:
            self.hooks.run('before_error_response')
            if self.error_response:
                self.error_response()
            self.hooks.run('after_error_response')
            cherrypy.serving.response.finalize()
        except cherrypy.HTTPRedirect:
            inst = sys.exc_info()[1]
            inst.set_response()
            cherrypy.serving.response.finalize()

class ResponseBody(object):
    """The body of the HTTP response (the response entity)."""
    unicode_err = 'Page handlers MUST return bytes. Use tools.encode if you wish to return unicode.'

    def __get__(self, obj, objclass=None):
        if False:
            while True:
                i = 10
        if obj is None:
            return self
        else:
            return obj._body

    def __set__(self, obj, value):
        if False:
            while True:
                i = 10
        if isinstance(value, str):
            raise ValueError(self.unicode_err)
        elif isinstance(value, list):
            if any((isinstance(item, str) for item in value)):
                raise ValueError(self.unicode_err)
        obj._body = encoding.prepare_iter(value)

class Response(object):
    """An HTTP Response, including status, headers, and body."""
    status = ''
    'The HTTP Status-Code and Reason-Phrase.'
    header_list = []
    '\n    A list of the HTTP response headers as (name, value) tuples.\n    In general, you should use response.headers (a dict) instead. This\n    attribute is generated from response.headers and is not valid until\n    after the finalize phase.'
    headers = httputil.HeaderMap()
    "\n    A dict-like object containing the response headers. Keys are header\n    names (in Title-Case format); however, you may get and set them in\n    a case-insensitive manner. That is, headers['Content-Type'] and\n    headers['content-type'] refer to the same value. Values are header\n    values (decoded according to :rfc:`2047` if necessary).\n\n    .. seealso:: classes :class:`HeaderMap`, :class:`HeaderElement`\n    "
    cookie = SimpleCookie()
    'See help(Cookie).'
    body = ResponseBody()
    'The body (entity) of the HTTP response.'
    time = None
    'The value of time.time() when created. Use in HTTP dates.'
    stream = False
    'If False, buffer the response body.'

    def __init__(self):
        if False:
            print('Hello World!')
        self.status = None
        self.header_list = None
        self._body = []
        self.time = time.time()
        self.headers = httputil.HeaderMap()
        dict.update(self.headers, {'Content-Type': 'text/html', 'Server': 'CherryPy/' + cherrypy.__version__, 'Date': httputil.HTTPDate(self.time)})
        self.cookie = SimpleCookie()

    def collapse_body(self):
        if False:
            i = 10
            return i + 15
        'Collapse self.body to a single string; replace it and return it.'
        new_body = b''.join(self.body)
        self.body = new_body
        return new_body

    def _flush_body(self):
        if False:
            print('Hello World!')
        '\n        Discard self.body but consume any generator such that\n        any finalization can occur, such as is required by\n        caching.tee_output().\n        '
        consume(iter(self.body))

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        'Transform headers (and cookies) into self.header_list. (Core)'
        try:
            (code, reason, _) = httputil.valid_status(self.status)
        except ValueError:
            raise cherrypy.HTTPError(500, sys.exc_info()[1].args[0])
        headers = self.headers
        self.status = '%s %s' % (code, reason)
        self.output_status = ntob(str(code), 'ascii') + b' ' + headers.encode(reason)
        if self.stream:
            if dict.get(headers, 'Content-Length') is None:
                dict.pop(headers, 'Content-Length', None)
        elif code < 200 or code in (204, 205, 304):
            dict.pop(headers, 'Content-Length', None)
            self._flush_body()
            self.body = b''
        elif dict.get(headers, 'Content-Length') is None:
            content = self.collapse_body()
            dict.__setitem__(headers, 'Content-Length', len(content))
        self.header_list = h = headers.output()
        cookie = self.cookie.output()
        if cookie:
            for line in cookie.split('\r\n'):
                (name, value) = line.split(': ', 1)
                if isinstance(name, str):
                    name = name.encode('ISO-8859-1')
                if isinstance(value, str):
                    value = headers.encode(value)
                h.append((name, value))

class LazyUUID4(object):

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'Return UUID4 and keep it for future calls.'
        return str(self.uuid4)

    @property
    def uuid4(self):
        if False:
            print('Hello World!')
        "Provide unique id on per-request basis using UUID4.\n\n        It's evaluated lazily on render.\n        "
        try:
            self._uuid4
        except AttributeError:
            self._uuid4 = uuid.uuid4()
        return self._uuid4