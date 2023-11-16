"""Functions for builtin CherryPy tools."""
import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator

def validate_etags(autotags=False, debug=False):
    if False:
        while True:
            i = 10
    "Validate the current ETag against If-Match, If-None-Match headers.\n\n    If autotags is True, an ETag response-header value will be provided\n    from an MD5 hash of the response body (unless some other code has\n    already provided an ETag header). If False (the default), the ETag\n    will not be automatic.\n\n    WARNING: the autotags feature is not designed for URL's which allow\n    methods other than GET. For example, if a POST to the same URL returns\n    no content, the automatic ETag will be incorrect, breaking a fundamental\n    use for entity tags in a possibly destructive fashion. Likewise, if you\n    raise 304 Not Modified, the response body will be empty, the ETag hash\n    will be incorrect, and your application will break.\n    See :rfc:`2616` Section 14.24.\n    "
    response = cherrypy.serving.response
    if hasattr(response, 'ETag'):
        return
    (status, reason, msg) = _httputil.valid_status(response.status)
    etag = response.headers.get('ETag')
    if etag:
        if debug:
            cherrypy.log('ETag already set: %s' % etag, 'TOOLS.ETAGS')
    elif not autotags:
        if debug:
            cherrypy.log('Autotags off', 'TOOLS.ETAGS')
    elif status != 200:
        if debug:
            cherrypy.log('Status not 200', 'TOOLS.ETAGS')
    else:
        etag = response.collapse_body()
        etag = '"%s"' % md5(etag).hexdigest()
        if debug:
            cherrypy.log('Setting ETag: %s' % etag, 'TOOLS.ETAGS')
        response.headers['ETag'] = etag
    response.ETag = etag
    if debug:
        cherrypy.log('Status: %s' % status, 'TOOLS.ETAGS')
    if status >= 200 and status <= 299:
        request = cherrypy.serving.request
        conditions = request.headers.elements('If-Match') or []
        conditions = [str(x) for x in conditions]
        if debug:
            cherrypy.log('If-Match conditions: %s' % repr(conditions), 'TOOLS.ETAGS')
        if conditions and (not (conditions == ['*'] or etag in conditions)):
            raise cherrypy.HTTPError(412, 'If-Match failed: ETag %r did not match %r' % (etag, conditions))
        conditions = request.headers.elements('If-None-Match') or []
        conditions = [str(x) for x in conditions]
        if debug:
            cherrypy.log('If-None-Match conditions: %s' % repr(conditions), 'TOOLS.ETAGS')
        if conditions == ['*'] or etag in conditions:
            if debug:
                cherrypy.log('request.method: %s' % request.method, 'TOOLS.ETAGS')
            if request.method in ('GET', 'HEAD'):
                raise cherrypy.HTTPRedirect([], 304)
            else:
                raise cherrypy.HTTPError(412, 'If-None-Match failed: ETag %r matched %r' % (etag, conditions))

def validate_since():
    if False:
        print('Hello World!')
    'Validate the current Last-Modified against If-Modified-Since headers.\n\n    If no code has set the Last-Modified response header, then no validation\n    will be performed.\n    '
    response = cherrypy.serving.response
    lastmod = response.headers.get('Last-Modified')
    if lastmod:
        (status, reason, msg) = _httputil.valid_status(response.status)
        request = cherrypy.serving.request
        since = request.headers.get('If-Unmodified-Since')
        if since and since != lastmod:
            if status >= 200 and status <= 299 or status == 412:
                raise cherrypy.HTTPError(412)
        since = request.headers.get('If-Modified-Since')
        if since and since == lastmod:
            if status >= 200 and status <= 299 or status == 304:
                if request.method in ('GET', 'HEAD'):
                    raise cherrypy.HTTPRedirect([], 304)
                else:
                    raise cherrypy.HTTPError(412)

def allow(methods=None, debug=False):
    if False:
        print('Hello World!')
    "Raise 405 if request.method not in methods (default ['GET', 'HEAD']).\n\n    The given methods are case-insensitive, and may be in any order.\n    If only one method is allowed, you may supply a single string;\n    if more than one, supply a list of strings.\n\n    Regardless of whether the current method is allowed or not, this\n    also emits an 'Allow' response header, containing the given methods.\n    "
    if not isinstance(methods, (tuple, list)):
        methods = [methods]
    methods = [m.upper() for m in methods if m]
    if not methods:
        methods = ['GET', 'HEAD']
    elif 'GET' in methods and 'HEAD' not in methods:
        methods.append('HEAD')
    cherrypy.response.headers['Allow'] = ', '.join(methods)
    if cherrypy.request.method not in methods:
        if debug:
            cherrypy.log('request.method %r not in methods %r' % (cherrypy.request.method, methods), 'TOOLS.ALLOW')
        raise cherrypy.HTTPError(405)
    elif debug:
        cherrypy.log('request.method %r in methods %r' % (cherrypy.request.method, methods), 'TOOLS.ALLOW')

def proxy(base=None, local='X-Forwarded-Host', remote='X-Forwarded-For', scheme='X-Forwarded-Proto', debug=False):
    if False:
        while True:
            i = 10
    "Change the base URL (scheme://host[:port][/path]).\n\n    For running a CP server behind Apache, lighttpd, or other HTTP server.\n\n    For Apache and lighttpd, you should leave the 'local' argument at the\n    default value of 'X-Forwarded-Host'. For Squid, you probably want to set\n    tools.proxy.local = 'Origin'.\n\n    If you want the new request.base to include path info (not just the host),\n    you must explicitly set base to the full base path, and ALSO set 'local'\n    to '', so that the X-Forwarded-Host request header (which never includes\n    path info) does not override it. Regardless, the value for 'base' MUST\n    NOT end in a slash.\n\n    cherrypy.request.remote.ip (the IP address of the client) will be\n    rewritten if the header specified by the 'remote' arg is valid.\n    By default, 'remote' is set to 'X-Forwarded-For'. If you do not\n    want to rewrite remote.ip, set the 'remote' arg to an empty string.\n    "
    request = cherrypy.serving.request
    if scheme:
        s = request.headers.get(scheme, None)
        if debug:
            cherrypy.log('Testing scheme %r:%r' % (scheme, s), 'TOOLS.PROXY')
        if s == 'on' and 'ssl' in scheme.lower():
            scheme = 'https'
        else:
            scheme = s
    if not scheme:
        scheme = request.base[:request.base.find('://')]
    if local:
        lbase = request.headers.get(local, None)
        if debug:
            cherrypy.log('Testing local %r:%r' % (local, lbase), 'TOOLS.PROXY')
        if lbase is not None:
            base = lbase.split(',')[0]
    if not base:
        default = urllib.parse.urlparse(request.base).netloc
        base = request.headers.get('Host', default)
    if base.find('://') == -1:
        base = scheme + '://' + base
    request.base = base
    if remote:
        xff = request.headers.get(remote)
        if debug:
            cherrypy.log('Testing remote %r:%r' % (remote, xff), 'TOOLS.PROXY')
        if xff:
            if remote == 'X-Forwarded-For':
                xff = next((ip.strip() for ip in xff.split(',')))
            request.remote.ip = xff

def ignore_headers(headers=('Range',), debug=False):
    if False:
        i = 10
        return i + 15
    "Delete request headers whose field names are included in 'headers'.\n\n    This is a useful tool for working behind certain HTTP servers;\n    for example, Apache duplicates the work that CP does for 'Range'\n    headers, and will doubly-truncate the response.\n    "
    request = cherrypy.serving.request
    for name in headers:
        if name in request.headers:
            if debug:
                cherrypy.log('Ignoring request header %r' % name, 'TOOLS.IGNORE_HEADERS')
            del request.headers[name]

def response_headers(headers=None, debug=False):
    if False:
        print('Hello World!')
    'Set headers on the response.'
    if debug:
        cherrypy.log('Setting response headers: %s' % repr(headers), 'TOOLS.RESPONSE_HEADERS')
    for (name, value) in headers or []:
        cherrypy.serving.response.headers[name] = value
response_headers.failsafe = True

def referer(pattern, accept=True, accept_missing=False, error=403, message='Forbidden Referer header.', debug=False):
    if False:
        print('Hello World!')
    'Raise HTTPError if Referer header does/does not match the given pattern.\n\n    pattern\n        A regular expression pattern to test against the Referer.\n\n    accept\n        If True, the Referer must match the pattern; if False,\n        the Referer must NOT match the pattern.\n\n    accept_missing\n        If True, permit requests with no Referer header.\n\n    error\n        The HTTP error code to return to the client on failure.\n\n    message\n        A string to include in the response body on failure.\n\n    '
    try:
        ref = cherrypy.serving.request.headers['Referer']
        match = bool(re.match(pattern, ref))
        if debug:
            cherrypy.log('Referer %r matches %r' % (ref, pattern), 'TOOLS.REFERER')
        if accept == match:
            return
    except KeyError:
        if debug:
            cherrypy.log('No Referer header', 'TOOLS.REFERER')
        if accept_missing:
            return
    raise cherrypy.HTTPError(error, message)

class SessionAuth(object):
    """Assert that the user is logged in."""
    session_key = 'username'
    debug = False

    def check_username_and_password(self, username, password):
        if False:
            for i in range(10):
                print('nop')
        pass

    def anonymous(self):
        if False:
            for i in range(10):
                print('nop')
        'Provide a temporary user name for anonymous users.'
        pass

    def on_login(self, username):
        if False:
            while True:
                i = 10
        pass

    def on_logout(self, username):
        if False:
            print('Hello World!')
        pass

    def on_check(self, username):
        if False:
            print('Hello World!')
        pass

    def login_screen(self, from_page='..', username='', error_msg='', **kwargs):
        if False:
            return 10
        return (str('<html><body>\nMessage: %(error_msg)s\n<form method="post" action="do_login">\n    Login: <input type="text" name="username" value="%(username)s" size="10" />\n    <br />\n    Password: <input type="password" name="password" size="10" />\n    <br />\n    <input type="hidden" name="from_page" value="%(from_page)s" />\n    <br />\n    <input type="submit" />\n</form>\n</body></html>') % vars()).encode('utf-8')

    def do_login(self, username, password, from_page='..', **kwargs):
        if False:
            print('Hello World!')
        'Login. May raise redirect, or return True if request handled.'
        response = cherrypy.serving.response
        error_msg = self.check_username_and_password(username, password)
        if error_msg:
            body = self.login_screen(from_page, username, error_msg)
            response.body = body
            if 'Content-Length' in response.headers:
                del response.headers['Content-Length']
            return True
        else:
            cherrypy.serving.request.login = username
            cherrypy.session[self.session_key] = username
            self.on_login(username)
            raise cherrypy.HTTPRedirect(from_page or '/')

    def do_logout(self, from_page='..', **kwargs):
        if False:
            while True:
                i = 10
        'Logout. May raise redirect, or return True if request handled.'
        sess = cherrypy.session
        username = sess.get(self.session_key)
        sess[self.session_key] = None
        if username:
            cherrypy.serving.request.login = None
            self.on_logout(username)
        raise cherrypy.HTTPRedirect(from_page)

    def do_check(self):
        if False:
            while True:
                i = 10
        'Assert username. Raise redirect, or return True if request handled.\n        '
        sess = cherrypy.session
        request = cherrypy.serving.request
        response = cherrypy.serving.response
        username = sess.get(self.session_key)
        if not username:
            sess[self.session_key] = username = self.anonymous()
            self._debug_message('No session[username], trying anonymous')
        if not username:
            url = cherrypy.url(qs=request.query_string)
            self._debug_message('No username, routing to login_screen with from_page %(url)r', locals())
            response.body = self.login_screen(url)
            if 'Content-Length' in response.headers:
                del response.headers['Content-Length']
            return True
        self._debug_message('Setting request.login to %(username)r', locals())
        request.login = username
        self.on_check(username)

    def _debug_message(self, template, context={}):
        if False:
            i = 10
            return i + 15
        if not self.debug:
            return
        cherrypy.log(template % context, 'TOOLS.SESSAUTH')

    def run(self):
        if False:
            i = 10
            return i + 15
        request = cherrypy.serving.request
        response = cherrypy.serving.response
        path = request.path_info
        if path.endswith('login_screen'):
            self._debug_message('routing %(path)r to login_screen', locals())
            response.body = self.login_screen()
            return True
        elif path.endswith('do_login'):
            if request.method != 'POST':
                response.headers['Allow'] = 'POST'
                self._debug_message('do_login requires POST')
                raise cherrypy.HTTPError(405)
            self._debug_message('routing %(path)r to do_login', locals())
            return self.do_login(**request.params)
        elif path.endswith('do_logout'):
            if request.method != 'POST':
                response.headers['Allow'] = 'POST'
                raise cherrypy.HTTPError(405)
            self._debug_message('routing %(path)r to do_logout', locals())
            return self.do_logout(**request.params)
        else:
            self._debug_message('No special path, running do_check')
            return self.do_check()

def session_auth(**kwargs):
    if False:
        return 10
    'Session authentication hook.\n\n    Any attribute of the SessionAuth class may be overridden\n    via a keyword arg to this function:\n\n    ' + '\n    '.join(('{!s}: {!s}'.format(k, type(getattr(SessionAuth, k)).__name__) for k in dir(SessionAuth) if not k.startswith('__')))
    sa = SessionAuth()
    for (k, v) in kwargs.items():
        setattr(sa, k, v)
    return sa.run()

def log_traceback(severity=logging.ERROR, debug=False):
    if False:
        i = 10
        return i + 15
    "Write the last error's traceback to the cherrypy error log."
    cherrypy.log('', 'HTTP', severity=severity, traceback=True)

def log_request_headers(debug=False):
    if False:
        for i in range(10):
            print('nop')
    'Write request headers to the cherrypy error log.'
    h = ['  %s: %s' % (k, v) for (k, v) in cherrypy.serving.request.header_list]
    cherrypy.log('\nRequest Headers:\n' + '\n'.join(h), 'HTTP')

def log_hooks(debug=False):
    if False:
        print('Hello World!')
    'Write request.hooks to the cherrypy error log.'
    request = cherrypy.serving.request
    msg = []
    from cherrypy import _cprequest
    points = _cprequest.hookpoints
    for k in request.hooks.keys():
        if k not in points:
            points.append(k)
    for k in points:
        msg.append('    %s:' % k)
        v = request.hooks.get(k, [])
        v.sort()
        for h in v:
            msg.append('        %r' % h)
    cherrypy.log('\nRequest Hooks for ' + cherrypy.url() + ':\n' + '\n'.join(msg), 'HTTP')

def redirect(url='', internal=True, debug=False):
    if False:
        i = 10
        return i + 15
    'Raise InternalRedirect or HTTPRedirect to the given url.'
    if debug:
        cherrypy.log('Redirecting %sto: %s' % ({True: 'internal ', False: ''}[internal], url), 'TOOLS.REDIRECT')
    if internal:
        raise cherrypy.InternalRedirect(url)
    else:
        raise cherrypy.HTTPRedirect(url)

def trailing_slash(missing=True, extra=False, status=None, debug=False):
    if False:
        return 10
    'Redirect if path_info has (missing|extra) trailing slash.'
    request = cherrypy.serving.request
    pi = request.path_info
    if debug:
        cherrypy.log('is_index: %r, missing: %r, extra: %r, path_info: %r' % (request.is_index, missing, extra, pi), 'TOOLS.TRAILING_SLASH')
    if request.is_index is True:
        if missing:
            if not pi.endswith('/'):
                new_url = cherrypy.url(pi + '/', request.query_string)
                raise cherrypy.HTTPRedirect(new_url, status=status or 301)
    elif request.is_index is False:
        if extra:
            if pi.endswith('/') and pi != '/':
                new_url = cherrypy.url(pi[:-1], request.query_string)
                raise cherrypy.HTTPRedirect(new_url, status=status or 301)

def flatten(debug=False):
    if False:
        while True:
            i = 10
    "Wrap response.body in a generator that recursively iterates over body.\n\n    This allows cherrypy.response.body to consist of 'nested generators';\n    that is, a set of generators that yield generators.\n    "

    def flattener(input):
        if False:
            i = 10
            return i + 15
        numchunks = 0
        for x in input:
            if not is_iterator(x):
                numchunks += 1
                yield x
            else:
                for y in flattener(x):
                    numchunks += 1
                    yield y
        if debug:
            cherrypy.log('Flattened %d chunks' % numchunks, 'TOOLS.FLATTEN')
    response = cherrypy.serving.response
    response.body = flattener(response.body)

def accept(media=None, debug=False):
    if False:
        print('Hello World!')
    'Return the client\'s preferred media-type (from the given Content-Types).\n\n    If \'media\' is None (the default), no test will be performed.\n\n    If \'media\' is provided, it should be the Content-Type value (as a string)\n    or values (as a list or tuple of strings) which the current resource\n    can emit. The client\'s acceptable media ranges (as declared in the\n    Accept request header) will be matched in order to these Content-Type\n    values; the first such string is returned. That is, the return value\n    will always be one of the strings provided in the \'media\' arg (or None\n    if \'media\' is None).\n\n    If no match is found, then HTTPError 406 (Not Acceptable) is raised.\n    Note that most web browsers send */* as a (low-quality) acceptable\n    media range, which should match any Content-Type. In addition, "...if\n    no Accept header field is present, then it is assumed that the client\n    accepts all media types."\n\n    Matching types are checked in order of client preference first,\n    and then in the order of the given \'media\' values.\n\n    Note that this function does not honor accept-params (other than "q").\n    '
    if not media:
        return
    if isinstance(media, text_or_bytes):
        media = [media]
    request = cherrypy.serving.request
    ranges = request.headers.elements('Accept')
    if not ranges:
        if debug:
            cherrypy.log('No Accept header elements', 'TOOLS.ACCEPT')
        return media[0]
    else:
        for element in ranges:
            if element.qvalue > 0:
                if element.value == '*/*':
                    if debug:
                        cherrypy.log('Match due to */*', 'TOOLS.ACCEPT')
                    return media[0]
                elif element.value.endswith('/*'):
                    mtype = element.value[:-1]
                    for m in media:
                        if m.startswith(mtype):
                            if debug:
                                cherrypy.log('Match due to %s' % element.value, 'TOOLS.ACCEPT')
                            return m
                elif element.value in media:
                    if debug:
                        cherrypy.log('Match due to %s' % element.value, 'TOOLS.ACCEPT')
                    return element.value
    ah = request.headers.get('Accept')
    if ah is None:
        msg = 'Your client did not send an Accept header.'
    else:
        msg = 'Your client sent this Accept header: %s.' % ah
    msg += ' But this resource only emits these media types: %s.' % ', '.join(media)
    raise cherrypy.HTTPError(406, msg)

class MonitoredHeaderMap(_httputil.HeaderMap):

    def transform_key(self, key):
        if False:
            for i in range(10):
                print('nop')
        self.accessed_headers.add(key)
        return super(MonitoredHeaderMap, self).transform_key(key)

    def __init__(self):
        if False:
            print('Hello World!')
        self.accessed_headers = set()
        super(MonitoredHeaderMap, self).__init__()

def autovary(ignore=None, debug=False):
    if False:
        while True:
            i = 10
    'Auto-populate the Vary response header based on request.header access.\n    '
    request = cherrypy.serving.request
    req_h = request.headers
    request.headers = MonitoredHeaderMap()
    request.headers.update(req_h)
    if ignore is None:
        ignore = set(['Content-Disposition', 'Content-Length', 'Content-Type'])

    def set_response_header():
        if False:
            return 10
        resp_h = cherrypy.serving.response.headers
        v = set([e.value for e in resp_h.elements('Vary')])
        if debug:
            cherrypy.log('Accessed headers: %s' % request.headers.accessed_headers, 'TOOLS.AUTOVARY')
        v = v.union(request.headers.accessed_headers)
        v = v.difference(ignore)
        v = list(v)
        v.sort()
        resp_h['Vary'] = ', '.join(v)
    request.hooks.attach('before_finalize', set_response_header, 95)

def convert_params(exception=ValueError, error=400):
    if False:
        for i in range(10):
            print('nop')
    'Convert request params based on function annotations, with error handling.\n\n    exception\n        Exception class to catch.\n\n    status\n        The HTTP error code to return to the client on failure.\n    '
    request = cherrypy.serving.request
    types = request.handler.callable.__annotations__
    with cherrypy.HTTPError.handle(exception, error):
        for key in set(types).intersection(request.params):
            request.params[key] = types[key](request.params[key])