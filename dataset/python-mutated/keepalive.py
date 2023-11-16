"""An HTTP handler for urllib2 that supports HTTP 1.1 and keepalive.

>>> import urllib2
>>> from keepalive import HTTPHandler
>>> keepalive_handler = HTTPHandler()
>>> opener = _urllib.request.build_opener(keepalive_handler)
>>> _urllib.request.install_opener(opener)
>>> 
>>> fo = _urllib.request.urlopen('http://www.python.org')

If a connection to a given host is requested, and all of the existing
connections are still in use, another connection will be opened.  If
the handler tries to use an existing connection but it fails in some
way, it will be closed and removed from the pool.

To remove the handler, simply re-run build_opener with no arguments, and
install that opener.

You can explicitly close connections by using the close_connection()
method of the returned file-like object (described below) or you can
use the handler methods:

  close_connection(host)
  close_all()
  open_connections()

NOTE: using the close_connection and close_all methods of the handler
should be done with care when using multiple threads.
  * there is nothing that prevents another thread from creating new
    connections immediately after connections are closed
  * no checks are done to prevent in-use connections from being closed

>>> keepalive_handler.close_all()

EXTRA ATTRIBUTES AND METHODS

  Upon a status of 200, the object returned has a few additional
  attributes and methods, which should not be used if you want to
  remain consistent with the normal urllib2-returned objects:

    close_connection()  -  close the connection to the host
    readlines()         -  you know, readlines()
    status              -  the return status (ie 404)
    reason              -  english translation of status (ie 'File not found')

  If you want the best of both worlds, use this inside an
  AttributeError-catching try:

  >>> try: status = fo.status
  >>> except AttributeError: status = None

  Unfortunately, these are ONLY there if status == 200, so it's not
  easy to distinguish between non-200 responses.  The reason is that
  urllib2 tries to do clever things with error codes 301, 302, 401,
  and 407, and it wraps the object upon return.

  For python versions earlier than 2.4, you can avoid this fancy error
  handling by setting the module-level global HANDLE_ERRORS to zero.
  You see, prior to 2.4, it's the HTTP Handler's job to determine what
  to handle specially, and what to just pass up.  HANDLE_ERRORS == 0
  means "pass everything up".  In python 2.4, however, this job no
  longer belongs to the HTTP Handler and is now done by a NEW handler,
  HTTPErrorProcessor.  Here's the bottom line:

    python version < 2.4
        HANDLE_ERRORS == 1  (default) pass up 200, treat the rest as
                            errors
        HANDLE_ERRORS == 0  pass everything up, error processing is
                            left to the calling code
    python version >= 2.4
        HANDLE_ERRORS == 1  pass up 200, treat the rest as errors
        HANDLE_ERRORS == 0  (default) pass everything up, let the
                            other handlers (specifically,
                            HTTPErrorProcessor) decide what to do

  In practice, setting the variable either way makes little difference
  in python 2.4, so for the most consistent behavior across versions,
  you probably just want to use the defaults, which will give you
  exceptions on errors.

"""
from __future__ import print_function
try:
    from thirdparty.six.moves import http_client as _http_client
    from thirdparty.six.moves import range as _range
    from thirdparty.six.moves import urllib as _urllib
except ImportError:
    from six.moves import http_client as _http_client
    from six.moves import range as _range
    from six.moves import urllib as _urllib
import socket
import threading
DEBUG = None
import sys
if sys.version_info < (2, 4):
    HANDLE_ERRORS = 1
else:
    HANDLE_ERRORS = 0

class ConnectionManager:
    """
    The connection manager must be able to:
      * keep track of all existing
      """

    def __init__(self):
        if False:
            return 10
        self._lock = threading.Lock()
        self._hostmap = {}
        self._connmap = {}
        self._readymap = {}

    def add(self, host, connection, ready):
        if False:
            i = 10
            return i + 15
        self._lock.acquire()
        try:
            if host not in self._hostmap:
                self._hostmap[host] = []
            self._hostmap[host].append(connection)
            self._connmap[connection] = host
            self._readymap[connection] = ready
        finally:
            self._lock.release()

    def remove(self, connection):
        if False:
            while True:
                i = 10
        self._lock.acquire()
        try:
            try:
                host = self._connmap[connection]
            except KeyError:
                pass
            else:
                del self._connmap[connection]
                del self._readymap[connection]
                self._hostmap[host].remove(connection)
                if not self._hostmap[host]:
                    del self._hostmap[host]
        finally:
            self._lock.release()

    def set_ready(self, connection, ready):
        if False:
            print('Hello World!')
        try:
            self._readymap[connection] = ready
        except KeyError:
            pass

    def get_ready_conn(self, host):
        if False:
            return 10
        conn = None
        try:
            self._lock.acquire()
            if host in self._hostmap:
                for c in self._hostmap[host]:
                    if self._readymap.get(c):
                        self._readymap[c] = 0
                        conn = c
                        break
        finally:
            self._lock.release()
        return conn

    def get_all(self, host=None):
        if False:
            for i in range(10):
                print('nop')
        if host:
            return list(self._hostmap.get(host, []))
        else:
            return dict(self._hostmap)

class KeepAliveHandler:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._cm = ConnectionManager()

    def open_connections(self):
        if False:
            while True:
                i = 10
        "return a list of connected hosts and the number of connections\n        to each.  [('foo.com:80', 2), ('bar.org', 1)]"
        return [(host, len(li)) for (host, li) in self._cm.get_all().items()]

    def close_connection(self, host):
        if False:
            i = 10
            return i + 15
        "close connection(s) to <host>\n        host is the host:port spec, as in 'www.cnn.com:8080' as passed in.\n        no error occurs if there is no connection to that host."
        for h in self._cm.get_all(host):
            self._cm.remove(h)
            h.close()

    def close_all(self):
        if False:
            for i in range(10):
                print('nop')
        'close all open connections'
        for (host, conns) in self._cm.get_all().items():
            for h in conns:
                self._cm.remove(h)
                h.close()

    def _request_closed(self, request, host, connection):
        if False:
            for i in range(10):
                print('nop')
        'tells us that this request is now closed and the the\n        connection is ready for another request'
        self._cm.set_ready(connection, 1)

    def _remove_connection(self, host, connection, close=0):
        if False:
            while True:
                i = 10
        if close:
            connection.close()
        self._cm.remove(connection)

    def do_open(self, req):
        if False:
            print('Hello World!')
        host = req.host
        if not host:
            raise _urllib.error.URLError('no host given')
        try:
            h = self._cm.get_ready_conn(host)
            while h:
                r = self._reuse_connection(h, req, host)
                if r:
                    break
                h.close()
                self._cm.remove(h)
                h = self._cm.get_ready_conn(host)
            else:
                h = self._get_connection(host)
                if DEBUG:
                    DEBUG.info('creating new connection to %s (%d)', host, id(h))
                self._cm.add(host, h, 0)
                self._start_transaction(h, req)
                r = h.getresponse()
        except (socket.error, _http_client.HTTPException) as err:
            raise _urllib.error.URLError(err)
        if DEBUG:
            DEBUG.info('STATUS: %s, %s', r.status, r.reason)
        if r.will_close:
            if DEBUG:
                DEBUG.info('server will close connection, discarding')
            self._cm.remove(h)
        r._handler = self
        r._host = host
        r._url = req.get_full_url()
        r._connection = h
        r.code = r.status
        r.headers = r.msg
        r.msg = r.reason
        if r.status == 200 or not HANDLE_ERRORS:
            return r
        else:
            return self.parent.error('http', req, r, r.status, r.msg, r.headers)

    def _reuse_connection(self, h, req, host):
        if False:
            while True:
                i = 10
        'start the transaction with a re-used connection\n        return a response object (r) upon success or None on failure.\n        This DOES not close or remove bad connections in cases where\n        it returns.  However, if an unexpected exception occurs, it\n        will close and remove the connection before re-raising.\n        '
        try:
            self._start_transaction(h, req)
            r = h.getresponse()
        except (socket.error, _http_client.HTTPException):
            r = None
        except:
            if DEBUG:
                DEBUG.error('unexpected exception - closing ' + 'connection to %s (%d)', host, id(h))
            self._cm.remove(h)
            h.close()
            raise
        if r is None or r.version == 9:
            if DEBUG:
                DEBUG.info('failed to re-use connection to %s (%d)', host, id(h))
            r = None
        elif DEBUG:
            DEBUG.info('re-using connection to %s (%d)', host, id(h))
        return r

    def _start_transaction(self, h, req):
        if False:
            for i in range(10):
                print('nop')
        try:
            if req.data:
                data = req.data
                if hasattr(req, 'selector'):
                    h.putrequest(req.get_method() or 'POST', req.selector, skip_host=req.has_header('Host'), skip_accept_encoding=req.has_header('Accept-encoding'))
                else:
                    h.putrequest(req.get_method() or 'POST', req.get_selector(), skip_host=req.has_header('Host'), skip_accept_encoding=req.has_header('Accept-encoding'))
                if 'Content-type' not in req.headers:
                    h.putheader('Content-type', 'application/x-www-form-urlencoded')
                if 'Content-length' not in req.headers:
                    h.putheader('Content-length', '%d' % len(data))
            elif hasattr(req, 'selector'):
                h.putrequest(req.get_method() or 'GET', req.selector, skip_host=req.has_header('Host'), skip_accept_encoding=req.has_header('Accept-encoding'))
            else:
                h.putrequest(req.get_method() or 'GET', req.get_selector(), skip_host=req.has_header('Host'), skip_accept_encoding=req.has_header('Accept-encoding'))
        except (socket.error, _http_client.HTTPException) as err:
            raise _urllib.error.URLError(err)
        if 'Connection' not in req.headers:
            req.headers['Connection'] = 'keep-alive'
        for args in self.parent.addheaders:
            if args[0] not in req.headers:
                h.putheader(*args)
        for (k, v) in req.headers.items():
            h.putheader(k, v)
        h.endheaders()
        if req.data:
            h.send(data)

    def _get_connection(self, host):
        if False:
            print('Hello World!')
        return NotImplementedError

class HTTPHandler(KeepAliveHandler, _urllib.request.HTTPHandler):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        KeepAliveHandler.__init__(self)

    def http_open(self, req):
        if False:
            return 10
        return self.do_open(req)

    def _get_connection(self, host):
        if False:
            for i in range(10):
                print('nop')
        return HTTPConnection(host)

class HTTPSHandler(KeepAliveHandler, _urllib.request.HTTPSHandler):

    def __init__(self, ssl_factory=None):
        if False:
            return 10
        KeepAliveHandler.__init__(self)
        if not ssl_factory:
            try:
                import sslfactory
                ssl_factory = sslfactory.get_factory()
            except ImportError:
                pass
        self._ssl_factory = ssl_factory

    def https_open(self, req):
        if False:
            print('Hello World!')
        return self.do_open(req)

    def _get_connection(self, host):
        if False:
            print('Hello World!')
        try:
            return self._ssl_factory.get_https_connection(host)
        except AttributeError:
            return HTTPSConnection(host)

class HTTPResponse(_http_client.HTTPResponse):

    def __init__(self, sock, debuglevel=0, strict=0, method=None):
        if False:
            for i in range(10):
                print('nop')
        if method:
            _http_client.HTTPResponse.__init__(self, sock, debuglevel, method)
        else:
            _http_client.HTTPResponse.__init__(self, sock, debuglevel)
        self.fileno = sock.fileno
        self.code = None
        self._method = method
        self._rbuf = b''
        self._rbufsize = 8096
        self._handler = None
        self._host = None
        self._url = None
        self._connection = None
    _raw_read = _http_client.HTTPResponse.read

    def close(self):
        if False:
            return 10
        if self.fp:
            self.fp.close()
            self.fp = None
            if self._handler:
                self._handler._request_closed(self, self._host, self._connection)

    def _close_conn(self):
        if False:
            i = 10
            return i + 15
        self.close()

    def close_connection(self):
        if False:
            while True:
                i = 10
        self._handler._remove_connection(self._host, self._connection, close=1)
        self.close()

    def info(self):
        if False:
            return 10
        return self.headers

    def geturl(self):
        if False:
            i = 10
            return i + 15
        return self._url

    def read(self, amt=None):
        if False:
            for i in range(10):
                print('nop')
        if self._rbuf and (not amt is None):
            L = len(self._rbuf)
            if amt > L:
                amt -= L
            else:
                s = self._rbuf[:amt]
                self._rbuf = self._rbuf[amt:]
                return s
        s = self._rbuf + self._raw_read(amt)
        self._rbuf = b''
        return s

    def readline(self, limit=-1):
        if False:
            for i in range(10):
                print('nop')
        data = b''
        i = self._rbuf.find('\n')
        while i < 0 and (not 0 < limit <= len(self._rbuf)):
            new = self._raw_read(self._rbufsize)
            if not new:
                break
            i = new.find('\n')
            if i >= 0:
                i = i + len(self._rbuf)
            self._rbuf = self._rbuf + new
        if i < 0:
            i = len(self._rbuf)
        else:
            i = i + 1
        if 0 <= limit < len(self._rbuf):
            i = limit
        (data, self._rbuf) = (self._rbuf[:i], self._rbuf[i:])
        return data

    def readlines(self, sizehint=0):
        if False:
            print('Hello World!')
        total = 0
        list = []
        while 1:
            line = self.readline()
            if not line:
                break
            list.append(line)
            total += len(line)
            if sizehint and total >= sizehint:
                break
        return list

class HTTPConnection(_http_client.HTTPConnection):
    response_class = HTTPResponse

class HTTPSConnection(_http_client.HTTPSConnection):
    response_class = HTTPResponse

def error_handler(url):
    if False:
        for i in range(10):
            print('nop')
    global HANDLE_ERRORS
    orig = HANDLE_ERRORS
    keepalive_handler = HTTPHandler()
    opener = _urllib.request.build_opener(keepalive_handler)
    _urllib.request.install_opener(opener)
    pos = {0: 'off', 1: 'on'}
    for i in (0, 1):
        print('  fancy error handling %s (HANDLE_ERRORS = %i)' % (pos[i], i))
        HANDLE_ERRORS = i
        try:
            fo = _urllib.request.urlopen(url)
            foo = fo.read()
            fo.close()
            try:
                (status, reason) = (fo.status, fo.reason)
            except AttributeError:
                (status, reason) = (None, None)
        except IOError as e:
            print('  EXCEPTION: %s' % e)
            raise
        else:
            print('  status = %s, reason = %s' % (status, reason))
    HANDLE_ERRORS = orig
    hosts = keepalive_handler.open_connections()
    print('open connections:', hosts)
    keepalive_handler.close_all()

def continuity(url):
    if False:
        i = 10
        return i + 15
    from hashlib import md5
    format = '%25s: %s'
    opener = _urllib.request.build_opener()
    _urllib.request.install_opener(opener)
    fo = _urllib.request.urlopen(url)
    foo = fo.read()
    fo.close()
    m = md5(foo)
    print(format % ('normal urllib', m.hexdigest()))
    opener = _urllib.request.build_opener(HTTPHandler())
    _urllib.request.install_opener(opener)
    fo = _urllib.request.urlopen(url)
    foo = fo.read()
    fo.close()
    m = md5(foo)
    print(format % ('keepalive read', m.hexdigest()))
    fo = _urllib.request.urlopen(url)
    foo = ''
    while 1:
        f = fo.readline()
        if f:
            foo = foo + f
        else:
            break
    fo.close()
    m = md5(foo)
    print(format % ('keepalive readline', m.hexdigest()))

def comp(N, url):
    if False:
        i = 10
        return i + 15
    print('  making %i connections to:\n  %s' % (N, url))
    sys.stdout.write('  first using the normal urllib handlers')
    opener = _urllib.request.build_opener()
    _urllib.request.install_opener(opener)
    t1 = fetch(N, url)
    print('  TIME: %.3f s' % t1)
    sys.stdout.write('  now using the keepalive handler       ')
    opener = _urllib.request.build_opener(HTTPHandler())
    _urllib.request.install_opener(opener)
    t2 = fetch(N, url)
    print('  TIME: %.3f s' % t2)
    print('  improvement factor: %.2f' % (t1 / t2,))

def fetch(N, url, delay=0):
    if False:
        return 10
    import time
    lens = []
    starttime = time.time()
    for i in _range(N):
        if delay and i > 0:
            time.sleep(delay)
        fo = _urllib.request.urlopen(url)
        foo = fo.read()
        fo.close()
        lens.append(len(foo))
    diff = time.time() - starttime
    j = 0
    for i in lens[1:]:
        j = j + 1
        if not i == lens[0]:
            print('WARNING: inconsistent length on read %i: %i' % (j, i))
    return diff

def test_timeout(url):
    if False:
        while True:
            i = 10
    global DEBUG
    dbbackup = DEBUG

    class FakeLogger:

        def debug(self, msg, *args):
            if False:
                i = 10
                return i + 15
            print(msg % args)
        info = warning = error = debug
    DEBUG = FakeLogger()
    print('  fetching the file to establish a connection')
    fo = _urllib.request.urlopen(url)
    data1 = fo.read()
    fo.close()
    i = 20
    print('  waiting %i seconds for the server to close the connection' % i)
    while i > 0:
        sys.stdout.write('\r  %2i' % i)
        sys.stdout.flush()
        time.sleep(1)
        i -= 1
    sys.stderr.write('\r')
    print('  fetching the file a second time')
    fo = _urllib.request.urlopen(url)
    data2 = fo.read()
    fo.close()
    if data1 == data2:
        print('  data are identical')
    else:
        print('  ERROR: DATA DIFFER')
    DEBUG = dbbackup

def test(url, N=10):
    if False:
        while True:
            i = 10
    print('checking error hander (do this on a non-200)')
    try:
        error_handler(url)
    except IOError as e:
        print('exiting - exception will prevent further tests')
        sys.exit()
    print()
    print("performing continuity test (making sure stuff isn't corrupted)")
    continuity(url)
    print()
    print('performing speed comparison')
    comp(N, url)
    print()
    print('performing dropped-connection check')
    test_timeout(url)
if __name__ == '__main__':
    import time
    import sys
    try:
        N = int(sys.argv[1])
        url = sys.argv[2]
    except:
        print('%s <integer> <url>' % sys.argv[0])
    else:
        test(url, N)