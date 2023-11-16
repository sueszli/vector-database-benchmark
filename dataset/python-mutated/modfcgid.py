"""Wrapper for mod_fcgid, for use as a CherryPy HTTP server when testing.

To autostart fcgid, the "apache" executable or script must be
on your system path, or you must override the global APACHE_PATH.
On some platforms, "apache" may be called "apachectl", "apache2ctl",
or "httpd"--create a symlink to them if needed.

You'll also need the WSGIServer from flup.servers.
See http://projects.amor.org/misc/wiki/ModPythonGateway


KNOWN BUGS
==========

1. Apache processes Range headers automatically; CherryPy's truncated
    output is then truncated again by Apache. See test_core.testRanges.
    This was worked around in http://www.cherrypy.dev/changeset/1319.
2. Apache does not allow custom HTTP methods like CONNECT as per the spec.
    See test_core.testHTTPMethods.
3. Max request header and body settings do not work with Apache.
4. Apache replaces status "reason phrases" automatically. For example,
    CherryPy may set "304 Not modified" but Apache will write out
    "304 Not Modified" (capital "M").
5. Apache does not allow custom error codes as per the spec.
6. Apache (or perhaps modpython, or modpython_gateway) unquotes %xx in the
    Request-URI too early.
7. mod_python will not read request bodies which use the "chunked"
    transfer-coding (it passes REQUEST_CHUNKED_ERROR to ap_setup_client_block
    instead of REQUEST_CHUNKED_DECHUNK, see Apache2's http_protocol.c and
    mod_python's requestobject.c).
8. Apache will output a "Content-Length: 0" response header even if there's
    no response entity body. This isn't really a bug; it just differs from
    the CherryPy default.
"""
import os
import re
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.process import servers
from cherrypy.test import helper
curdir = os.path.join(os.getcwd(), os.path.dirname(__file__))

def read_process(cmd, args=''):
    if False:
        for i in range(10):
            print('nop')
    (pipein, pipeout) = os.popen4('%s %s' % (cmd, args))
    try:
        firstline = pipeout.readline()
        if re.search('(not recognized|No such file|not found)', firstline, re.IGNORECASE):
            raise IOError('%s must be on your system path.' % cmd)
        output = firstline + pipeout.read()
    finally:
        pipeout.close()
    return output
APACHE_PATH = 'httpd'
CONF_PATH = 'fcgi.conf'
conf_fcgid = '\n# Apache2 server conf file for testing CherryPy with mod_fcgid.\n\nDocumentRoot "%(root)s"\nServerName 127.0.0.1\nListen %(port)s\nLoadModule fastcgi_module modules/mod_fastcgi.dll\nLoadModule rewrite_module modules/mod_rewrite.so\n\nOptions ExecCGI\nSetHandler fastcgi-script\nRewriteEngine On\nRewriteRule ^(.*)$ /fastcgi.pyc [L]\nFastCgiExternalServer "%(server)s" -host 127.0.0.1:4000\n'

class ModFCGISupervisor(helper.LocalSupervisor):
    using_apache = True
    using_wsgi = True
    template = conf_fcgid

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'FCGI Server on %s:%s' % (self.host, self.port)

    def start(self, modulename):
        if False:
            i = 10
            return i + 15
        cherrypy.server.httpserver = servers.FlupFCGIServer(application=cherrypy.tree, bindAddress=('127.0.0.1', 4000))
        cherrypy.server.httpserver.bind_addr = ('127.0.0.1', 4000)
        self.start_apache()
        helper.LocalServer.start(self, modulename)

    def start_apache(self):
        if False:
            i = 10
            return i + 15
        fcgiconf = CONF_PATH
        if not os.path.isabs(fcgiconf):
            fcgiconf = os.path.join(curdir, fcgiconf)
        with open(fcgiconf, 'wb') as f:
            server = repr(os.path.join(curdir, 'fastcgi.pyc'))[1:-1]
            output = self.template % {'port': self.port, 'root': curdir, 'server': server}
            output = ntob(output.replace('\r\n', '\n'))
            f.write(output)
        result = read_process(APACHE_PATH, '-k start -f %s' % fcgiconf)
        if result:
            print(result)

    def stop(self):
        if False:
            while True:
                i = 10
        'Gracefully shutdown a server that is serving forever.'
        read_process(APACHE_PATH, '-k stop')
        helper.LocalServer.stop(self)

    def sync_apps(self):
        if False:
            print('Hello World!')
        cherrypy.server.httpserver.fcgiserver.application = self.get_app()