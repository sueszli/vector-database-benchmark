"""Wrapper for mod_fastcgi, for use as a CherryPy HTTP server when testing.

To autostart fastcgi, the "apache" executable or script must be
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
from cherrypy.process import servers
from cherrypy.test import helper
curdir = os.path.join(os.getcwd(), os.path.dirname(__file__))

def read_process(cmd, args=''):
    if False:
        return 10
    (pipein, pipeout) = os.popen4('%s %s' % (cmd, args))
    try:
        firstline = pipeout.readline()
        if re.search('(not recognized|No such file|not found)', firstline, re.IGNORECASE):
            raise IOError('%s must be on your system path.' % cmd)
        output = firstline + pipeout.read()
    finally:
        pipeout.close()
    return output
APACHE_PATH = 'apache2ctl'
CONF_PATH = 'fastcgi.conf'
conf_fastcgi = '\n# Apache2 server conf file for testing CherryPy with mod_fastcgi.\n# fumanchu: I had to hard-code paths due to crazy Debian layouts :(\nServerRoot /usr/lib/apache2\nUser #1000\nErrorLog %(root)s/mod_fastcgi.error.log\n\nDocumentRoot "%(root)s"\nServerName 127.0.0.1\nListen %(port)s\nLoadModule fastcgi_module modules/mod_fastcgi.so\nLoadModule rewrite_module modules/mod_rewrite.so\n\nOptions +ExecCGI\nSetHandler fastcgi-script\nRewriteEngine On\nRewriteRule ^(.*)$ /fastcgi.pyc [L]\nFastCgiExternalServer "%(server)s" -host 127.0.0.1:4000\n'

def erase_script_name(environ, start_response):
    if False:
        print('Hello World!')
    environ['SCRIPT_NAME'] = ''
    return cherrypy.tree(environ, start_response)

class ModFCGISupervisor(helper.LocalWSGISupervisor):
    httpserver_class = 'cherrypy.process.servers.FlupFCGIServer'
    using_apache = True
    using_wsgi = True
    template = conf_fastcgi

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'FCGI Server on %s:%s' % (self.host, self.port)

    def start(self, modulename):
        if False:
            for i in range(10):
                print('nop')
        cherrypy.server.httpserver = servers.FlupFCGIServer(application=erase_script_name, bindAddress=('127.0.0.1', 4000))
        cherrypy.server.httpserver.bind_addr = ('127.0.0.1', 4000)
        cherrypy.server.socket_port = 4000
        self.start_apache()
        cherrypy.engine.start()
        self.sync_apps()

    def start_apache(self):
        if False:
            for i in range(10):
                print('nop')
        fcgiconf = CONF_PATH
        if not os.path.isabs(fcgiconf):
            fcgiconf = os.path.join(curdir, fcgiconf)
        with open(fcgiconf, 'wb') as f:
            server = repr(os.path.join(curdir, 'fastcgi.pyc'))[1:-1]
            output = self.template % {'port': self.port, 'root': curdir, 'server': server}
            output = output.replace('\r\n', '\n')
            f.write(output)
        result = read_process(APACHE_PATH, '-k start -f %s' % fcgiconf)
        if result:
            print(result)

    def stop(self):
        if False:
            i = 10
            return i + 15
        'Gracefully shutdown a server that is serving forever.'
        read_process(APACHE_PATH, '-k stop')
        helper.LocalWSGISupervisor.stop(self)

    def sync_apps(self):
        if False:
            for i in range(10):
                print('nop')
        cherrypy.server.httpserver.fcgiserver.application = self.get_app(erase_script_name)