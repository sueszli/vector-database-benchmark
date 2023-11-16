"""Wrapper for mod_python, for use as a CherryPy HTTP server when testing.

To autostart modpython, the "apache" executable or script must be
on your system path, or you must override the global APACHE_PATH.
On some platforms, "apache" may be called "apachectl" or "apache2ctl"--
create a symlink to them if needed.

If you wish to test the WSGI interface instead of our _cpmodpy interface,
you also need the 'modpython_gateway' module at:
http://projects.amor.org/misc/wiki/ModPythonGateway


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
from cherrypy.test import helper
curdir = os.path.join(os.getcwd(), os.path.dirname(__file__))

def read_process(cmd, args=''):
    if False:
        print('Hello World!')
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
CONF_PATH = 'test_mp.conf'
conf_modpython_gateway = '\n# Apache2 server conf file for testing CherryPy with modpython_gateway.\n\nServerName 127.0.0.1\nDocumentRoot "/"\nListen %(port)s\nLoadModule python_module modules/mod_python.so\n\nSetHandler python-program\nPythonFixupHandler cherrypy.test.modpy::wsgisetup\nPythonOption testmod %(modulename)s\nPythonHandler modpython_gateway::handler\nPythonOption wsgi.application cherrypy::tree\nPythonOption socket_host %(host)s\nPythonDebug On\n'
conf_cpmodpy = '\n# Apache2 server conf file for testing CherryPy with _cpmodpy.\n\nServerName 127.0.0.1\nDocumentRoot "/"\nListen %(port)s\nLoadModule python_module modules/mod_python.so\n\nSetHandler python-program\nPythonFixupHandler cherrypy.test.modpy::cpmodpysetup\nPythonHandler cherrypy._cpmodpy::handler\nPythonOption cherrypy.setup cherrypy.test.%(modulename)s::setup_server\nPythonOption socket_host %(host)s\nPythonDebug On\n'

class ModPythonSupervisor(helper.Supervisor):
    using_apache = True
    using_wsgi = False
    template = None

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'ModPython Server on %s:%s' % (self.host, self.port)

    def start(self, modulename):
        if False:
            print('Hello World!')
        mpconf = CONF_PATH
        if not os.path.isabs(mpconf):
            mpconf = os.path.join(curdir, mpconf)
        with open(mpconf, 'wb') as f:
            f.write(self.template % {'port': self.port, 'modulename': modulename, 'host': self.host})
        result = read_process(APACHE_PATH, '-k start -f %s' % mpconf)
        if result:
            print(result)

    def stop(self):
        if False:
            while True:
                i = 10
        'Gracefully shutdown a server that is serving forever.'
        read_process(APACHE_PATH, '-k stop')
loaded = False

def wsgisetup(req):
    if False:
        return 10
    global loaded
    if not loaded:
        loaded = True
        options = req.get_options()
        cherrypy.config.update({'log.error_file': os.path.join(curdir, 'test.log'), 'environment': 'test_suite', 'server.socket_host': options['socket_host']})
        modname = options['testmod']
        mod = __import__(modname, globals(), locals(), [''])
        mod.setup_server()
        cherrypy.server.unsubscribe()
        cherrypy.engine.start()
    from mod_python import apache
    return apache.OK

def cpmodpysetup(req):
    if False:
        i = 10
        return i + 15
    global loaded
    if not loaded:
        loaded = True
        options = req.get_options()
        cherrypy.config.update({'log.error_file': os.path.join(curdir, 'test.log'), 'environment': 'test_suite', 'server.socket_host': options['socket_host']})
    from mod_python import apache
    return apache.OK