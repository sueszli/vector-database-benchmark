"""CherryPy Application and Tree objects."""
import os
import cherrypy
from cherrypy import _cpconfig, _cplogging, _cprequest, _cpwsgi, tools
from cherrypy.lib import httputil, reprconf

class Application(object):
    """A CherryPy Application.

    Servers and gateways should not instantiate Request objects directly.
    Instead, they should ask an Application object for a request object.

    An instance of this class may also be used as a WSGI callable
    (WSGI application object) for itself.
    """
    root = None
    'The top-most container of page handlers for this app. Handlers should\n    be arranged in a hierarchy of attributes, matching the expected URI\n    hierarchy; the default dispatcher then searches this hierarchy for a\n    matching handler. When using a dispatcher other than the default,\n    this value may be None.'
    config = {}
    "A dict of {path: pathconf} pairs, where 'pathconf' is itself a dict\n    of {key: value} pairs."
    namespaces = reprconf.NamespaceSet()
    toolboxes = {'tools': cherrypy.tools}
    log = None
    'A LogManager instance. See _cplogging.'
    wsgiapp = None
    'A CPWSGIApp instance. See _cpwsgi.'
    request_class = _cprequest.Request
    response_class = _cprequest.Response
    relative_urls = False

    def __init__(self, root, script_name='', config=None):
        if False:
            i = 10
            return i + 15
        'Initialize Application with given root.'
        self.log = _cplogging.LogManager(id(self), cherrypy.log.logger_root)
        self.root = root
        self.script_name = script_name
        self.wsgiapp = _cpwsgi.CPWSGIApp(self)
        self.namespaces = self.namespaces.copy()
        self.namespaces['log'] = lambda k, v: setattr(self.log, k, v)
        self.namespaces['wsgi'] = self.wsgiapp.namespace_handler
        self.config = self.__class__.config.copy()
        if config:
            self.merge(config)

    def __repr__(self):
        if False:
            return 10
        'Generate a representation of the Application instance.'
        return '%s.%s(%r, %r)' % (self.__module__, self.__class__.__name__, self.root, self.script_name)
    script_name_doc = 'The URI "mount point" for this app. A mount point\n    is that portion of the URI which is constant for all URIs that are\n    serviced by this application; it does not include scheme, host, or proxy\n    ("virtual host") portions of the URI.\n\n    For example, if script_name is "/my/cool/app", then the URL\n    "http://www.example.com/my/cool/app/page1" might be handled by a\n    "page1" method on the root object.\n\n    The value of script_name MUST NOT end in a slash. If the script_name\n    refers to the root of the URI, it MUST be an empty string (not "/").\n\n    If script_name is explicitly set to None, then the script_name will be\n    provided for each call from request.wsgi_environ[\'SCRIPT_NAME\'].\n    '

    @property
    def script_name(self):
        if False:
            while True:
                i = 10
        'The URI "mount point" for this app.\n\n        A mount point is that portion of the URI which is constant for all URIs\n        that are serviced by this application; it does not include scheme,\n        host, or proxy ("virtual host") portions of the URI.\n\n        For example, if script_name is "/my/cool/app", then the URL\n        "http://www.example.com/my/cool/app/page1" might be handled by a\n        "page1" method on the root object.\n\n        The value of script_name MUST NOT end in a slash. If the script_name\n        refers to the root of the URI, it MUST be an empty string (not "/").\n\n        If script_name is explicitly set to None, then the script_name will be\n        provided for each call from request.wsgi_environ[\'SCRIPT_NAME\'].\n        '
        if self._script_name is not None:
            return self._script_name
        return cherrypy.serving.request.wsgi_environ['SCRIPT_NAME'].rstrip('/')

    @script_name.setter
    def script_name(self, value):
        if False:
            return 10
        if value:
            value = value.rstrip('/')
        self._script_name = value

    def merge(self, config):
        if False:
            while True:
                i = 10
        'Merge the given config into self.config.'
        _cpconfig.merge(self.config, config)
        self.namespaces(self.config.get('/', {}))

    def find_config(self, path, key, default=None):
        if False:
            while True:
                i = 10
        'Return the most-specific value for key along path, or default.'
        trail = path or '/'
        while trail:
            nodeconf = self.config.get(trail, {})
            if key in nodeconf:
                return nodeconf[key]
            lastslash = trail.rfind('/')
            if lastslash == -1:
                break
            elif lastslash == 0 and trail != '/':
                trail = '/'
            else:
                trail = trail[:lastslash]
        return default

    def get_serving(self, local, remote, scheme, sproto):
        if False:
            return 10
        'Create and return a Request and Response object.'
        req = self.request_class(local, remote, scheme, sproto)
        req.app = self
        for (name, toolbox) in self.toolboxes.items():
            req.namespaces[name] = toolbox
        resp = self.response_class()
        cherrypy.serving.load(req, resp)
        cherrypy.engine.publish('acquire_thread')
        cherrypy.engine.publish('before_request')
        return (req, resp)

    def release_serving(self):
        if False:
            while True:
                i = 10
        'Release the current serving (request and response).'
        req = cherrypy.serving.request
        cherrypy.engine.publish('after_request')
        try:
            req.close()
        except Exception:
            cherrypy.log(traceback=True, severity=40)
        cherrypy.serving.clear()

    def __call__(self, environ, start_response):
        if False:
            print('Hello World!')
        'Call a WSGI-callable.'
        return self.wsgiapp(environ, start_response)

class Tree(object):
    """A registry of CherryPy applications, mounted at diverse points.

    An instance of this class may also be used as a WSGI callable
    (WSGI application object), in which case it dispatches to all
    mounted apps.
    """
    apps = {}
    '\n    A dict of the form {script name: application}, where "script name"\n    is a string declaring the URI mount point (no trailing slash), and\n    "application" is an instance of cherrypy.Application (or an arbitrary\n    WSGI callable if you happen to be using a WSGI server).'

    def __init__(self):
        if False:
            print('Hello World!')
        'Initialize registry Tree.'
        self.apps = {}

    def mount(self, root, script_name='', config=None):
        if False:
            return 10
        'Mount a new app from a root object, script_name, and config.\n\n        root\n            An instance of a "controller class" (a collection of page\n            handler methods) which represents the root of the application.\n            This may also be an Application instance, or None if using\n            a dispatcher other than the default.\n\n        script_name\n            A string containing the "mount point" of the application.\n            This should start with a slash, and be the path portion of the\n            URL at which to mount the given root. For example, if root.index()\n            will handle requests to "http://www.example.com:8080/dept/app1/",\n            then the script_name argument would be "/dept/app1".\n\n            It MUST NOT end in a slash. If the script_name refers to the\n            root of the URI, it MUST be an empty string (not "/").\n\n        config\n            A file or dict containing application config.\n        '
        if script_name is None:
            raise TypeError("The 'script_name' argument may not be None. Application objects may, however, possess a script_name of None (in order to inpect the WSGI environ for SCRIPT_NAME upon each request). You cannot mount such Applications on this Tree; you must pass them to a WSGI server interface directly.")
        script_name = script_name.rstrip('/')
        if isinstance(root, Application):
            app = root
            if script_name != '' and script_name != app.script_name:
                raise ValueError('Cannot specify a different script name and pass an Application instance to cherrypy.mount')
            script_name = app.script_name
        else:
            app = Application(root, script_name)
            needs_favicon = script_name == '' and root is not None and (not hasattr(root, 'favicon_ico'))
            if needs_favicon:
                favicon = os.path.join(os.getcwd(), os.path.dirname(__file__), 'favicon.ico')
                root.favicon_ico = tools.staticfile.handler(favicon)
        if config:
            app.merge(config)
        self.apps[script_name] = app
        return app

    def graft(self, wsgi_callable, script_name=''):
        if False:
            print('Hello World!')
        'Mount a wsgi callable at the given script_name.'
        script_name = script_name.rstrip('/')
        self.apps[script_name] = wsgi_callable

    def script_name(self, path=None):
        if False:
            for i in range(10):
                print('nop')
        'Return the script_name of the app at the given path, or None.\n\n        If path is None, cherrypy.request is used.\n        '
        if path is None:
            try:
                request = cherrypy.serving.request
                path = httputil.urljoin(request.script_name, request.path_info)
            except AttributeError:
                return None
        while True:
            if path in self.apps:
                return path
            if path == '':
                return None
            path = path[:path.rfind('/')]

    def __call__(self, environ, start_response):
        if False:
            return 10
        'Pre-initialize WSGI env and call WSGI-callable.'
        env1x = environ
        path = httputil.urljoin(env1x.get('SCRIPT_NAME', ''), env1x.get('PATH_INFO', ''))
        sn = self.script_name(path or '/')
        if sn is None:
            start_response('404 Not Found', [])
            return []
        app = self.apps[sn]
        environ = environ.copy()
        environ['SCRIPT_NAME'] = sn
        environ['PATH_INFO'] = path[len(sn.rstrip('/')):]
        return app(environ, start_response)