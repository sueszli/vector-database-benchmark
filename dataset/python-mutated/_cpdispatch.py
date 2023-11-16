"""CherryPy dispatchers.

A 'dispatcher' is the object which looks up the 'page handler' callable
and collects config for the current request based on the path_info, other
request attributes, and the application architecture. The core calls the
dispatcher as early as possible, passing it a 'path_info' argument.

The default dispatcher discovers the page handler by matching path_info
to a hierarchical arrangement of objects, starting at request.app.root.
"""
import string
import sys
import types
try:
    classtype = (type, types.ClassType)
except AttributeError:
    classtype = type
import cherrypy

class PageHandler(object):
    """Callable which sets response.body."""

    def __init__(self, callable, *args, **kwargs):
        if False:
            return 10
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    @property
    def args(self):
        if False:
            return 10
        'The ordered args should be accessible from post dispatch hooks.'
        return cherrypy.serving.request.args

    @args.setter
    def args(self, args):
        if False:
            i = 10
            return i + 15
        cherrypy.serving.request.args = args
        return cherrypy.serving.request.args

    @property
    def kwargs(self):
        if False:
            print('Hello World!')
        'The named kwargs should be accessible from post dispatch hooks.'
        return cherrypy.serving.request.kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        if False:
            for i in range(10):
                print('nop')
        cherrypy.serving.request.kwargs = kwargs
        return cherrypy.serving.request.kwargs

    def __call__(self):
        if False:
            while True:
                i = 10
        try:
            return self.callable(*self.args, **self.kwargs)
        except TypeError:
            x = sys.exc_info()[1]
            try:
                test_callable_spec(self.callable, self.args, self.kwargs)
            except cherrypy.HTTPError:
                raise sys.exc_info()[1]
            except Exception:
                raise x
            raise

def test_callable_spec(callable, callable_args, callable_kwargs):
    if False:
        return 10
    "\n    Inspect callable and test to see if the given args are suitable for it.\n\n    When an error occurs during the handler's invoking stage there are 2\n    erroneous cases:\n    1.  Too many parameters passed to a function which doesn't define\n        one of *args or **kwargs.\n    2.  Too little parameters are passed to the function.\n\n    There are 3 sources of parameters to a cherrypy handler.\n    1.  query string parameters are passed as keyword parameters to the\n        handler.\n    2.  body parameters are also passed as keyword parameters.\n    3.  when partial matching occurs, the final path atoms are passed as\n        positional args.\n    Both the query string and path atoms are part of the URI.  If they are\n    incorrect, then a 404 Not Found should be raised. Conversely the body\n    parameters are part of the request; if they are invalid a 400 Bad Request.\n    "
    show_mismatched_params = getattr(cherrypy.serving.request, 'show_mismatched_params', False)
    try:
        (args, varargs, varkw, defaults) = getargspec(callable)
    except TypeError:
        if isinstance(callable, object) and hasattr(callable, '__call__'):
            (args, varargs, varkw, defaults) = getargspec(callable.__call__)
        else:
            raise
    if args and (hasattr(callable, '__call__') or inspect.ismethod(callable)):
        args = args[1:]
    arg_usage = dict([(arg, 0) for arg in args])
    vararg_usage = 0
    varkw_usage = 0
    extra_kwargs = set()
    for (i, value) in enumerate(callable_args):
        try:
            arg_usage[args[i]] += 1
        except IndexError:
            vararg_usage += 1
    for key in callable_kwargs.keys():
        try:
            arg_usage[key] += 1
        except KeyError:
            varkw_usage += 1
            extra_kwargs.add(key)
    args_with_defaults = args[-len(defaults or []):]
    for (i, val) in enumerate(defaults or []):
        if arg_usage[args_with_defaults[i]] == 0:
            arg_usage[args_with_defaults[i]] += 1
    missing_args = []
    multiple_args = []
    for (key, usage) in arg_usage.items():
        if usage == 0:
            missing_args.append(key)
        elif usage > 1:
            multiple_args.append(key)
    if missing_args:
        message = None
        if show_mismatched_params:
            message = 'Missing parameters: %s' % ','.join(missing_args)
        raise cherrypy.HTTPError(404, message=message)
    if not varargs and vararg_usage > 0:
        raise cherrypy.HTTPError(404)
    body_params = cherrypy.serving.request.body.params or {}
    body_params = set(body_params.keys())
    qs_params = set(callable_kwargs.keys()) - body_params
    if multiple_args:
        if qs_params.intersection(set(multiple_args)):
            error = 404
        else:
            error = 400
        message = None
        if show_mismatched_params:
            message = 'Multiple values for parameters: %s' % ','.join(multiple_args)
        raise cherrypy.HTTPError(error, message=message)
    if not varkw and varkw_usage > 0:
        extra_qs_params = set(qs_params).intersection(extra_kwargs)
        if extra_qs_params:
            message = None
            if show_mismatched_params:
                message = 'Unexpected query string parameters: %s' % ', '.join(extra_qs_params)
            raise cherrypy.HTTPError(404, message=message)
        extra_body_params = set(body_params).intersection(extra_kwargs)
        if extra_body_params:
            message = None
            if show_mismatched_params:
                message = 'Unexpected body parameters: %s' % ', '.join(extra_body_params)
            raise cherrypy.HTTPError(400, message=message)
try:
    import inspect
except ImportError:

    def test_callable_spec(callable, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        return None
else:

    def getargspec(callable):
        if False:
            i = 10
            return i + 15
        return inspect.getfullargspec(callable)[:4]

class LateParamPageHandler(PageHandler):
    """When passing cherrypy.request.params to the page handler, we do not
    want to capture that dict too early; we want to give tools like the
    decoding tool a chance to modify the params dict in-between the lookup
    of the handler and the actual calling of the handler. This subclass
    takes that into account, and allows request.params to be 'bound late'
    (it's more complicated than that, but that's the effect).
    """

    @property
    def kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Page handler kwargs (with cherrypy.request.params copied in).'
        kwargs = cherrypy.serving.request.params.copy()
        if self._kwargs:
            kwargs.update(self._kwargs)
        return kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        if False:
            while True:
                i = 10
        cherrypy.serving.request.kwargs = kwargs
        self._kwargs = kwargs
if sys.version_info < (3, 0):
    punctuation_to_underscores = string.maketrans(string.punctuation, '_' * len(string.punctuation))

    def validate_translator(t):
        if False:
            print('Hello World!')
        if not isinstance(t, str) or len(t) != 256:
            raise ValueError('The translate argument must be a str of len 256.')
else:
    punctuation_to_underscores = str.maketrans(string.punctuation, '_' * len(string.punctuation))

    def validate_translator(t):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(t, dict):
            raise ValueError('The translate argument must be a dict.')

class Dispatcher(object):
    """CherryPy Dispatcher which walks a tree of objects to find a handler.

    The tree is rooted at cherrypy.request.app.root, and each hierarchical
    component in the path_info argument is matched to a corresponding nested
    attribute of the root object. Matching handlers must have an 'exposed'
    attribute which evaluates to True. The special method name "index"
    matches a URI which ends in a slash ("/"). The special method name
    "default" may match a portion of the path_info (but only when no longer
    substring of the path_info matches some other object).

    This is the default, built-in dispatcher for CherryPy.
    """
    dispatch_method_name = '_cp_dispatch'
    '\n    The name of the dispatch method that nodes may optionally implement\n    to provide their own dynamic dispatch algorithm.\n    '

    def __init__(self, dispatch_method_name=None, translate=punctuation_to_underscores):
        if False:
            i = 10
            return i + 15
        validate_translator(translate)
        self.translate = translate
        if dispatch_method_name:
            self.dispatch_method_name = dispatch_method_name

    def __call__(self, path_info):
        if False:
            i = 10
            return i + 15
        'Set handler and config for the current request.'
        request = cherrypy.serving.request
        (func, vpath) = self.find_handler(path_info)
        if func:
            vpath = [x.replace('%2F', '/') for x in vpath]
            request.handler = LateParamPageHandler(func, *vpath)
        else:
            request.handler = cherrypy.NotFound()

    def find_handler(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Return the appropriate page handler, plus any virtual path.\n\n        This will return two objects. The first will be a callable,\n        which can be used to generate page output. Any parameters from\n        the query string or request body will be sent to that callable\n        as keyword arguments.\n\n        The callable is found by traversing the application\'s tree,\n        starting from cherrypy.request.app.root, and matching path\n        components to successive objects in the tree. For example, the\n        URL "/path/to/handler" might return root.path.to.handler.\n\n        The second object returned will be a list of names which are\n        \'virtual path\' components: parts of the URL which are dynamic,\n        and were not used when looking up the handler.\n        These virtual path components are passed to the handler as\n        positional arguments.\n        '
        request = cherrypy.serving.request
        app = request.app
        root = app.root
        dispatch_name = self.dispatch_method_name
        fullpath = [x for x in path.strip('/').split('/') if x] + ['index']
        fullpath_len = len(fullpath)
        segleft = fullpath_len
        nodeconf = {}
        if hasattr(root, '_cp_config'):
            nodeconf.update(root._cp_config)
        if '/' in app.config:
            nodeconf.update(app.config['/'])
        object_trail = [['root', root, nodeconf, segleft]]
        node = root
        iternames = fullpath[:]
        while iternames:
            name = iternames[0]
            objname = name.translate(self.translate)
            nodeconf = {}
            subnode = getattr(node, objname, None)
            pre_len = len(iternames)
            if subnode is None:
                dispatch = getattr(node, dispatch_name, None)
                if dispatch and hasattr(dispatch, '__call__') and (not getattr(dispatch, 'exposed', False)) and (pre_len > 1):
                    index_name = iternames.pop()
                    subnode = dispatch(vpath=iternames)
                    iternames.append(index_name)
                else:
                    iternames.pop(0)
            else:
                iternames.pop(0)
            segleft = len(iternames)
            if segleft > pre_len:
                raise cherrypy.CherryPyException('A vpath segment was added.  Custom dispatchers may only remove elements.  While trying to process {0} in {1}'.format(name, fullpath))
            elif segleft == pre_len:
                iternames.pop(0)
                segleft -= 1
            node = subnode
            if node is not None:
                if hasattr(node, '_cp_config'):
                    nodeconf.update(node._cp_config)
            existing_len = fullpath_len - pre_len
            if existing_len != 0:
                curpath = '/' + '/'.join(fullpath[0:existing_len])
            else:
                curpath = ''
            new_segs = fullpath[fullpath_len - pre_len:fullpath_len - segleft]
            for seg in new_segs:
                curpath += '/' + seg
                if curpath in app.config:
                    nodeconf.update(app.config[curpath])
            object_trail.append([name, node, nodeconf, segleft])

        def set_conf():
            if False:
                while True:
                    i = 10
            'Collapse all object_trail config into cherrypy.request.config.\n            '
            base = cherrypy.config.copy()
            for (name, obj, conf, segleft) in object_trail:
                base.update(conf)
                if 'tools.staticdir.dir' in conf:
                    base['tools.staticdir.section'] = '/' + '/'.join(fullpath[0:fullpath_len - segleft])
            return base
        num_candidates = len(object_trail) - 1
        for i in range(num_candidates, -1, -1):
            (name, candidate, nodeconf, segleft) = object_trail[i]
            if candidate is None:
                continue
            if hasattr(candidate, 'default'):
                defhandler = candidate.default
                if getattr(defhandler, 'exposed', False):
                    conf = getattr(defhandler, '_cp_config', {})
                    object_trail.insert(i + 1, ['default', defhandler, conf, segleft])
                    request.config = set_conf()
                    request.is_index = path.endswith('/')
                    return (defhandler, fullpath[fullpath_len - segleft:-1])
            if getattr(candidate, 'exposed', False):
                request.config = set_conf()
                if i == num_candidates:
                    request.is_index = True
                else:
                    request.is_index = False
                return (candidate, fullpath[fullpath_len - segleft:-1])
        request.config = set_conf()
        return (None, [])

class MethodDispatcher(Dispatcher):
    """Additional dispatch based on cherrypy.request.method.upper().

    Methods named GET, POST, etc will be called on an exposed class.
    The method names must be all caps; the appropriate Allow header
    will be output showing all capitalized method names as allowable
    HTTP verbs.

    Note that the containing class must be exposed, not the methods.
    """

    def __call__(self, path_info):
        if False:
            while True:
                i = 10
        'Set handler and config for the current request.'
        request = cherrypy.serving.request
        (resource, vpath) = self.find_handler(path_info)
        if resource:
            avail = [m for m in dir(resource) if m.isupper()]
            if 'GET' in avail and 'HEAD' not in avail:
                avail.append('HEAD')
            avail.sort()
            cherrypy.serving.response.headers['Allow'] = ', '.join(avail)
            meth = request.method.upper()
            func = getattr(resource, meth, None)
            if func is None and meth == 'HEAD':
                func = getattr(resource, 'GET', None)
            if func:
                if hasattr(func, '_cp_config'):
                    request.config.update(func._cp_config)
                vpath = [x.replace('%2F', '/') for x in vpath]
                request.handler = LateParamPageHandler(func, *vpath)
            else:
                request.handler = cherrypy.HTTPError(405)
        else:
            request.handler = cherrypy.NotFound()

class RoutesDispatcher(object):
    """A Routes based dispatcher for CherryPy."""

    def __init__(self, full_result=False, **mapper_options):
        if False:
            for i in range(10):
                print('nop')
        "\n        Routes dispatcher\n\n        Set full_result to True if you wish the controller\n        and the action to be passed on to the page handler\n        parameters. By default they won't be.\n        "
        import routes
        self.full_result = full_result
        self.controllers = {}
        self.mapper = routes.Mapper(**mapper_options)
        self.mapper.controller_scan = self.controllers.keys

    def connect(self, name, route, controller, **kwargs):
        if False:
            i = 10
            return i + 15
        self.controllers[name] = controller
        self.mapper.connect(name, route, controller=name, **kwargs)

    def redirect(self, url):
        if False:
            i = 10
            return i + 15
        raise cherrypy.HTTPRedirect(url)

    def __call__(self, path_info):
        if False:
            i = 10
            return i + 15
        'Set handler and config for the current request.'
        func = self.find_handler(path_info)
        if func:
            cherrypy.serving.request.handler = LateParamPageHandler(func)
        else:
            cherrypy.serving.request.handler = cherrypy.NotFound()

    def find_handler(self, path_info):
        if False:
            while True:
                i = 10
        'Find the right page handler, and set request.config.'
        import routes
        request = cherrypy.serving.request
        config = routes.request_config()
        config.mapper = self.mapper
        if hasattr(request, 'wsgi_environ'):
            config.environ = request.wsgi_environ
        config.host = request.headers.get('Host', None)
        config.protocol = request.scheme
        config.redirect = self.redirect
        result = self.mapper.match(path_info)
        config.mapper_dict = result
        params = {}
        if result:
            params = result.copy()
        if not self.full_result:
            params.pop('controller', None)
            params.pop('action', None)
        request.params.update(params)
        request.config = base = cherrypy.config.copy()
        curpath = ''

        def merge(nodeconf):
            if False:
                while True:
                    i = 10
            if 'tools.staticdir.dir' in nodeconf:
                nodeconf['tools.staticdir.section'] = curpath or '/'
            base.update(nodeconf)
        app = request.app
        root = app.root
        if hasattr(root, '_cp_config'):
            merge(root._cp_config)
        if '/' in app.config:
            merge(app.config['/'])
        atoms = [x for x in path_info.split('/') if x]
        if atoms:
            last = atoms.pop()
        else:
            last = None
        for atom in atoms:
            curpath = '/'.join((curpath, atom))
            if curpath in app.config:
                merge(app.config[curpath])
        handler = None
        if result:
            controller = result.get('controller')
            controller = self.controllers.get(controller, controller)
            if controller:
                if isinstance(controller, classtype):
                    controller = controller()
                if hasattr(controller, '_cp_config'):
                    merge(controller._cp_config)
            action = result.get('action')
            if action is not None:
                handler = getattr(controller, action, None)
                if hasattr(handler, '_cp_config'):
                    merge(handler._cp_config)
            else:
                handler = controller
        if last:
            curpath = '/'.join((curpath, last))
            if curpath in app.config:
                merge(app.config[curpath])
        return handler

def XMLRPCDispatcher(next_dispatcher=Dispatcher()):
    if False:
        i = 10
        return i + 15
    from cherrypy.lib import xmlrpcutil

    def xmlrpc_dispatch(path_info):
        if False:
            while True:
                i = 10
        path_info = xmlrpcutil.patched_path(path_info)
        return next_dispatcher(path_info)
    return xmlrpc_dispatch

def VirtualHost(next_dispatcher=Dispatcher(), use_x_forwarded_host=True, **domains):
    if False:
        return 10
    '\n    Select a different handler based on the Host header.\n\n    This can be useful when running multiple sites within one CP server.\n    It allows several domains to point to different parts of a single\n    website structure. For example::\n\n        http://www.domain.example  ->  root\n        http://www.domain2.example  ->  root/domain2/\n        http://www.domain2.example:443  ->  root/secure\n\n    can be accomplished via the following config::\n\n        [/]\n        request.dispatch = cherrypy.dispatch.VirtualHost(\n            **{\'www.domain2.example\': \'/domain2\',\n               \'www.domain2.example:443\': \'/secure\',\n              })\n\n    next_dispatcher\n        The next dispatcher object in the dispatch chain.\n        The VirtualHost dispatcher adds a prefix to the URL and calls\n        another dispatcher. Defaults to cherrypy.dispatch.Dispatcher().\n\n    use_x_forwarded_host\n        If True (the default), any "X-Forwarded-Host"\n        request header will be used instead of the "Host" header. This\n        is commonly added by HTTP servers (such as Apache) when proxying.\n\n    ``**domains``\n        A dict of {host header value: virtual prefix} pairs.\n        The incoming "Host" request header is looked up in this dict,\n        and, if a match is found, the corresponding "virtual prefix"\n        value will be prepended to the URL path before calling the\n        next dispatcher. Note that you often need separate entries\n        for "example.com" and "www.example.com". In addition, "Host"\n        headers may contain the port number.\n    '
    from cherrypy.lib import httputil

    def vhost_dispatch(path_info):
        if False:
            for i in range(10):
                print('nop')
        request = cherrypy.serving.request
        header = request.headers.get
        domain = header('Host', '')
        if use_x_forwarded_host:
            domain = header('X-Forwarded-Host', domain)
        prefix = domains.get(domain, '')
        if prefix:
            path_info = httputil.urljoin(prefix, path_info)
        result = next_dispatcher(path_info)
        section = request.config.get('tools.staticdir.section')
        if section:
            section = section[len(prefix):]
            request.config['tools.staticdir.section'] = section
        return result
    return vhost_dispatch