"""
Bottle is a fast and simple micro-framework for small web applications. It
offers request dispatching (Routes) with URL parameter support, templates,
a built-in HTTP Server and adapters for many third party WSGI/HTTP-server and
template engines - all in a single file and with no dependencies other than the
Python Standard Library.

Homepage and documentation: http://bottlepy.org/

Copyright (c) 2009-2018, Marcel Hellkamp.
License: MIT (see LICENSE for details)
"""
from __future__ import print_function
import sys
__author__ = 'Marcel Hellkamp'
__version__ = '0.13-dev'
__license__ = 'MIT'

def _cli_parse(args):
    if False:
        while True:
            i = 10
    from argparse import ArgumentParser
    parser = ArgumentParser(prog=args[0], usage='%(prog)s [options] package.module:app')
    opt = parser.add_argument
    opt('--version', action='store_true', help='show version number.')
    opt('-b', '--bind', metavar='ADDRESS', help='bind socket to ADDRESS.')
    opt('-s', '--server', default='wsgiref', help='use SERVER as backend.')
    opt('-p', '--plugin', action='append', help='install additional plugin/s.')
    opt('-c', '--conf', action='append', metavar='FILE', help='load config values from FILE.')
    opt('-C', '--param', action='append', metavar='NAME=VALUE', help='override config values.')
    opt('--debug', action='store_true', help='start server in debug mode.')
    opt('--reload', action='store_true', help='auto-reload on file changes.')
    opt('app', help='WSGI app entry point.', nargs='?')
    cli_args = parser.parse_args(args[1:])
    return (cli_args, parser)

def _cli_patch(cli_args):
    if False:
        while True:
            i = 10
    (parsed_args, _) = _cli_parse(cli_args)
    opts = parsed_args
    if opts.server:
        if opts.server.startswith('gevent'):
            import gevent.monkey
            gevent.monkey.patch_all()
        elif opts.server.startswith('eventlet'):
            import eventlet
            eventlet.monkey_patch()
if __name__ == '__main__':
    _cli_patch(sys.argv)
import base64, calendar, cgi, email.utils, functools, hmac, itertools, mimetypes, os, re, tempfile, threading, time, warnings, weakref, hashlib
from types import FunctionType
from datetime import date as datedate, datetime, timedelta
from tempfile import NamedTemporaryFile
from traceback import format_exc, print_exc
from unicodedata import normalize
try:
    from ujson import dumps as json_dumps, loads as json_lds
except ImportError:
    from json import dumps as json_dumps, loads as json_lds
py = sys.version_info
py3k = py.major > 2
if py3k:
    import http.client as httplib
    import _thread as thread
    from urllib.parse import urljoin, SplitResult as UrlSplitResult
    from urllib.parse import urlencode, quote as urlquote, unquote as urlunquote
    urlunquote = functools.partial(urlunquote, encoding='latin1')
    from http.cookies import SimpleCookie, Morsel, CookieError
    from collections.abc import MutableMapping as DictMixin
    from types import ModuleType as new_module
    import pickle
    from io import BytesIO
    import configparser
    from inspect import getfullargspec

    def getargspec(func):
        if False:
            print('Hello World!')
        spec = getfullargspec(func)
        kwargs = makelist(spec[0]) + makelist(spec.kwonlyargs)
        return (kwargs, spec[1], spec[2], spec[3])
    basestring = str
    unicode = str
    json_loads = lambda s: json_lds(touni(s))
    callable = lambda x: hasattr(x, '__call__')
    imap = map

    def _raise(*a):
        if False:
            i = 10
            return i + 15
        raise a[0](a[1]).with_traceback(a[2])
else:
    import httplib
    import thread
    from urlparse import urljoin, SplitResult as UrlSplitResult
    from urllib import urlencode, quote as urlquote, unquote as urlunquote
    from Cookie import SimpleCookie, Morsel, CookieError
    from itertools import imap
    import cPickle as pickle
    from imp import new_module
    from StringIO import StringIO as BytesIO
    import ConfigParser as configparser
    from collections import MutableMapping as DictMixin
    from inspect import getargspec
    unicode = unicode
    json_loads = json_lds
    exec(compile('def _raise(*a): raise a[0], a[1], a[2]', '<py3fix>', 'exec'))

def tob(s, enc='utf8'):
    if False:
        return 10
    if isinstance(s, unicode):
        return s.encode(enc)
    return b'' if s is None else bytes(s)

def touni(s, enc='utf8', err='strict'):
    if False:
        i = 10
        return i + 15
    if isinstance(s, bytes):
        return s.decode(enc, err)
    return unicode('' if s is None else s)
tonat = touni if py3k else tob

def _stderr(*args):
    if False:
        i = 10
        return i + 15
    try:
        print(*args, file=sys.stderr)
    except (IOError, AttributeError):
        pass

def update_wrapper(wrapper, wrapped, *a, **ka):
    if False:
        return 10
    try:
        functools.update_wrapper(wrapper, wrapped, *a, **ka)
    except AttributeError:
        pass

def depr(major, minor, cause, fix):
    if False:
        while True:
            i = 10
    text = 'Warning: Use of deprecated feature or API. (Deprecated in Bottle-%d.%d)\nCause: %s\nFix: %s\n' % (major, minor, cause, fix)
    if DEBUG == 'strict':
        raise DeprecationWarning(text)
    warnings.warn(text, DeprecationWarning, stacklevel=3)
    return DeprecationWarning(text)

def makelist(data):
    if False:
        print('Hello World!')
    if isinstance(data, (tuple, list, set, dict)):
        return list(data)
    elif data:
        return [data]
    else:
        return []

class DictProperty(object):
    """ Property that maps to a key in a local dict-like attribute. """

    def __init__(self, attr, key=None, read_only=False):
        if False:
            return 10
        (self.attr, self.key, self.read_only) = (attr, key, read_only)

    def __call__(self, func):
        if False:
            while True:
                i = 10
        functools.update_wrapper(self, func, updated=[])
        (self.getter, self.key) = (func, self.key or func.__name__)
        return self

    def __get__(self, obj, cls):
        if False:
            while True:
                i = 10
        if obj is None:
            return self
        (key, storage) = (self.key, getattr(obj, self.attr))
        if key not in storage:
            storage[key] = self.getter(obj)
        return storage[key]

    def __set__(self, obj, value):
        if False:
            for i in range(10):
                print('nop')
        if self.read_only:
            raise AttributeError('Read-Only property.')
        getattr(obj, self.attr)[self.key] = value

    def __delete__(self, obj):
        if False:
            print('Hello World!')
        if self.read_only:
            raise AttributeError('Read-Only property.')
        del getattr(obj, self.attr)[self.key]

class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property. """

    def __init__(self, func):
        if False:
            print('Hello World!')
        update_wrapper(self, func)
        self.func = func

    def __get__(self, obj, cls):
        if False:
            while True:
                i = 10
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

class lazy_attribute(object):
    """ A property that caches itself to the class object. """

    def __init__(self, func):
        if False:
            return 10
        functools.update_wrapper(self, func, updated=[])
        self.getter = func

    def __get__(self, obj, cls):
        if False:
            while True:
                i = 10
        value = self.getter(cls)
        setattr(cls, self.__name__, value)
        return value

class BottleException(Exception):
    """ A base class for exceptions used by bottle. """
    pass

class RouteError(BottleException):
    """ This is a base class for all routing related exceptions """

class RouteReset(BottleException):
    """ If raised by a plugin or request handler, the route is reset and all
        plugins are re-applied. """

class RouterUnknownModeError(RouteError):
    pass

class RouteSyntaxError(RouteError):
    """ The route parser found something not supported by this router. """

class RouteBuildError(RouteError):
    """ The route could not be built. """

def _re_flatten(p):
    if False:
        print('Hello World!')
    ' Turn all capturing groups in a regular expression pattern into\n        non-capturing groups. '
    if '(' not in p:
        return p
    return re.sub('(\\\\*)(\\(\\?P<[^>]+>|\\((?!\\?))', lambda m: m.group(0) if len(m.group(1)) % 2 else m.group(1) + '(?:', p)

class Router(object):
    """ A Router is an ordered collection of route->target pairs. It is used to
        efficiently match WSGI requests against a number of routes and return
        the first target that satisfies the request. The target may be anything,
        usually a string, ID or callable object. A route consists of a path-rule
        and a HTTP method.

        The path-rule is either a static path (e.g. `/contact`) or a dynamic
        path that contains wildcards (e.g. `/wiki/<page>`). The wildcard syntax
        and details on the matching order are described in docs:`routing`.
    """
    default_pattern = '[^/]+'
    default_filter = 're'
    _MAX_GROUPS_PER_PATTERN = 99

    def __init__(self, strict=False):
        if False:
            i = 10
            return i + 15
        self.rules = []
        self._groups = {}
        self.builder = {}
        self.static = {}
        self.dyna_routes = {}
        self.dyna_regexes = {}
        self.strict_order = strict
        self.filters = {'re': lambda conf: (_re_flatten(conf or self.default_pattern), None, None), 'int': lambda conf: ('-?\\d+', int, lambda x: str(int(x))), 'float': lambda conf: ('-?[\\d.]+', float, lambda x: str(float(x))), 'path': lambda conf: ('.+?', None, None)}

    def add_filter(self, name, func):
        if False:
            while True:
                i = 10
        ' Add a filter. The provided function is called with the configuration\n        string as parameter and must return a (regexp, to_python, to_url) tuple.\n        The first element is a string, the last two are callables or None. '
        self.filters[name] = func
    rule_syntax = re.compile('(\\\\*)(?:(?::([a-zA-Z_][a-zA-Z_0-9]*)?()(?:#(.*?)#)?)|(?:<([a-zA-Z_][a-zA-Z_0-9]*)?(?::([a-zA-Z_]*)(?::((?:\\\\.|[^\\\\>])+)?)?)?>))')

    def _itertokens(self, rule):
        if False:
            print('Hello World!')
        (offset, prefix) = (0, '')
        for match in self.rule_syntax.finditer(rule):
            prefix += rule[offset:match.start()]
            g = match.groups()
            if g[2] is not None:
                depr(0, 13, 'Use of old route syntax.', 'Use <name> instead of :name in routes.')
            if len(g[0]) % 2:
                prefix += match.group(0)[len(g[0]):]
                offset = match.end()
                continue
            if prefix:
                yield (prefix, None, None)
            (name, filtr, conf) = g[4:7] if g[2] is None else g[1:4]
            yield (name, filtr or 'default', conf or None)
            (offset, prefix) = (match.end(), '')
        if offset <= len(rule) or prefix:
            yield (prefix + rule[offset:], None, None)

    def add(self, rule, method, target, name=None):
        if False:
            return 10
        ' Add a new rule or replace the target for an existing rule. '
        anons = 0
        keys = []
        pattern = ''
        filters = []
        builder = []
        is_static = True
        for (key, mode, conf) in self._itertokens(rule):
            if mode:
                is_static = False
                if mode == 'default':
                    mode = self.default_filter
                (mask, in_filter, out_filter) = self.filters[mode](conf)
                if not key:
                    pattern += '(?:%s)' % mask
                    key = 'anon%d' % anons
                    anons += 1
                else:
                    pattern += '(?P<%s>%s)' % (key, mask)
                    keys.append(key)
                if in_filter:
                    filters.append((key, in_filter))
                builder.append((key, out_filter or str))
            elif key:
                pattern += re.escape(key)
                builder.append((None, key))
        self.builder[rule] = builder
        if name:
            self.builder[name] = builder
        if is_static and (not self.strict_order):
            self.static.setdefault(method, {})
            self.static[method][self.build(rule)] = (target, None)
            return
        try:
            re_pattern = re.compile('^(%s)$' % pattern)
            re_match = re_pattern.match
        except re.error as e:
            raise RouteSyntaxError('Could not add Route: %s (%s)' % (rule, e))
        if filters:

            def getargs(path):
                if False:
                    while True:
                        i = 10
                url_args = re_match(path).groupdict()
                for (name, wildcard_filter) in filters:
                    try:
                        url_args[name] = wildcard_filter(url_args[name])
                    except ValueError:
                        raise HTTPError(400, 'Path has wrong format.')
                return url_args
        elif re_pattern.groupindex:

            def getargs(path):
                if False:
                    return 10
                return re_match(path).groupdict()
        else:
            getargs = None
        flatpat = _re_flatten(pattern)
        whole_rule = (rule, flatpat, target, getargs)
        if (flatpat, method) in self._groups:
            if DEBUG:
                msg = 'Route <%s %s> overwrites a previously defined route'
                warnings.warn(msg % (method, rule), RuntimeWarning)
            self.dyna_routes[method][self._groups[flatpat, method]] = whole_rule
        else:
            self.dyna_routes.setdefault(method, []).append(whole_rule)
            self._groups[flatpat, method] = len(self.dyna_routes[method]) - 1
        self._compile(method)

    def _compile(self, method):
        if False:
            print('Hello World!')
        all_rules = self.dyna_routes[method]
        comborules = self.dyna_regexes[method] = []
        maxgroups = self._MAX_GROUPS_PER_PATTERN
        for x in range(0, len(all_rules), maxgroups):
            some = all_rules[x:x + maxgroups]
            combined = (flatpat for (_, flatpat, _, _) in some)
            combined = '|'.join(('(^%s$)' % flatpat for flatpat in combined))
            combined = re.compile(combined).match
            rules = [(target, getargs) for (_, _, target, getargs) in some]
            comborules.append((combined, rules))

    def build(self, _name, *anons, **query):
        if False:
            return 10
        ' Build an URL by filling the wildcards in a rule. '
        builder = self.builder.get(_name)
        if not builder:
            raise RouteBuildError('No route with that name.', _name)
        try:
            for (i, value) in enumerate(anons):
                query['anon%d' % i] = value
            url = ''.join([f(query.pop(n)) if n else f for (n, f) in builder])
            return url if not query else url + '?' + urlencode(query)
        except KeyError as E:
            raise RouteBuildError('Missing URL argument: %r' % E.args[0])

    def match(self, environ):
        if False:
            return 10
        ' Return a (target, url_args) tuple or raise HTTPError(400/404/405). '
        verb = environ['REQUEST_METHOD'].upper()
        path = environ['PATH_INFO'] or '/'
        methods = ('PROXY', 'HEAD', 'GET', 'ANY') if verb == 'HEAD' else ('PROXY', verb, 'ANY')
        for method in methods:
            if method in self.static and path in self.static[method]:
                (target, getargs) = self.static[method][path]
                return (target, getargs(path) if getargs else {})
            elif method in self.dyna_regexes:
                for (combined, rules) in self.dyna_regexes[method]:
                    match = combined(path)
                    if match:
                        (target, getargs) = rules[match.lastindex - 1]
                        return (target, getargs(path) if getargs else {})
        allowed = set([])
        nocheck = set(methods)
        for method in set(self.static) - nocheck:
            if path in self.static[method]:
                allowed.add(method)
        for method in set(self.dyna_regexes) - allowed - nocheck:
            for (combined, rules) in self.dyna_regexes[method]:
                match = combined(path)
                if match:
                    allowed.add(method)
        if allowed:
            allow_header = ','.join(sorted(allowed))
            raise HTTPError(405, 'Method not allowed.', Allow=allow_header)
        raise HTTPError(404, 'Not found: ' + repr(path))

class Route(object):
    """ This class wraps a route callback along with route specific metadata and
        configuration and applies Plugins on demand. It is also responsible for
        turning an URL path rule into a regular expression usable by the Router.
    """

    def __init__(self, app, rule, method, callback, name=None, plugins=None, skiplist=None, **config):
        if False:
            print('Hello World!')
        self.app = app
        self.rule = rule
        self.method = method
        self.callback = callback
        self.name = name or None
        self.plugins = plugins or []
        self.skiplist = skiplist or []
        self.config = app.config._make_overlay()
        self.config.load_dict(config)

    @cached_property
    def call(self):
        if False:
            while True:
                i = 10
        ' The route callback with all plugins applied. This property is\n            created on demand and then cached to speed up subsequent requests.'
        return self._make_callback()

    def reset(self):
        if False:
            i = 10
            return i + 15
        ' Forget any cached values. The next time :attr:`call` is accessed,\n            all plugins are re-applied. '
        self.__dict__.pop('call', None)

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        ' Do all on-demand work immediately (useful for debugging).'
        self.call

    def all_plugins(self):
        if False:
            return 10
        ' Yield all Plugins affecting this route. '
        unique = set()
        for p in reversed(self.app.plugins + self.plugins):
            if True in self.skiplist:
                break
            name = getattr(p, 'name', False)
            if name and (name in self.skiplist or name in unique):
                continue
            if p in self.skiplist or type(p) in self.skiplist:
                continue
            if name:
                unique.add(name)
            yield p

    def _make_callback(self):
        if False:
            print('Hello World!')
        callback = self.callback
        for plugin in self.all_plugins():
            try:
                if hasattr(plugin, 'apply'):
                    callback = plugin.apply(callback, self)
                else:
                    callback = plugin(callback)
            except RouteReset:
                return self._make_callback()
            if callback is not self.callback:
                update_wrapper(callback, self.callback)
        return callback

    def get_undecorated_callback(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return the callback. If the callback is a decorated function, try to\n            recover the original function. '
        func = self.callback
        func = getattr(func, '__func__' if py3k else 'im_func', func)
        closure_attr = '__closure__' if py3k else 'func_closure'
        while hasattr(func, closure_attr) and getattr(func, closure_attr):
            attributes = getattr(func, closure_attr)
            func = attributes[0].cell_contents
            if not isinstance(func, FunctionType):
                func = filter(lambda x: isinstance(x, FunctionType), map(lambda x: x.cell_contents, attributes))
                func = list(func)[0]
        return func

    def get_callback_args(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return a list of argument names the callback (most likely) accepts\n            as keyword arguments. If the callback is a decorated function, try\n            to recover the original function before inspection. '
        return getargspec(self.get_undecorated_callback())[0]

    def get_config(self, key, default=None):
        if False:
            return 10
        ' Lookup a config field and return its value, first checking the\n            route.config, then route.app.config.'
        depr(0, 13, 'Route.get_config() is deprecated.', 'The Route.config property already includes values from the application config for missing keys. Access it directly.')
        return self.config.get(key, default)

    def __repr__(self):
        if False:
            return 10
        cb = self.get_undecorated_callback()
        return '<%s %s -> %s:%s>' % (self.method, self.rule, cb.__module__, cb.__name__)

class Bottle(object):
    """ Each Bottle object represents a single, distinct web application and
        consists of routes, callbacks, plugins, resources and configuration.
        Instances are callable WSGI applications.

        :param catchall: If true (default), handle all exceptions. Turn off to
                         let debugging middleware handle exceptions.
    """

    @lazy_attribute
    def _global_config(cls):
        if False:
            return 10
        cfg = ConfigDict()
        cfg.meta_set('catchall', 'validate', bool)
        return cfg

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self.config = self._global_config._make_overlay()
        self.config._add_change_listener(functools.partial(self.trigger_hook, 'config'))
        self.config.update({'catchall': True})
        if kwargs.get('catchall') is False:
            depr(0, 13, 'Bottle(catchall) keyword argument.', "The 'catchall' setting is now part of the app configuration. Fix: `app.config['catchall'] = False`")
            self.config['catchall'] = False
        if kwargs.get('autojson') is False:
            depr(0, 13, 'Bottle(autojson) keyword argument.', "The 'autojson' setting is now part of the app configuration. Fix: `app.config['json.enable'] = False`")
            self.config['json.disable'] = True
        self._mounts = []
        self.resources = ResourceManager()
        self.routes = []
        self.router = Router()
        self.error_handler = {}
        self.plugins = []
        self.install(JSONPlugin())
        self.install(TemplatePlugin())
    catchall = DictProperty('config', 'catchall')
    __hook_names = ('before_request', 'after_request', 'app_reset', 'config')
    __hook_reversed = {'after_request'}

    @cached_property
    def _hooks(self):
        if False:
            return 10
        return dict(((name, []) for name in self.__hook_names))

    def add_hook(self, name, func):
        if False:
            return 10
        ' Attach a callback to a hook. Three hooks are currently implemented:\n\n            before_request\n                Executed once before each request. The request context is\n                available, but no routing has happened yet.\n            after_request\n                Executed once after each request regardless of its outcome.\n            app_reset\n                Called whenever :meth:`Bottle.reset` is called.\n        '
        if name in self.__hook_reversed:
            self._hooks[name].insert(0, func)
        else:
            self._hooks[name].append(func)

    def remove_hook(self, name, func):
        if False:
            for i in range(10):
                print('nop')
        ' Remove a callback from a hook. '
        if name in self._hooks and func in self._hooks[name]:
            self._hooks[name].remove(func)
            return True

    def trigger_hook(self, __name, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Trigger a hook and return a list of results. '
        return [hook(*args, **kwargs) for hook in self._hooks[__name][:]]

    def hook(self, name):
        if False:
            for i in range(10):
                print('nop')
        ' Return a decorator that attaches a callback to a hook. See\n            :meth:`add_hook` for details.'

        def decorator(func):
            if False:
                return 10
            self.add_hook(name, func)
            return func
        return decorator

    def _mount_wsgi(self, prefix, app, **options):
        if False:
            while True:
                i = 10
        segments = [p for p in prefix.split('/') if p]
        if not segments:
            raise ValueError('WSGI applications cannot be mounted to "/".')
        path_depth = len(segments)

        def mountpoint_wrapper():
            if False:
                print('Hello World!')
            try:
                request.path_shift(path_depth)
                rs = HTTPResponse([])

                def start_response(status, headerlist, exc_info=None):
                    if False:
                        for i in range(10):
                            print('nop')
                    if exc_info:
                        _raise(*exc_info)
                    if py3k:
                        status = status.encode('latin1').decode('utf8')
                        headerlist = [(k, v.encode('latin1').decode('utf8')) for (k, v) in headerlist]
                    rs.status = status
                    for (name, value) in headerlist:
                        rs.add_header(name, value)
                    return rs.body.append
                body = app(request.environ, start_response)
                rs.body = itertools.chain(rs.body, body) if rs.body else body
                return rs
            finally:
                request.path_shift(-path_depth)
        options.setdefault('skip', True)
        options.setdefault('method', 'PROXY')
        options.setdefault('mountpoint', {'prefix': prefix, 'target': app})
        options['callback'] = mountpoint_wrapper
        self.route('/%s/<:re:.*>' % '/'.join(segments), **options)
        if not prefix.endswith('/'):
            self.route('/' + '/'.join(segments), **options)

    def _mount_app(self, prefix, app, **options):
        if False:
            i = 10
            return i + 15
        if app in self._mounts or '_mount.app' in app.config:
            depr(0, 13, 'Application mounted multiple times. Falling back to WSGI mount.', 'Clone application before mounting to a different location.')
            return self._mount_wsgi(prefix, app, **options)
        if options:
            depr(0, 13, 'Unsupported mount options. Falling back to WSGI mount.', 'Do not specify any route options when mounting bottle application.')
            return self._mount_wsgi(prefix, app, **options)
        if not prefix.endswith('/'):
            depr(0, 13, "Prefix must end in '/'. Falling back to WSGI mount.", "Consider adding an explicit redirect from '/prefix' to '/prefix/' in the parent application.")
            return self._mount_wsgi(prefix, app, **options)
        self._mounts.append(app)
        app.config['_mount.prefix'] = prefix
        app.config['_mount.app'] = self
        for route in app.routes:
            route.rule = prefix + route.rule.lstrip('/')
            self.add_route(route)

    def mount(self, prefix, app, **options):
        if False:
            print('Hello World!')
        " Mount an application (:class:`Bottle` or plain WSGI) to a specific\n            URL prefix. Example::\n\n                parent_app.mount('/prefix/', child_app)\n\n            :param prefix: path prefix or `mount-point`.\n            :param app: an instance of :class:`Bottle` or a WSGI application.\n\n            Plugins from the parent application are not applied to the routes\n            of the mounted child application. If you need plugins in the child\n            application, install them separately.\n\n            While it is possible to use path wildcards within the prefix path\n            (:class:`Bottle` childs only), it is highly discouraged.\n\n            The prefix path must end with a slash. If you want to access the\n            root of the child application via `/prefix` in addition to\n            `/prefix/`, consider adding a route with a 307 redirect to the\n            parent application.\n        "
        if not prefix.startswith('/'):
            raise ValueError("Prefix must start with '/'")
        if isinstance(app, Bottle):
            return self._mount_app(prefix, app, **options)
        else:
            return self._mount_wsgi(prefix, app, **options)

    def merge(self, routes):
        if False:
            i = 10
            return i + 15
        " Merge the routes of another :class:`Bottle` application or a list of\n            :class:`Route` objects into this application. The routes keep their\n            'owner', meaning that the :data:`Route.app` attribute is not\n            changed. "
        if isinstance(routes, Bottle):
            routes = routes.routes
        for route in routes:
            self.add_route(route)

    def install(self, plugin):
        if False:
            i = 10
            return i + 15
        ' Add a plugin to the list of plugins and prepare it for being\n            applied to all routes of this application. A plugin may be a simple\n            decorator or an object that implements the :class:`Plugin` API.\n        '
        if hasattr(plugin, 'setup'):
            plugin.setup(self)
        if not callable(plugin) and (not hasattr(plugin, 'apply')):
            raise TypeError('Plugins must be callable or implement .apply()')
        self.plugins.append(plugin)
        self.reset()
        return plugin

    def uninstall(self, plugin):
        if False:
            for i in range(10):
                print('nop')
        ' Uninstall plugins. Pass an instance to remove a specific plugin, a type\n            object to remove all plugins that match that type, a string to remove\n            all plugins with a matching ``name`` attribute or ``True`` to remove all\n            plugins. Return the list of removed plugins. '
        (removed, remove) = ([], plugin)
        for (i, plugin) in list(enumerate(self.plugins))[::-1]:
            if remove is True or remove is plugin or remove is type(plugin) or (getattr(plugin, 'name', True) == remove):
                removed.append(plugin)
                del self.plugins[i]
                if hasattr(plugin, 'close'):
                    plugin.close()
        if removed:
            self.reset()
        return removed

    def reset(self, route=None):
        if False:
            i = 10
            return i + 15
        ' Reset all routes (force plugins to be re-applied) and clear all\n            caches. If an ID or route object is given, only that specific route\n            is affected. '
        if route is None:
            routes = self.routes
        elif isinstance(route, Route):
            routes = [route]
        else:
            routes = [self.routes[route]]
        for route in routes:
            route.reset()
        if DEBUG:
            for route in routes:
                route.prepare()
        self.trigger_hook('app_reset')

    def close(self):
        if False:
            print('Hello World!')
        ' Close the application and all installed plugins. '
        for plugin in self.plugins:
            if hasattr(plugin, 'close'):
                plugin.close()

    def run(self, **kwargs):
        if False:
            return 10
        ' Calls :func:`run` with the same parameters. '
        run(self, **kwargs)

    def match(self, environ):
        if False:
            i = 10
            return i + 15
        ' Search for a matching route and return a (:class:`Route`, urlargs)\n            tuple. The second value is a dictionary with parameters extracted\n            from the URL. Raise :exc:`HTTPError` (404/405) on a non-match.'
        return self.router.match(environ)

    def get_url(self, routename, **kargs):
        if False:
            while True:
                i = 10
        ' Return a string that matches a named route '
        scriptname = request.environ.get('SCRIPT_NAME', '').strip('/') + '/'
        location = self.router.build(routename, **kargs).lstrip('/')
        return urljoin(urljoin('/', scriptname), location)

    def add_route(self, route):
        if False:
            return 10
        ' Add a route object, but do not change the :data:`Route.app`\n            attribute.'
        self.routes.append(route)
        self.router.add(route.rule, route.method, route, name=route.name)
        if DEBUG:
            route.prepare()

    def route(self, path=None, method='GET', callback=None, name=None, apply=None, skip=None, **config):
        if False:
            print('Hello World!')
        " A decorator to bind a function to a request URL. Example::\n\n                @app.route('/hello/<name>')\n                def hello(name):\n                    return 'Hello %s' % name\n\n            The ``<name>`` part is a wildcard. See :class:`Router` for syntax\n            details.\n\n            :param path: Request path or a list of paths to listen to. If no\n              path is specified, it is automatically generated from the\n              signature of the function.\n            :param method: HTTP method (`GET`, `POST`, `PUT`, ...) or a list of\n              methods to listen to. (default: `GET`)\n            :param callback: An optional shortcut to avoid the decorator\n              syntax. ``route(..., callback=func)`` equals ``route(...)(func)``\n            :param name: The name for this route. (default: None)\n            :param apply: A decorator or plugin or a list of plugins. These are\n              applied to the route callback in addition to installed plugins.\n            :param skip: A list of plugins, plugin classes or names. Matching\n              plugins are not installed to this route. ``True`` skips all.\n\n            Any additional keyword arguments are stored as route-specific\n            configuration and passed to plugins (see :meth:`Plugin.apply`).\n        "
        if callable(path):
            (path, callback) = (None, path)
        plugins = makelist(apply)
        skiplist = makelist(skip)

        def decorator(callback):
            if False:
                while True:
                    i = 10
            if isinstance(callback, basestring):
                callback = load(callback)
            for rule in makelist(path) or yieldroutes(callback):
                for verb in makelist(method):
                    verb = verb.upper()
                    route = Route(self, rule, verb, callback, name=name, plugins=plugins, skiplist=skiplist, **config)
                    self.add_route(route)
            return callback
        return decorator(callback) if callback else decorator

    def get(self, path=None, method='GET', **options):
        if False:
            return 10
        ' Equals :meth:`route`. '
        return self.route(path, method, **options)

    def post(self, path=None, method='POST', **options):
        if False:
            while True:
                i = 10
        ' Equals :meth:`route` with a ``POST`` method parameter. '
        return self.route(path, method, **options)

    def put(self, path=None, method='PUT', **options):
        if False:
            i = 10
            return i + 15
        ' Equals :meth:`route` with a ``PUT`` method parameter. '
        return self.route(path, method, **options)

    def delete(self, path=None, method='DELETE', **options):
        if False:
            for i in range(10):
                print('nop')
        ' Equals :meth:`route` with a ``DELETE`` method parameter. '
        return self.route(path, method, **options)

    def patch(self, path=None, method='PATCH', **options):
        if False:
            return 10
        ' Equals :meth:`route` with a ``PATCH`` method parameter. '
        return self.route(path, method, **options)

    def error(self, code=500, callback=None):
        if False:
            return 10
        " Register an output handler for a HTTP error code. Can\n            be used as a decorator or called directly ::\n\n                def error_handler_500(error):\n                    return 'error_handler_500'\n\n                app.error(code=500, callback=error_handler_500)\n\n                @app.error(404)\n                def error_handler_404(error):\n                    return 'error_handler_404'\n\n        "

        def decorator(callback):
            if False:
                i = 10
                return i + 15
            if isinstance(callback, basestring):
                callback = load(callback)
            self.error_handler[int(code)] = callback
            return callback
        return decorator(callback) if callback else decorator

    def default_error_handler(self, res):
        if False:
            for i in range(10):
                print('nop')
        return tob(template(ERROR_PAGE_TEMPLATE, e=res, template_settings=dict(name='__ERROR_PAGE_TEMPLATE')))

    def _handle(self, environ):
        if False:
            print('Hello World!')
        path = environ['bottle.raw_path'] = environ['PATH_INFO']
        if py3k:
            environ['PATH_INFO'] = path.encode('latin1').decode('utf8', 'ignore')
        environ['bottle.app'] = self
        request.bind(environ)
        response.bind()
        try:
            while True:
                out = None
                try:
                    self.trigger_hook('before_request')
                    (route, args) = self.router.match(environ)
                    environ['route.handle'] = route
                    environ['bottle.route'] = route
                    environ['route.url_args'] = args
                    out = route.call(**args)
                    break
                except HTTPResponse as E:
                    out = E
                    break
                except RouteReset:
                    depr(0, 13, 'RouteReset exception deprecated', 'Call route.call() after route.reset() and return the result.')
                    route.reset()
                    continue
                finally:
                    if isinstance(out, HTTPResponse):
                        out.apply(response)
                    try:
                        self.trigger_hook('after_request')
                    except HTTPResponse as E:
                        out = E
                        out.apply(response)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as E:
            if not self.catchall:
                raise
            stacktrace = format_exc()
            environ['wsgi.errors'].write(stacktrace)
            environ['wsgi.errors'].flush()
            environ['bottle.exc_info'] = sys.exc_info()
            out = HTTPError(500, 'Internal Server Error', E, stacktrace)
            out.apply(response)
        return out

    def _cast(self, out, peek=None):
        if False:
            for i in range(10):
                print('nop')
        ' Try to convert the parameter into something WSGI compatible and set\n        correct HTTP headers when possible.\n        Support: False, str, unicode, dict, HTTPResponse, HTTPError, file-like,\n        iterable of strings and iterable of unicodes\n        '
        if not out:
            if 'Content-Length' not in response:
                response['Content-Length'] = 0
            return []
        if isinstance(out, (tuple, list)) and isinstance(out[0], (bytes, unicode)):
            out = out[0][0:0].join(out)
        if isinstance(out, unicode):
            out = out.encode(response.charset)
        if isinstance(out, bytes):
            if 'Content-Length' not in response:
                response['Content-Length'] = len(out)
            return [out]
        if isinstance(out, HTTPError):
            out.apply(response)
            out = self.error_handler.get(out.status_code, self.default_error_handler)(out)
            return self._cast(out)
        if isinstance(out, HTTPResponse):
            out.apply(response)
            return self._cast(out.body)
        if hasattr(out, 'read'):
            if 'wsgi.file_wrapper' in request.environ:
                return request.environ['wsgi.file_wrapper'](out)
            elif hasattr(out, 'close') or not hasattr(out, '__iter__'):
                return WSGIFileWrapper(out)
        try:
            iout = iter(out)
            first = next(iout)
            while not first:
                first = next(iout)
        except StopIteration:
            return self._cast('')
        except HTTPResponse as E:
            first = E
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as error:
            if not self.catchall:
                raise
            first = HTTPError(500, 'Unhandled exception', error, format_exc())
        if isinstance(first, HTTPResponse):
            return self._cast(first)
        elif isinstance(first, bytes):
            new_iter = itertools.chain([first], iout)
        elif isinstance(first, unicode):
            encoder = lambda x: x.encode(response.charset)
            new_iter = imap(encoder, itertools.chain([first], iout))
        else:
            msg = 'Unsupported response type: %s' % type(first)
            return self._cast(HTTPError(500, msg))
        if hasattr(out, 'close'):
            new_iter = _closeiter(new_iter, out.close)
        return new_iter

    def wsgi(self, environ, start_response):
        if False:
            return 10
        ' The bottle WSGI-interface. '
        try:
            out = self._cast(self._handle(environ))
            if response._status_code in (100, 101, 204, 304) or environ['REQUEST_METHOD'] == 'HEAD':
                if hasattr(out, 'close'):
                    out.close()
                out = []
            exc_info = environ.get('bottle.exc_info')
            if exc_info is not None:
                del environ['bottle.exc_info']
            start_response(response._wsgi_status_line(), response.headerlist, exc_info)
            return out
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as E:
            if not self.catchall:
                raise
            err = '<h1>Critical error while processing request: %s</h1>' % html_escape(environ.get('PATH_INFO', '/'))
            if DEBUG:
                err += '<h2>Error:</h2>\n<pre>\n%s\n</pre>\n<h2>Traceback:</h2>\n<pre>\n%s\n</pre>\n' % (html_escape(repr(E)), html_escape(format_exc()))
            environ['wsgi.errors'].write(err)
            environ['wsgi.errors'].flush()
            headers = [('Content-Type', 'text/html; charset=UTF-8')]
            start_response('500 INTERNAL SERVER ERROR', headers, sys.exc_info())
            return [tob(err)]

    def __call__(self, environ, start_response):
        if False:
            print('Hello World!')
        " Each instance of :class:'Bottle' is a WSGI application. "
        return self.wsgi(environ, start_response)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        ' Use this application as default for all module-level shortcuts. '
        default_app.push(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        default_app.pop()

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        if name in self.__dict__:
            raise AttributeError('Attribute %s already defined. Plugin conflict?' % name)
        self.__dict__[name] = value

class BaseRequest(object):
    """ A wrapper for WSGI environment dictionaries that adds a lot of
        convenient access methods and properties. Most of them are read-only.

        Adding new attributes to a request actually adds them to the environ
        dictionary (as 'bottle.request.ext.<name>'). This is the recommended
        way to store and access request-specific data.
    """
    __slots__ = ('environ',)
    MEMFILE_MAX = 102400

    def __init__(self, environ=None):
        if False:
            return 10
        ' Wrap a WSGI environ dictionary. '
        self.environ = {} if environ is None else environ
        self.environ['bottle.request'] = self

    @DictProperty('environ', 'bottle.app', read_only=True)
    def app(self):
        if False:
            for i in range(10):
                print('nop')
        ' Bottle application handling this request. '
        raise RuntimeError('This request is not connected to an application.')

    @DictProperty('environ', 'bottle.route', read_only=True)
    def route(self):
        if False:
            while True:
                i = 10
        ' The bottle :class:`Route` object that matches this request. '
        raise RuntimeError('This request is not connected to a route.')

    @DictProperty('environ', 'route.url_args', read_only=True)
    def url_args(self):
        if False:
            return 10
        ' The arguments extracted from the URL. '
        raise RuntimeError('This request is not connected to a route.')

    @property
    def path(self):
        if False:
            i = 10
            return i + 15
        ' The value of ``PATH_INFO`` with exactly one prefixed slash (to fix\n            broken clients and avoid the "empty path" edge case). '
        return '/' + self.environ.get('PATH_INFO', '').lstrip('/')

    @property
    def method(self):
        if False:
            for i in range(10):
                print('nop')
        ' The ``REQUEST_METHOD`` value as an uppercase string. '
        return self.environ.get('REQUEST_METHOD', 'GET').upper()

    @DictProperty('environ', 'bottle.request.headers', read_only=True)
    def headers(self):
        if False:
            i = 10
            return i + 15
        ' A :class:`WSGIHeaderDict` that provides case-insensitive access to\n            HTTP request headers. '
        return WSGIHeaderDict(self.environ)

    def get_header(self, name, default=None):
        if False:
            while True:
                i = 10
        ' Return the value of a request header, or a given default value. '
        return self.headers.get(name, default)

    @DictProperty('environ', 'bottle.request.cookies', read_only=True)
    def cookies(self):
        if False:
            print('Hello World!')
        ' Cookies parsed into a :class:`FormsDict`. Signed cookies are NOT\n            decoded. Use :meth:`get_cookie` if you expect signed cookies. '
        cookies = SimpleCookie(self.environ.get('HTTP_COOKIE', '')).values()
        return FormsDict(((c.key, c.value) for c in cookies))

    def get_cookie(self, key, default=None, secret=None, digestmod=hashlib.sha256):
        if False:
            print('Hello World!')
        ' Return the content of a cookie. To read a `Signed Cookie`, the\n            `secret` must match the one used to create the cookie (see\n            :meth:`BaseResponse.set_cookie`). If anything goes wrong (missing\n            cookie or wrong signature), return a default value. '
        value = self.cookies.get(key)
        if secret:
            if value and value.startswith('!') and ('?' in value):
                (sig, msg) = map(tob, value[1:].split('?', 1))
                hash = hmac.new(tob(secret), msg, digestmod=digestmod).digest()
                if _lscmp(sig, base64.b64encode(hash)):
                    dst = pickle.loads(base64.b64decode(msg))
                    if dst and dst[0] == key:
                        return dst[1]
            return default
        return value or default

    @DictProperty('environ', 'bottle.request.query', read_only=True)
    def query(self):
        if False:
            print('Hello World!')
        ' The :attr:`query_string` parsed into a :class:`FormsDict`. These\n            values are sometimes called "URL arguments" or "GET parameters", but\n            not to be confused with "URL wildcards" as they are provided by the\n            :class:`Router`. '
        get = self.environ['bottle.get'] = FormsDict()
        pairs = _parse_qsl(self.environ.get('QUERY_STRING', ''))
        for (key, value) in pairs:
            get[key] = value
        return get

    @DictProperty('environ', 'bottle.request.forms', read_only=True)
    def forms(self):
        if False:
            return 10
        ' Form values parsed from an `url-encoded` or `multipart/form-data`\n            encoded POST or PUT request body. The result is returned as a\n            :class:`FormsDict`. All keys and values are strings. File uploads\n            are stored separately in :attr:`files`. '
        forms = FormsDict()
        forms.recode_unicode = self.POST.recode_unicode
        for (name, item) in self.POST.allitems():
            if not isinstance(item, FileUpload):
                forms[name] = item
        return forms

    @DictProperty('environ', 'bottle.request.params', read_only=True)
    def params(self):
        if False:
            i = 10
            return i + 15
        ' A :class:`FormsDict` with the combined values of :attr:`query` and\n            :attr:`forms`. File uploads are stored in :attr:`files`. '
        params = FormsDict()
        for (key, value) in self.query.allitems():
            params[key] = value
        for (key, value) in self.forms.allitems():
            params[key] = value
        return params

    @DictProperty('environ', 'bottle.request.files', read_only=True)
    def files(self):
        if False:
            i = 10
            return i + 15
        ' File uploads parsed from `multipart/form-data` encoded POST or PUT\n            request body. The values are instances of :class:`FileUpload`.\n\n        '
        files = FormsDict()
        files.recode_unicode = self.POST.recode_unicode
        for (name, item) in self.POST.allitems():
            if isinstance(item, FileUpload):
                files[name] = item
        return files

    @DictProperty('environ', 'bottle.request.json', read_only=True)
    def json(self):
        if False:
            print('Hello World!')
        ' If the ``Content-Type`` header is ``application/json`` or\n            ``application/json-rpc``, this property holds the parsed content\n            of the request body. Only requests smaller than :attr:`MEMFILE_MAX`\n            are processed to avoid memory exhaustion.\n            Invalid JSON raises a 400 error response.\n        '
        ctype = self.environ.get('CONTENT_TYPE', '').lower().split(';')[0]
        if ctype in ('application/json', 'application/json-rpc'):
            b = self._get_body_string(self.MEMFILE_MAX)
            if not b:
                return None
            try:
                return json_loads(b)
            except (ValueError, TypeError):
                raise HTTPError(400, 'Invalid JSON')
        return None

    def _iter_body(self, read, bufsize):
        if False:
            return 10
        maxread = max(0, self.content_length)
        while maxread:
            part = read(min(maxread, bufsize))
            if not part:
                break
            yield part
            maxread -= len(part)

    @staticmethod
    def _iter_chunked(read, bufsize):
        if False:
            print('Hello World!')
        err = HTTPError(400, 'Error while parsing chunked transfer body.')
        (rn, sem, bs) = (tob('\r\n'), tob(';'), tob(''))
        while True:
            header = read(1)
            while header[-2:] != rn:
                c = read(1)
                header += c
                if not c:
                    raise err
                if len(header) > bufsize:
                    raise err
            (size, _, _) = header.partition(sem)
            try:
                maxread = int(tonat(size.strip()), 16)
            except ValueError:
                raise err
            if maxread == 0:
                break
            buff = bs
            while maxread > 0:
                if not buff:
                    buff = read(min(maxread, bufsize))
                (part, buff) = (buff[:maxread], buff[maxread:])
                if not part:
                    raise err
                yield part
                maxread -= len(part)
            if read(2) != rn:
                raise err

    @DictProperty('environ', 'bottle.request.body', read_only=True)
    def _body(self):
        if False:
            i = 10
            return i + 15
        try:
            read_func = self.environ['wsgi.input'].read
        except KeyError:
            self.environ['wsgi.input'] = BytesIO()
            return self.environ['wsgi.input']
        body_iter = self._iter_chunked if self.chunked else self._iter_body
        (body, body_size, is_temp_file) = (BytesIO(), 0, False)
        for part in body_iter(read_func, self.MEMFILE_MAX):
            body.write(part)
            body_size += len(part)
            if not is_temp_file and body_size > self.MEMFILE_MAX:
                (body, tmp) = (NamedTemporaryFile(mode='w+b'), body)
                body.write(tmp.getvalue())
                del tmp
                is_temp_file = True
        self.environ['wsgi.input'] = body
        body.seek(0)
        return body

    def _get_body_string(self, maxread):
        if False:
            i = 10
            return i + 15
        ' Read body into a string. Raise HTTPError(413) on requests that are\n            too large. '
        if self.content_length > maxread:
            raise HTTPError(413, 'Request entity too large')
        data = self.body.read(maxread + 1)
        if len(data) > maxread:
            raise HTTPError(413, 'Request entity too large')
        return data

    @property
    def body(self):
        if False:
            return 10
        ' The HTTP request body as a seek-able file-like object. Depending on\n            :attr:`MEMFILE_MAX`, this is either a temporary file or a\n            :class:`io.BytesIO` instance. Accessing this property for the first\n            time reads and replaces the ``wsgi.input`` environ variable.\n            Subsequent accesses just do a `seek(0)` on the file object. '
        self._body.seek(0)
        return self._body

    @property
    def chunked(self):
        if False:
            i = 10
            return i + 15
        ' True if Chunked transfer encoding was. '
        return 'chunked' in self.environ.get('HTTP_TRANSFER_ENCODING', '').lower()
    GET = query

    @DictProperty('environ', 'bottle.request.post', read_only=True)
    def POST(self):
        if False:
            for i in range(10):
                print('nop')
        ' The values of :attr:`forms` and :attr:`files` combined into a single\n            :class:`FormsDict`. Values are either strings (form values) or\n            instances of :class:`cgi.FieldStorage` (file uploads).\n        '
        post = FormsDict()
        if not self.content_type.startswith('multipart/'):
            body = tonat(self._get_body_string(self.MEMFILE_MAX), 'latin1')
            for (key, value) in _parse_qsl(body):
                post[key] = value
            return post
        safe_env = {'QUERY_STRING': ''}
        for key in ('REQUEST_METHOD', 'CONTENT_TYPE', 'CONTENT_LENGTH'):
            if key in self.environ:
                safe_env[key] = self.environ[key]
        args = dict(fp=self.body, environ=safe_env, keep_blank_values=True)
        if py3k:
            args['encoding'] = 'utf8'
            post.recode_unicode = False
        data = cgi.FieldStorage(**args)
        self['_cgi.FieldStorage'] = data
        data = data.list or []
        for item in data:
            if item.filename is None:
                post[item.name] = item.value
            else:
                post[item.name] = FileUpload(item.file, item.name, item.filename, item.headers)
        return post

    @property
    def url(self):
        if False:
            return 10
        ' The full request URI including hostname and scheme. If your app\n            lives behind a reverse proxy or load balancer and you get confusing\n            results, make sure that the ``X-Forwarded-Host`` header is set\n            correctly. '
        return self.urlparts.geturl()

    @DictProperty('environ', 'bottle.request.urlparts', read_only=True)
    def urlparts(self):
        if False:
            while True:
                i = 10
        ' The :attr:`url` string as an :class:`urlparse.SplitResult` tuple.\n            The tuple contains (scheme, host, path, query_string and fragment),\n            but the fragment is always empty because it is not visible to the\n            server. '
        env = self.environ
        http = env.get('HTTP_X_FORWARDED_PROTO') or env.get('wsgi.url_scheme', 'http')
        host = env.get('HTTP_X_FORWARDED_HOST') or env.get('HTTP_HOST')
        if not host:
            host = env.get('SERVER_NAME', '127.0.0.1')
            port = env.get('SERVER_PORT')
            if port and port != ('80' if http == 'http' else '443'):
                host += ':' + port
        path = urlquote(self.fullpath)
        return UrlSplitResult(http, host, path, env.get('QUERY_STRING'), '')

    @property
    def fullpath(self):
        if False:
            i = 10
            return i + 15
        ' Request path including :attr:`script_name` (if present). '
        return urljoin(self.script_name, self.path.lstrip('/'))

    @property
    def query_string(self):
        if False:
            while True:
                i = 10
        ' The raw :attr:`query` part of the URL (everything in between ``?``\n            and ``#``) as a string. '
        return self.environ.get('QUERY_STRING', '')

    @property
    def script_name(self):
        if False:
            return 10
        " The initial portion of the URL's `path` that was removed by a higher\n            level (server or routing middleware) before the application was\n            called. This script path is returned with leading and tailing\n            slashes. "
        script_name = self.environ.get('SCRIPT_NAME', '').strip('/')
        return '/' + script_name + '/' if script_name else '/'

    def path_shift(self, shift=1):
        if False:
            print('Hello World!')
        ' Shift path segments from :attr:`path` to :attr:`script_name` and\n            vice versa.\n\n           :param shift: The number of path segments to shift. May be negative\n                         to change the shift direction. (default: 1)\n        '
        (script, path) = path_shift(self.environ.get('SCRIPT_NAME', '/'), self.path, shift)
        (self['SCRIPT_NAME'], self['PATH_INFO']) = (script, path)

    @property
    def content_length(self):
        if False:
            i = 10
            return i + 15
        ' The request body length as an integer. The client is responsible to\n            set this header. Otherwise, the real length of the body is unknown\n            and -1 is returned. In this case, :attr:`body` will be empty. '
        return int(self.environ.get('CONTENT_LENGTH') or -1)

    @property
    def content_type(self):
        if False:
            return 10
        ' The Content-Type header as a lowercase-string (default: empty). '
        return self.environ.get('CONTENT_TYPE', '').lower()

    @property
    def is_xhr(self):
        if False:
            return 10
        ' True if the request was triggered by a XMLHttpRequest. This only\n            works with JavaScript libraries that support the `X-Requested-With`\n            header (most of the popular libraries do). '
        requested_with = self.environ.get('HTTP_X_REQUESTED_WITH', '')
        return requested_with.lower() == 'xmlhttprequest'

    @property
    def is_ajax(self):
        if False:
            print('Hello World!')
        ' Alias for :attr:`is_xhr`. "Ajax" is not the right term. '
        return self.is_xhr

    @property
    def auth(self):
        if False:
            for i in range(10):
                print('nop')
        ' HTTP authentication data as a (user, password) tuple. This\n            implementation currently supports basic (not digest) authentication\n            only. If the authentication happened at a higher level (e.g. in the\n            front web-server or a middleware), the password field is None, but\n            the user field is looked up from the ``REMOTE_USER`` environ\n            variable. On any errors, None is returned. '
        basic = parse_auth(self.environ.get('HTTP_AUTHORIZATION', ''))
        if basic:
            return basic
        ruser = self.environ.get('REMOTE_USER')
        if ruser:
            return (ruser, None)
        return None

    @property
    def remote_route(self):
        if False:
            print('Hello World!')
        ' A list of all IPs that were involved in this request, starting with\n            the client IP and followed by zero or more proxies. This does only\n            work if all proxies support the ```X-Forwarded-For`` header. Note\n            that this information can be forged by malicious clients. '
        proxy = self.environ.get('HTTP_X_FORWARDED_FOR')
        if proxy:
            return [ip.strip() for ip in proxy.split(',')]
        remote = self.environ.get('REMOTE_ADDR')
        return [remote] if remote else []

    @property
    def remote_addr(self):
        if False:
            for i in range(10):
                print('nop')
        ' The client IP as a string. Note that this information can be forged\n            by malicious clients. '
        route = self.remote_route
        return route[0] if route else None

    def copy(self):
        if False:
            print('Hello World!')
        ' Return a new :class:`Request` with a shallow :attr:`environ` copy. '
        return Request(self.environ.copy())

    def get(self, value, default=None):
        if False:
            for i in range(10):
                print('nop')
        return self.environ.get(value, default)

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self.environ[key]

    def __delitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        self[key] = ''
        del self.environ[key]

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.environ)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.environ)

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        return self.environ.keys()

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        ' Change an environ value and clear all caches that depend on it. '
        if self.environ.get('bottle.request.readonly'):
            raise KeyError('The environ dictionary is read-only.')
        self.environ[key] = value
        todelete = ()
        if key == 'wsgi.input':
            todelete = ('body', 'forms', 'files', 'params', 'post', 'json')
        elif key == 'QUERY_STRING':
            todelete = ('query', 'params')
        elif key.startswith('HTTP_'):
            todelete = ('headers', 'cookies')
        for key in todelete:
            self.environ.pop('bottle.request.' + key, None)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s: %s %s>' % (self.__class__.__name__, self.method, self.url)

    def __getattr__(self, name):
        if False:
            return 10
        ' Search in self.environ for additional user defined attributes. '
        try:
            var = self.environ['bottle.request.ext.%s' % name]
            return var.__get__(self) if hasattr(var, '__get__') else var
        except KeyError:
            raise AttributeError('Attribute %r not defined.' % name)

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        if name == 'environ':
            return object.__setattr__(self, name, value)
        key = 'bottle.request.ext.%s' % name
        if hasattr(self, name):
            raise AttributeError('Attribute already defined: %s' % name)
        self.environ[key] = value

    def __delattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        try:
            del self.environ['bottle.request.ext.%s' % name]
        except KeyError:
            raise AttributeError('Attribute not defined: %s' % name)

def _hkey(key):
    if False:
        while True:
            i = 10
    if '\n' in key or '\r' in key or '\x00' in key:
        raise ValueError('Header names must not contain control characters: %r' % key)
    return key.title().replace('_', '-')

def _hval(value):
    if False:
        i = 10
        return i + 15
    value = tonat(value)
    if '\n' in value or '\r' in value or '\x00' in value:
        raise ValueError('Header value must not contain control characters: %r' % value)
    return value

class HeaderProperty(object):

    def __init__(self, name, reader=None, writer=None, default=''):
        if False:
            i = 10
            return i + 15
        (self.name, self.default) = (name, default)
        (self.reader, self.writer) = (reader, writer)
        self.__doc__ = 'Current value of the %r header.' % name.title()

    def __get__(self, obj, _):
        if False:
            i = 10
            return i + 15
        if obj is None:
            return self
        value = obj.get_header(self.name, self.default)
        return self.reader(value) if self.reader else value

    def __set__(self, obj, value):
        if False:
            print('Hello World!')
        obj[self.name] = self.writer(value) if self.writer else value

    def __delete__(self, obj):
        if False:
            return 10
        del obj[self.name]

class BaseResponse(object):
    """ Storage class for a response body as well as headers and cookies.

        This class does support dict-like case-insensitive item-access to
        headers, but is NOT a dict. Most notably, iterating over a response
        yields parts of the body and not the headers.

        :param body: The response body as one of the supported types.
        :param status: Either an HTTP status code (e.g. 200) or a status line
                       including the reason phrase (e.g. '200 OK').
        :param headers: A dictionary or a list of name-value pairs.

        Additional keyword arguments are added to the list of headers.
        Underscores in the header name are replaced with dashes.
    """
    default_status = 200
    default_content_type = 'text/html; charset=UTF-8'
    bad_headers = {204: frozenset(('Content-Type', 'Content-Length')), 304: frozenset(('Allow', 'Content-Encoding', 'Content-Language', 'Content-Length', 'Content-Range', 'Content-Type', 'Content-Md5', 'Last-Modified'))}

    def __init__(self, body='', status=None, headers=None, **more_headers):
        if False:
            return 10
        self._cookies = None
        self._headers = {}
        self.body = body
        self.status = status or self.default_status
        if headers:
            if isinstance(headers, dict):
                headers = headers.items()
            for (name, value) in headers:
                self.add_header(name, value)
        if more_headers:
            for (name, value) in more_headers.items():
                self.add_header(name, value)

    def copy(self, cls=None):
        if False:
            print('Hello World!')
        ' Returns a copy of self. '
        cls = cls or BaseResponse
        assert issubclass(cls, BaseResponse)
        copy = cls()
        copy.status = self.status
        copy._headers = dict(((k, v[:]) for (k, v) in self._headers.items()))
        if self._cookies:
            cookies = copy._cookies = SimpleCookie()
            for (k, v) in self._cookies.items():
                cookies[k] = v.value
                cookies[k].update(v)
        return copy

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.body)

    def close(self):
        if False:
            while True:
                i = 10
        if hasattr(self.body, 'close'):
            self.body.close()

    @property
    def status_line(self):
        if False:
            i = 10
            return i + 15
        ' The HTTP status line as a string (e.g. ``404 Not Found``).'
        return self._status_line

    @property
    def status_code(self):
        if False:
            print('Hello World!')
        ' The HTTP status code as an integer (e.g. 404).'
        return self._status_code

    def _set_status(self, status):
        if False:
            while True:
                i = 10
        if isinstance(status, int):
            (code, status) = (status, _HTTP_STATUS_LINES.get(status))
        elif ' ' in status:
            if '\n' in status or '\r' in status or '\x00' in status:
                raise ValueError('Status line must not include control chars.')
            status = status.strip()
            code = int(status.split()[0])
        else:
            raise ValueError('String status line without a reason phrase.')
        if not 100 <= code <= 999:
            raise ValueError('Status code out of range.')
        self._status_code = code
        self._status_line = str(status or '%d Unknown' % code)

    def _get_status(self):
        if False:
            print('Hello World!')
        return self._status_line
    status = property(_get_status, _set_status, None, ' A writeable property to change the HTTP response status. It accepts\n            either a numeric code (100-999) or a string with a custom reason\n            phrase (e.g. "404 Brain not found"). Both :data:`status_line` and\n            :data:`status_code` are updated accordingly. The return value is\n            always a status string. ')
    del _get_status, _set_status

    @property
    def headers(self):
        if False:
            return 10
        ' An instance of :class:`HeaderDict`, a case-insensitive dict-like\n            view on the response headers. '
        hdict = HeaderDict()
        hdict.dict = self._headers
        return hdict

    def __contains__(self, name):
        if False:
            i = 10
            return i + 15
        return _hkey(name) in self._headers

    def __delitem__(self, name):
        if False:
            while True:
                i = 10
        del self._headers[_hkey(name)]

    def __getitem__(self, name):
        if False:
            print('Hello World!')
        return self._headers[_hkey(name)][-1]

    def __setitem__(self, name, value):
        if False:
            return 10
        self._headers[_hkey(name)] = [_hval(value)]

    def get_header(self, name, default=None):
        if False:
            print('Hello World!')
        ' Return the value of a previously defined header. If there is no\n            header with that name, return a default value. '
        return self._headers.get(_hkey(name), [default])[-1]

    def set_header(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        ' Create a new response header, replacing any previously defined\n            headers with the same name. '
        self._headers[_hkey(name)] = [_hval(value)]

    def add_header(self, name, value):
        if False:
            i = 10
            return i + 15
        ' Add an additional response header, not removing duplicates. '
        self._headers.setdefault(_hkey(name), []).append(_hval(value))

    def iter_headers(self):
        if False:
            return 10
        ' Yield (header, value) tuples, skipping headers that are not\n            allowed with the current response status code. '
        return self.headerlist

    def _wsgi_status_line(self):
        if False:
            while True:
                i = 10
        ' WSGI conform status line (latin1-encodeable) '
        if py3k:
            return self._status_line.encode('utf8').decode('latin1')
        return self._status_line

    @property
    def headerlist(self):
        if False:
            for i in range(10):
                print('nop')
        ' WSGI conform list of (header, value) tuples. '
        out = []
        headers = list(self._headers.items())
        if 'Content-Type' not in self._headers:
            headers.append(('Content-Type', [self.default_content_type]))
        if self._status_code in self.bad_headers:
            bad_headers = self.bad_headers[self._status_code]
            headers = [h for h in headers if h[0] not in bad_headers]
        out += [(name, val) for (name, vals) in headers for val in vals]
        if self._cookies:
            for c in self._cookies.values():
                out.append(('Set-Cookie', _hval(c.OutputString())))
        if py3k:
            out = [(k, v.encode('utf8').decode('latin1')) for (k, v) in out]
        return out
    content_type = HeaderProperty('Content-Type')
    content_length = HeaderProperty('Content-Length', reader=int, default=-1)
    expires = HeaderProperty('Expires', reader=lambda x: datetime.utcfromtimestamp(parse_date(x)), writer=lambda x: http_date(x))

    @property
    def charset(self, default='UTF-8'):
        if False:
            while True:
                i = 10
        ' Return the charset specified in the content-type header (default: utf8). '
        if 'charset=' in self.content_type:
            return self.content_type.split('charset=')[-1].split(';')[0].strip()
        return default

    def set_cookie(self, name, value, secret=None, digestmod=hashlib.sha256, **options):
        if False:
            while True:
                i = 10
        ' Create a new cookie or replace an old one. If the `secret` parameter is\n            set, create a `Signed Cookie` (described below).\n\n            :param name: the name of the cookie.\n            :param value: the value of the cookie.\n            :param secret: a signature key required for signed cookies.\n\n            Additionally, this method accepts all RFC 2109 attributes that are\n            supported by :class:`cookie.Morsel`, including:\n\n            :param maxage: maximum age in seconds. (default: None)\n            :param expires: a datetime object or UNIX timestamp. (default: None)\n            :param domain: the domain that is allowed to read the cookie.\n              (default: current domain)\n            :param path: limits the cookie to a given path (default: current path)\n            :param secure: limit the cookie to HTTPS connections (default: off).\n            :param httponly: prevents client-side javascript to read this cookie\n              (default: off, requires Python 2.6 or newer).\n            :param samesite: Control or disable third-party use for this cookie.\n              Possible values: `lax`, `strict` or `none` (default).\n\n            If neither `expires` nor `maxage` is set (default), the cookie will\n            expire at the end of the browser session (as soon as the browser\n            window is closed).\n\n            Signed cookies may store any pickle-able object and are\n            cryptographically signed to prevent manipulation. Keep in mind that\n            cookies are limited to 4kb in most browsers.\n\n            Warning: Pickle is a potentially dangerous format. If an attacker\n            gains access to the secret key, he could forge cookies that execute\n            code on server side if unpickled. Using pickle is discouraged and\n            support for it will be removed in later versions of bottle.\n\n            Warning: Signed cookies are not encrypted (the client can still see\n            the content) and not copy-protected (the client can restore an old\n            cookie). The main intention is to make pickling and unpickling\n            save, not to store secret information at client side.\n        '
        if not self._cookies:
            self._cookies = SimpleCookie()
        if py < (3, 8, 0):
            Morsel._reserved.setdefault('samesite', 'SameSite')
        if secret:
            if not isinstance(value, basestring):
                depr(0, 13, 'Pickling of arbitrary objects into cookies is deprecated.', 'Only store strings in cookies. JSON strings are fine, too.')
            encoded = base64.b64encode(pickle.dumps([name, value], -1))
            sig = base64.b64encode(hmac.new(tob(secret), encoded, digestmod=digestmod).digest())
            value = touni(tob('!') + sig + tob('?') + encoded)
        elif not isinstance(value, basestring):
            raise TypeError('Secret key required for non-string cookies.')
        if len(name) + len(value) > 3800:
            raise ValueError('Content does not fit into a cookie.')
        self._cookies[name] = value
        for (key, value) in options.items():
            if key in ('max_age', 'maxage'):
                key = 'max-age'
                if isinstance(value, timedelta):
                    value = value.seconds + value.days * 24 * 3600
            if key == 'expires':
                value = http_date(value)
            if key in ('same_site', 'samesite'):
                (key, value) = ('samesite', (value or 'none').lower())
                if value not in ('lax', 'strict', 'none'):
                    raise CookieError('Invalid value for SameSite')
            if key in ('secure', 'httponly') and (not value):
                continue
            self._cookies[name][key] = value

    def delete_cookie(self, key, **kwargs):
        if False:
            print('Hello World!')
        ' Delete a cookie. Be sure to use the same `domain` and `path`\n            settings as used to create the cookie. '
        kwargs['max_age'] = -1
        kwargs['expires'] = 0
        self.set_cookie(key, '', **kwargs)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        out = ''
        for (name, value) in self.headerlist:
            out += '%s: %s\n' % (name.title(), value.strip())
        return out

def _local_property():
    if False:
        for i in range(10):
            print('nop')
    ls = threading.local()

    def fget(_):
        if False:
            print('Hello World!')
        try:
            return ls.var
        except AttributeError:
            raise RuntimeError('Request context not initialized.')

    def fset(_, value):
        if False:
            i = 10
            return i + 15
        ls.var = value

    def fdel(_):
        if False:
            print('Hello World!')
        del ls.var
    return property(fget, fset, fdel, 'Thread-local property')

class LocalRequest(BaseRequest):
    """ A thread-local subclass of :class:`BaseRequest` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`request`). If accessed during a
        request/response cycle, this instance always refers to the *current*
        request (even on a multithreaded server). """
    bind = BaseRequest.__init__
    environ = _local_property()

class LocalResponse(BaseResponse):
    """ A thread-local subclass of :class:`BaseResponse` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`response`). Its attributes are used
        to build the HTTP response at the end of the request/response cycle.
    """
    bind = BaseResponse.__init__
    _status_line = _local_property()
    _status_code = _local_property()
    _cookies = _local_property()
    _headers = _local_property()
    body = _local_property()
Request = BaseRequest
Response = BaseResponse

class HTTPResponse(Response, BottleException):

    def __init__(self, body='', status=None, headers=None, **more_headers):
        if False:
            for i in range(10):
                print('nop')
        super(HTTPResponse, self).__init__(body, status, headers, **more_headers)

    def apply(self, other):
        if False:
            while True:
                i = 10
        other._status_code = self._status_code
        other._status_line = self._status_line
        other._headers = self._headers
        other._cookies = self._cookies
        other.body = self.body

class HTTPError(HTTPResponse):
    default_status = 500

    def __init__(self, status=None, body=None, exception=None, traceback=None, **more_headers):
        if False:
            return 10
        self.exception = exception
        self.traceback = traceback
        super(HTTPError, self).__init__(body, status, **more_headers)

class PluginError(BottleException):
    pass

class JSONPlugin(object):
    name = 'json'
    api = 2

    def __init__(self, json_dumps=json_dumps):
        if False:
            while True:
                i = 10
        self.json_dumps = json_dumps

    def setup(self, app):
        if False:
            i = 10
            return i + 15
        app.config._define('json.enable', default=True, validate=bool, help='Enable or disable automatic dict->json filter.')
        app.config._define('json.ascii', default=False, validate=bool, help='Use only 7-bit ASCII characters in output.')
        app.config._define('json.indent', default=True, validate=bool, help='Add whitespace to make json more readable.')
        app.config._define('json.dump_func', default=None, help='If defined, use this function to transform dict into json. The other options no longer apply.')

    def apply(self, callback, route):
        if False:
            print('Hello World!')
        dumps = self.json_dumps
        if not self.json_dumps:
            return callback

        @functools.wraps(callback)
        def wrapper(*a, **ka):
            if False:
                print('Hello World!')
            try:
                rv = callback(*a, **ka)
            except HTTPResponse as resp:
                rv = resp
            if isinstance(rv, dict):
                json_response = dumps(rv)
                response.content_type = 'application/json'
                return json_response
            elif isinstance(rv, HTTPResponse) and isinstance(rv.body, dict):
                rv.body = dumps(rv.body)
                rv.content_type = 'application/json'
            return rv
        return wrapper

class TemplatePlugin(object):
    """ This plugin applies the :func:`view` decorator to all routes with a
        `template` config parameter. If the parameter is a tuple, the second
        element must be a dict with additional options (e.g. `template_engine`)
        or default variables for the template. """
    name = 'template'
    api = 2

    def setup(self, app):
        if False:
            print('Hello World!')
        app.tpl = self

    def apply(self, callback, route):
        if False:
            for i in range(10):
                print('nop')
        conf = route.config.get('template')
        if isinstance(conf, (tuple, list)) and len(conf) == 2:
            return view(conf[0], **conf[1])(callback)
        elif isinstance(conf, str):
            return view(conf)(callback)
        else:
            return callback

class _ImportRedirect(object):

    def __init__(self, name, impmask):
        if False:
            print('Hello World!')
        ' Create a virtual package that redirects imports (see PEP 302). '
        self.name = name
        self.impmask = impmask
        self.module = sys.modules.setdefault(name, new_module(name))
        self.module.__dict__.update({'__file__': __file__, '__path__': [], '__all__': [], '__loader__': self})
        sys.meta_path.append(self)

    def find_spec(self, fullname, path, target=None):
        if False:
            while True:
                i = 10
        if '.' not in fullname:
            return
        if fullname.rsplit('.', 1)[0] != self.name:
            return
        from importlib.util import spec_from_loader
        return spec_from_loader(fullname, self)

    def find_module(self, fullname, path=None):
        if False:
            for i in range(10):
                print('nop')
        if '.' not in fullname:
            return
        if fullname.rsplit('.', 1)[0] != self.name:
            return
        return self

    def load_module(self, fullname):
        if False:
            for i in range(10):
                print('nop')
        if fullname in sys.modules:
            return sys.modules[fullname]
        modname = fullname.rsplit('.', 1)[1]
        realname = self.impmask % modname
        __import__(realname)
        module = sys.modules[fullname] = sys.modules[realname]
        setattr(self.module, modname, module)
        module.__loader__ = self
        return module

class MultiDict(DictMixin):
    """ This dict stores multiple values per key, but behaves exactly like a
        normal dict in that it returns only the newest value for any given key.
        There are special methods available to access the full list of values.
    """

    def __init__(self, *a, **k):
        if False:
            print('Hello World!')
        self.dict = dict(((k, [v]) for (k, v) in dict(*a, **k).items()))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.dict)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.dict)

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        return key in self.dict

    def __delitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        del self.dict[key]

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.dict[key][-1]

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.append(key, value)

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        return self.dict.keys()
    if py3k:

        def values(self):
            if False:
                while True:
                    i = 10
            return (v[-1] for v in self.dict.values())

        def items(self):
            if False:
                print('Hello World!')
            return ((k, v[-1]) for (k, v) in self.dict.items())

        def allitems(self):
            if False:
                while True:
                    i = 10
            return ((k, v) for (k, vl) in self.dict.items() for v in vl)
        iterkeys = keys
        itervalues = values
        iteritems = items
        iterallitems = allitems
    else:

        def values(self):
            if False:
                print('Hello World!')
            return [v[-1] for v in self.dict.values()]

        def items(self):
            if False:
                while True:
                    i = 10
            return [(k, v[-1]) for (k, v) in self.dict.items()]

        def iterkeys(self):
            if False:
                while True:
                    i = 10
            return self.dict.iterkeys()

        def itervalues(self):
            if False:
                for i in range(10):
                    print('nop')
            return (v[-1] for v in self.dict.itervalues())

        def iteritems(self):
            if False:
                while True:
                    i = 10
            return ((k, v[-1]) for (k, v) in self.dict.iteritems())

        def iterallitems(self):
            if False:
                i = 10
                return i + 15
            return ((k, v) for (k, vl) in self.dict.iteritems() for v in vl)

        def allitems(self):
            if False:
                print('Hello World!')
            return [(k, v) for (k, vl) in self.dict.iteritems() for v in vl]

    def get(self, key, default=None, index=-1, type=None):
        if False:
            return 10
        ' Return the most recent value for a key.\n\n            :param default: The default value to be returned if the key is not\n                   present or the type conversion fails.\n            :param index: An index for the list of available values.\n            :param type: If defined, this callable is used to cast the value\n                    into a specific type. Exception are suppressed and result in\n                    the default value to be returned.\n        '
        try:
            val = self.dict[key][index]
            return type(val) if type else val
        except Exception:
            pass
        return default

    def append(self, key, value):
        if False:
            i = 10
            return i + 15
        ' Add a new value to the list of values for this key. '
        self.dict.setdefault(key, []).append(value)

    def replace(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        ' Replace the list of values with a single value. '
        self.dict[key] = [value]

    def getall(self, key):
        if False:
            for i in range(10):
                print('nop')
        ' Return a (possibly empty) list of values for a key. '
        return self.dict.get(key) or []
    getone = get
    getlist = getall

class FormsDict(MultiDict):
    """ This :class:`MultiDict` subclass is used to store request form data.
        Additionally to the normal dict-like item access methods (which return
        unmodified data as native strings), this container also supports
        attribute-like access to its values. Attributes are automatically de-
        or recoded to match :attr:`input_encoding` (default: 'utf8'). Missing
        attributes default to an empty string. """
    input_encoding = 'utf8'
    recode_unicode = True

    def _fix(self, s, encoding=None):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(s, unicode) and self.recode_unicode:
            return s.encode('latin1').decode(encoding or self.input_encoding)
        elif isinstance(s, bytes):
            return s.decode(encoding or self.input_encoding)
        else:
            return s

    def decode(self, encoding=None):
        if False:
            for i in range(10):
                print('nop')
        ' Returns a copy with all keys and values de- or recoded to match\n            :attr:`input_encoding`. Some libraries (e.g. WTForms) want a\n            unicode dictionary. '
        copy = FormsDict()
        enc = copy.input_encoding = encoding or self.input_encoding
        copy.recode_unicode = False
        for (key, value) in self.allitems():
            copy.append(self._fix(key, enc), self._fix(value, enc))
        return copy

    def getunicode(self, name, default=None, encoding=None):
        if False:
            print('Hello World!')
        ' Return the value as a unicode string, or the default. '
        try:
            return self._fix(self[name], encoding)
        except (UnicodeError, KeyError):
            return default

    def __getattr__(self, name, default=unicode()):
        if False:
            return 10
        if name.startswith('__') and name.endswith('__'):
            return super(FormsDict, self).__getattr__(name)
        return self.getunicode(name, default=default)

class HeaderDict(MultiDict):
    """ A case-insensitive version of :class:`MultiDict` that defaults to
        replace the old value instead of appending it. """

    def __init__(self, *a, **ka):
        if False:
            i = 10
            return i + 15
        self.dict = {}
        if a or ka:
            self.update(*a, **ka)

    def __contains__(self, key):
        if False:
            return 10
        return _hkey(key) in self.dict

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        del self.dict[_hkey(key)]

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.dict[_hkey(key)][-1]

    def __setitem__(self, key, value):
        if False:
            return 10
        self.dict[_hkey(key)] = [_hval(value)]

    def append(self, key, value):
        if False:
            while True:
                i = 10
        self.dict.setdefault(_hkey(key), []).append(_hval(value))

    def replace(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.dict[_hkey(key)] = [_hval(value)]

    def getall(self, key):
        if False:
            return 10
        return self.dict.get(_hkey(key)) or []

    def get(self, key, default=None, index=-1):
        if False:
            while True:
                i = 10
        return MultiDict.get(self, _hkey(key), default, index)

    def filter(self, names):
        if False:
            return 10
        for name in (_hkey(n) for n in names):
            if name in self.dict:
                del self.dict[name]

class WSGIHeaderDict(DictMixin):
    """ This dict-like class wraps a WSGI environ dict and provides convenient
        access to HTTP_* fields. Keys and values are native strings
        (2.x bytes or 3.x unicode) and keys are case-insensitive. If the WSGI
        environment contains non-native string values, these are de- or encoded
        using a lossless 'latin1' character set.

        The API will remain stable even on changes to the relevant PEPs.
        Currently PEP 333, 444 and 3333 are supported. (PEP 444 is the only one
        that uses non-native strings.)
    """
    cgikeys = ('CONTENT_TYPE', 'CONTENT_LENGTH')

    def __init__(self, environ):
        if False:
            print('Hello World!')
        self.environ = environ

    def _ekey(self, key):
        if False:
            print('Hello World!')
        ' Translate header field name to CGI/WSGI environ key. '
        key = key.replace('-', '_').upper()
        if key in self.cgikeys:
            return key
        return 'HTTP_' + key

    def raw(self, key, default=None):
        if False:
            while True:
                i = 10
        ' Return the header value as is (may be bytes or unicode). '
        return self.environ.get(self._ekey(key), default)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        val = self.environ[self._ekey(key)]
        if py3k:
            if isinstance(val, unicode):
                val = val.encode('latin1').decode('utf8')
            else:
                val = val.decode('utf8')
        return val

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        raise TypeError('%s is read-only.' % self.__class__)

    def __delitem__(self, key):
        if False:
            return 10
        raise TypeError('%s is read-only.' % self.__class__)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for key in self.environ:
            if key[:5] == 'HTTP_':
                yield _hkey(key[5:])
            elif key in self.cgikeys:
                yield _hkey(key)

    def keys(self):
        if False:
            i = 10
            return i + 15
        return [x for x in self]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.keys())

    def __contains__(self, key):
        if False:
            return 10
        return self._ekey(key) in self.environ
_UNSET = object()

class ConfigDict(dict):
    """ A dict-like configuration storage with additional support for
        namespaces, validators, meta-data, overlays and more.

        This dict-like class is heavily optimized for read access. All read-only
        methods as well as item access should be as fast as the built-in dict.
    """
    __slots__ = ('_meta', '_change_listener', '_overlays', '_virtual_keys', '_source', '__weakref__')

    def __init__(self):
        if False:
            while True:
                i = 10
        self._meta = {}
        self._change_listener = []
        self._overlays = []
        self._source = None
        self._virtual_keys = set()

    def load_module(self, path, squash=True):
        if False:
            while True:
                i = 10
        'Load values from a Python module.\n\n           Example modue ``config.py``::\n\n                DEBUG = True\n                SQLITE = {\n                    "db": ":memory:"\n                }\n\n\n           >>> c = ConfigDict()\n           >>> c.load_module(\'config\')\n           {DEBUG: True, \'SQLITE.DB\': \'memory\'}\n           >>> c.load_module("config", False)\n           {\'DEBUG\': True, \'SQLITE\': {\'DB\': \'memory\'}}\n\n           :param squash: If true (default), dictionary values are assumed to\n                          represent namespaces (see :meth:`load_dict`).\n        '
        config_obj = load(path)
        obj = {key: getattr(config_obj, key) for key in dir(config_obj) if key.isupper()}
        if squash:
            self.load_dict(obj)
        else:
            self.update(obj)
        return self

    def load_config(self, filename, **options):
        if False:
            return 10
        ' Load values from an ``*.ini`` style config file.\n\n            A configuration file consists of sections, each led by a\n            ``[section]`` header, followed by key/value entries separated by\n            either ``=`` or ``:``. Section names and keys are case-insensitive.\n            Leading and trailing whitespace is removed from keys and values.\n            Values can be omitted, in which case the key/value delimiter may\n            also be left out. Values can also span multiple lines, as long as\n            they are indented deeper than the first line of the value. Commands\n            are prefixed by ``#`` or ``;`` and may only appear on their own on\n            an otherwise empty line.\n\n            Both section and key names may contain dots (``.``) as namespace\n            separators. The actual configuration parameter name is constructed\n            by joining section name and key name together and converting to\n            lower case.\n\n            The special sections ``bottle`` and ``ROOT`` refer to the root\n            namespace and the ``DEFAULT`` section defines default values for all\n            other sections.\n\n            With Python 3, extended string interpolation is enabled.\n\n            :param filename: The path of a config file, or a list of paths.\n            :param options: All keyword parameters are passed to the underlying\n                :class:`python:configparser.ConfigParser` constructor call.\n\n        '
        options.setdefault('allow_no_value', True)
        if py3k:
            options.setdefault('interpolation', configparser.ExtendedInterpolation())
        conf = configparser.ConfigParser(**options)
        conf.read(filename)
        for section in conf.sections():
            for key in conf.options(section):
                value = conf.get(section, key)
                if section not in ('bottle', 'ROOT'):
                    key = section + '.' + key
                self[key.lower()] = value
        return self

    def load_dict(self, source, namespace=''):
        if False:
            print('Hello World!')
        " Load values from a dictionary structure. Nesting can be used to\n            represent namespaces.\n\n            >>> c = ConfigDict()\n            >>> c.load_dict({'some': {'namespace': {'key': 'value'} } })\n            {'some.namespace.key': 'value'}\n        "
        for (key, value) in source.items():
            if isinstance(key, basestring):
                nskey = (namespace + '.' + key).strip('.')
                if isinstance(value, dict):
                    self.load_dict(value, namespace=nskey)
                else:
                    self[nskey] = value
            else:
                raise TypeError('Key has type %r (not a string)' % type(key))
        return self

    def update(self, *a, **ka):
        if False:
            return 10
        " If the first parameter is a string, all keys are prefixed with this\n            namespace. Apart from that it works just as the usual dict.update().\n\n            >>> c = ConfigDict()\n            >>> c.update('some.namespace', key='value')\n        "
        prefix = ''
        if a and isinstance(a[0], basestring):
            prefix = a[0].strip('.') + '.'
            a = a[1:]
        for (key, value) in dict(*a, **ka).items():
            self[prefix + key] = value

    def setdefault(self, key, value):
        if False:
            while True:
                i = 10
        if key not in self:
            self[key] = value
        return self[key]

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(key, basestring):
            raise TypeError('Key has type %r (not a string)' % type(key))
        self._virtual_keys.discard(key)
        value = self.meta_get(key, 'filter', lambda x: x)(value)
        if key in self and self[key] is value:
            return
        self._on_change(key, value)
        dict.__setitem__(self, key, value)
        for overlay in self._iter_overlays():
            overlay._set_virtual(key, value)

    def __delitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key not in self:
            raise KeyError(key)
        if key in self._virtual_keys:
            raise KeyError('Virtual keys cannot be deleted: %s' % key)
        if self._source and key in self._source:
            dict.__delitem__(self, key)
            self._set_virtual(key, self._source[key])
        else:
            self._on_change(key, None)
            dict.__delitem__(self, key)
            for overlay in self._iter_overlays():
                overlay._delete_virtual(key)

    def _set_virtual(self, key, value):
        if False:
            i = 10
            return i + 15
        ' Recursively set or update virtual keys. Do nothing if non-virtual\n            value is present. '
        if key in self and key not in self._virtual_keys:
            return
        self._virtual_keys.add(key)
        if key in self and self[key] is not value:
            self._on_change(key, value)
        dict.__setitem__(self, key, value)
        for overlay in self._iter_overlays():
            overlay._set_virtual(key, value)

    def _delete_virtual(self, key):
        if False:
            print('Hello World!')
        ' Recursively delete virtual entry. Do nothing if key is not virtual.\n        '
        if key not in self._virtual_keys:
            return
        if key in self:
            self._on_change(key, None)
        dict.__delitem__(self, key)
        self._virtual_keys.discard(key)
        for overlay in self._iter_overlays():
            overlay._delete_virtual(key)

    def _on_change(self, key, value):
        if False:
            while True:
                i = 10
        for cb in self._change_listener:
            if cb(self, key, value):
                return True

    def _add_change_listener(self, func):
        if False:
            return 10
        self._change_listener.append(func)
        return func

    def meta_get(self, key, metafield, default=None):
        if False:
            while True:
                i = 10
        ' Return the value of a meta field for a key. '
        return self._meta.get(key, {}).get(metafield, default)

    def meta_set(self, key, metafield, value):
        if False:
            i = 10
            return i + 15
        ' Set the meta field for a key to a new value. '
        self._meta.setdefault(key, {})[metafield] = value

    def meta_list(self, key):
        if False:
            i = 10
            return i + 15
        ' Return an iterable of meta field names defined for a key. '
        return self._meta.get(key, {}).keys()

    def _define(self, key, default=_UNSET, help=_UNSET, validate=_UNSET):
        if False:
            while True:
                i = 10
        ' (Unstable) Shortcut for plugins to define own config parameters. '
        if default is not _UNSET:
            self.setdefault(key, default)
        if help is not _UNSET:
            self.meta_set(key, 'help', help)
        if validate is not _UNSET:
            self.meta_set(key, 'validate', validate)

    def _iter_overlays(self):
        if False:
            return 10
        for ref in self._overlays:
            overlay = ref()
            if overlay is not None:
                yield overlay

    def _make_overlay(self):
        if False:
            print('Hello World!')
        " (Unstable) Create a new overlay that acts like a chained map: Values\n            missing in the overlay are copied from the source map. Both maps\n            share the same meta entries.\n\n            Entries that were copied from the source are called 'virtual'. You\n            can not delete virtual keys, but overwrite them, which turns them\n            into non-virtual entries. Setting keys on an overlay never affects\n            its source, but may affect any number of child overlays.\n\n            Other than collections.ChainMap or most other implementations, this\n            approach does not resolve missing keys on demand, but instead\n            actively copies all values from the source to the overlay and keeps\n            track of virtual and non-virtual keys internally. This removes any\n            lookup-overhead. Read-access is as fast as a build-in dict for both\n            virtual and non-virtual keys.\n\n            Changes are propagated recursively and depth-first. A failing\n            on-change handler in an overlay stops the propagation of virtual\n            values and may result in an partly updated tree. Take extra care\n            here and make sure that on-change handlers never fail.\n\n            Used by Route.config\n        "
        self._overlays[:] = [ref for ref in self._overlays if ref() is not None]
        overlay = ConfigDict()
        overlay._meta = self._meta
        overlay._source = self
        self._overlays.append(weakref.ref(overlay))
        for key in self:
            overlay._set_virtual(key, self[key])
        return overlay

class AppStack(list):
    """ A stack-like list. Calling it returns the head of the stack. """

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return the current default application. '
        return self.default

    def push(self, value=None):
        if False:
            print('Hello World!')
        ' Add a new :class:`Bottle` instance to the stack '
        if not isinstance(value, Bottle):
            value = Bottle()
        self.append(value)
        return value
    new_app = push

    @property
    def default(self):
        if False:
            print('Hello World!')
        try:
            return self[-1]
        except IndexError:
            return self.push()

class WSGIFileWrapper(object):

    def __init__(self, fp, buffer_size=1024 * 64):
        if False:
            i = 10
            return i + 15
        (self.fp, self.buffer_size) = (fp, buffer_size)
        for attr in ('fileno', 'close', 'read', 'readlines', 'tell', 'seek'):
            if hasattr(fp, attr):
                setattr(self, attr, getattr(fp, attr))

    def __iter__(self):
        if False:
            print('Hello World!')
        (buff, read) = (self.buffer_size, self.read)
        part = read(buff)
        while part:
            yield part
            part = read(buff)

class _closeiter(object):
    """ This only exists to be able to attach a .close method to iterators that
        do not support attribute assignment (most of itertools). """

    def __init__(self, iterator, close=None):
        if False:
            while True:
                i = 10
        self.iterator = iterator
        self.close_callbacks = makelist(close)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.iterator)

    def close(self):
        if False:
            while True:
                i = 10
        for func in self.close_callbacks:
            func()

class ResourceManager(object):
    """ This class manages a list of search paths and helps to find and open
        application-bound resources (files).

        :param base: default value for :meth:`add_path` calls.
        :param opener: callable used to open resources.
        :param cachemode: controls which lookups are cached. One of 'all',
                         'found' or 'none'.
    """

    def __init__(self, base='./', opener=open, cachemode='all'):
        if False:
            i = 10
            return i + 15
        self.opener = opener
        self.base = base
        self.cachemode = cachemode
        self.path = []
        self.cache = {}

    def add_path(self, path, base=None, index=None, create=False):
        if False:
            print('Hello World!')
        " Add a new path to the list of search paths. Return False if the\n            path does not exist.\n\n            :param path: The new search path. Relative paths are turned into\n                an absolute and normalized form. If the path looks like a file\n                (not ending in `/`), the filename is stripped off.\n            :param base: Path used to absolutize relative search paths.\n                Defaults to :attr:`base` which defaults to ``os.getcwd()``.\n            :param index: Position within the list of search paths. Defaults\n                to last index (appends to the list).\n\n            The `base` parameter makes it easy to reference files installed\n            along with a python module or package::\n\n                res.add_path('./resources/', __file__)\n        "
        base = os.path.abspath(os.path.dirname(base or self.base))
        path = os.path.abspath(os.path.join(base, os.path.dirname(path)))
        path += os.sep
        if path in self.path:
            self.path.remove(path)
        if create and (not os.path.isdir(path)):
            os.makedirs(path)
        if index is None:
            self.path.append(path)
        else:
            self.path.insert(index, path)
        self.cache.clear()
        return os.path.exists(path)

    def __iter__(self):
        if False:
            print('Hello World!')
        ' Iterate over all existing files in all registered paths. '
        search = self.path[:]
        while search:
            path = search.pop()
            if not os.path.isdir(path):
                continue
            for name in os.listdir(path):
                full = os.path.join(path, name)
                if os.path.isdir(full):
                    search.append(full)
                else:
                    yield full

    def lookup(self, name):
        if False:
            print('Hello World!')
        ' Search for a resource and return an absolute file path, or `None`.\n\n            The :attr:`path` list is searched in order. The first match is\n            returned. Symlinks are followed. The result is cached to speed up\n            future lookups. '
        if name not in self.cache or DEBUG:
            for path in self.path:
                fpath = os.path.join(path, name)
                if os.path.isfile(fpath):
                    if self.cachemode in ('all', 'found'):
                        self.cache[name] = fpath
                    return fpath
            if self.cachemode == 'all':
                self.cache[name] = None
        return self.cache[name]

    def open(self, name, mode='r', *args, **kwargs):
        if False:
            print('Hello World!')
        ' Find a resource and return a file object, or raise IOError. '
        fname = self.lookup(name)
        if not fname:
            raise IOError('Resource %r not found.' % name)
        return self.opener(fname, *args, mode=mode, **kwargs)

class FileUpload(object):

    def __init__(self, fileobj, name, filename, headers=None):
        if False:
            return 10
        ' Wrapper for file uploads. '
        self.file = fileobj
        self.name = name
        self.raw_filename = filename
        self.headers = HeaderDict(headers) if headers else HeaderDict()
    content_type = HeaderProperty('Content-Type')
    content_length = HeaderProperty('Content-Length', reader=int, default=-1)

    def get_header(self, name, default=None):
        if False:
            return 10
        ' Return the value of a header within the multipart part. '
        return self.headers.get(name, default)

    @cached_property
    def filename(self):
        if False:
            while True:
                i = 10
        " Name of the file on the client file system, but normalized to ensure\n            file system compatibility. An empty filename is returned as 'empty'.\n\n            Only ASCII letters, digits, dashes, underscores and dots are\n            allowed in the final filename. Accents are removed, if possible.\n            Whitespace is replaced by a single dash. Leading or tailing dots\n            or dashes are removed. The filename is limited to 255 characters.\n        "
        fname = self.raw_filename
        if not isinstance(fname, unicode):
            fname = fname.decode('utf8', 'ignore')
        fname = normalize('NFKD', fname)
        fname = fname.encode('ASCII', 'ignore').decode('ASCII')
        fname = os.path.basename(fname.replace('\\', os.path.sep))
        fname = re.sub('[^a-zA-Z0-9-_.\\s]', '', fname).strip()
        fname = re.sub('[-\\s]+', '-', fname).strip('.-')
        return fname[:255] or 'empty'

    def _copy_file(self, fp, chunk_size=2 ** 16):
        if False:
            print('Hello World!')
        (read, write, offset) = (self.file.read, fp.write, self.file.tell())
        while 1:
            buf = read(chunk_size)
            if not buf:
                break
            write(buf)
        self.file.seek(offset)

    def save(self, destination, overwrite=False, chunk_size=2 ** 16):
        if False:
            print('Hello World!')
        ' Save file to disk or copy its content to an open file(-like) object.\n            If *destination* is a directory, :attr:`filename` is added to the\n            path. Existing files are not overwritten by default (IOError).\n\n            :param destination: File path, directory or file(-like) object.\n            :param overwrite: If True, replace existing files. (default: False)\n            :param chunk_size: Bytes to read at a time. (default: 64kb)\n        '
        if isinstance(destination, basestring):
            if os.path.isdir(destination):
                destination = os.path.join(destination, self.filename)
            if not overwrite and os.path.exists(destination):
                raise IOError('File exists.')
            with open(destination, 'wb') as fp:
                self._copy_file(fp, chunk_size)
        else:
            self._copy_file(destination, chunk_size)

def abort(code=500, text='Unknown Error.'):
    if False:
        for i in range(10):
            print('nop')
    ' Aborts execution and causes a HTTP error. '
    raise HTTPError(code, text)

def redirect(url, code=None):
    if False:
        while True:
            i = 10
    ' Aborts execution and causes a 303 or 302 redirect, depending on\n        the HTTP protocol version. '
    if not code:
        code = 303 if request.get('SERVER_PROTOCOL') == 'HTTP/1.1' else 302
    res = response.copy(cls=HTTPResponse)
    res.status = code
    res.body = ''
    res.set_header('Location', urljoin(request.url, url))
    raise res

def _rangeiter(fp, offset, limit, bufsize=1024 * 1024):
    if False:
        return 10
    ' Yield chunks from a range in a file. '
    fp.seek(offset)
    while limit > 0:
        part = fp.read(min(limit, bufsize))
        if not part:
            break
        limit -= len(part)
        yield part

def static_file(filename, root, mimetype=True, download=False, charset='UTF-8', etag=None, headers=None):
    if False:
        for i in range(10):
            print('nop')
    ' Open a file in a safe way and return an instance of :exc:`HTTPResponse`\n        that can be sent back to the client.\n\n        :param filename: Name or path of the file to send, relative to ``root``.\n        :param root: Root path for file lookups. Should be an absolute directory\n            path.\n        :param mimetype: Provide the content-type header (default: guess from\n            file extension)\n        :param download: If True, ask the browser to open a `Save as...` dialog\n            instead of opening the file with the associated program. You can\n            specify a custom filename as a string. If not specified, the\n            original filename is used (default: False).\n        :param charset: The charset for files with a ``text/*`` mime-type.\n            (default: UTF-8)\n        :param etag: Provide a pre-computed ETag header. If set to ``False``,\n            ETag handling is disabled. (default: auto-generate ETag header)\n        :param headers: Additional headers dict to add to the response.\n\n        While checking user input is always a good idea, this function provides\n        additional protection against malicious ``filename`` parameters from\n        breaking out of the ``root`` directory and leaking sensitive information\n        to an attacker.\n\n        Read-protected files or files outside of the ``root`` directory are\n        answered with ``403 Access Denied``. Missing files result in a\n        ``404 Not Found`` response. Conditional requests (``If-Modified-Since``,\n        ``If-None-Match``) are answered with ``304 Not Modified`` whenever\n        possible. ``HEAD`` and ``Range`` requests (used by download managers to\n        check or continue partial downloads) are also handled automatically.\n\n    '
    root = os.path.join(os.path.abspath(root), '')
    filename = os.path.abspath(os.path.join(root, filename.strip('/\\')))
    headers = headers.copy() if headers else {}
    if not filename.startswith(root):
        return HTTPError(403, 'Access denied.')
    if not os.path.exists(filename) or not os.path.isfile(filename):
        return HTTPError(404, 'File does not exist.')
    if not os.access(filename, os.R_OK):
        return HTTPError(403, 'You do not have permission to access this file.')
    if mimetype is True:
        if download and download is not True:
            (mimetype, encoding) = mimetypes.guess_type(download)
        else:
            (mimetype, encoding) = mimetypes.guess_type(filename)
        if encoding:
            headers['Content-Encoding'] = encoding
    if mimetype:
        if (mimetype[:5] == 'text/' or mimetype == 'application/javascript') and charset and ('charset' not in mimetype):
            mimetype += '; charset=%s' % charset
        headers['Content-Type'] = mimetype
    if download:
        download = os.path.basename(filename if download is True else download)
        headers['Content-Disposition'] = 'attachment; filename="%s"' % download
    stats = os.stat(filename)
    headers['Content-Length'] = clen = stats.st_size
    headers['Last-Modified'] = email.utils.formatdate(stats.st_mtime, usegmt=True)
    headers['Date'] = email.utils.formatdate(time.time(), usegmt=True)
    getenv = request.environ.get
    if etag is None:
        etag = '%d:%d:%d:%d:%s' % (stats.st_dev, stats.st_ino, stats.st_mtime, clen, filename)
        etag = hashlib.sha1(tob(etag)).hexdigest()
    if etag:
        headers['ETag'] = etag
        check = getenv('HTTP_IF_NONE_MATCH')
        if check and check == etag:
            return HTTPResponse(status=304, **headers)
    ims = getenv('HTTP_IF_MODIFIED_SINCE')
    if ims:
        ims = parse_date(ims.split(';')[0].strip())
        if ims is not None and ims >= int(stats.st_mtime):
            return HTTPResponse(status=304, **headers)
    body = '' if request.method == 'HEAD' else open(filename, 'rb')
    headers['Accept-Ranges'] = 'bytes'
    range_header = getenv('HTTP_RANGE')
    if range_header:
        ranges = list(parse_range_header(range_header, clen))
        if not ranges:
            return HTTPError(416, 'Requested Range Not Satisfiable')
        (offset, end) = ranges[0]
        rlen = end - offset
        headers['Content-Range'] = 'bytes %d-%d/%d' % (offset, end - 1, clen)
        headers['Content-Length'] = str(rlen)
        if body:
            body = _closeiter(_rangeiter(body, offset, rlen), body.close)
        return HTTPResponse(body, status=206, **headers)
    return HTTPResponse(body, **headers)

def debug(mode=True):
    if False:
        return 10
    ' Change the debug level.\n    There is only one debug level supported at the moment.'
    global DEBUG
    if mode:
        warnings.simplefilter('default')
    DEBUG = bool(mode)

def http_date(value):
    if False:
        i = 10
        return i + 15
    if isinstance(value, basestring):
        return value
    if isinstance(value, datetime):
        value = value.utctimetuple()
    elif isinstance(value, datedate):
        value = value.timetuple()
    if not isinstance(value, (int, float)):
        value = calendar.timegm(value)
    return email.utils.formatdate(value, usegmt=True)

def parse_date(ims):
    if False:
        print('Hello World!')
    ' Parse rfc1123, rfc850 and asctime timestamps and return UTC epoch. '
    try:
        ts = email.utils.parsedate_tz(ims)
        return calendar.timegm(ts[:8] + (0,)) - (ts[9] or 0)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

def parse_auth(header):
    if False:
        print('Hello World!')
    ' Parse rfc2617 HTTP authentication header string (basic) and return (user,pass) tuple or None'
    try:
        (method, data) = header.split(None, 1)
        if method.lower() == 'basic':
            (user, pwd) = touni(base64.b64decode(tob(data))).split(':', 1)
            return (user, pwd)
    except (KeyError, ValueError):
        return None

def parse_range_header(header, maxlen=0):
    if False:
        for i in range(10):
            print('nop')
    ' Yield (start, end) ranges parsed from a HTTP Range header. Skip\n        unsatisfiable ranges. The end index is non-inclusive.'
    if not header or header[:6] != 'bytes=':
        return
    ranges = [r.split('-', 1) for r in header[6:].split(',') if '-' in r]
    for (start, end) in ranges:
        try:
            if not start:
                (start, end) = (max(0, maxlen - int(end)), maxlen)
            elif not end:
                (start, end) = (int(start), maxlen)
            else:
                (start, end) = (int(start), min(int(end) + 1, maxlen))
            if 0 <= start < end <= maxlen:
                yield (start, end)
        except ValueError:
            pass
_hsplit = re.compile('(?:(?:"((?:[^"\\\\]|\\\\.)*)")|([^;,=]+))([;,=]?)').findall

def _parse_http_header(h):
    if False:
        i = 10
        return i + 15
    ' Parses a typical multi-valued and parametrised HTTP header (e.g. Accept headers) and returns a list of values\n        and parameters. For non-standard or broken input, this implementation may return partial results.\n    :param h: A header string (e.g. ``text/html,text/plain;q=0.9,*/*;q=0.8``)\n    :return: List of (value, params) tuples. The second element is a (possibly empty) dict.\n    '
    values = []
    if '"' not in h:
        for value in h.split(','):
            parts = value.split(';')
            values.append((parts[0].strip(), {}))
            for attr in parts[1:]:
                (name, value) = attr.split('=', 1)
                values[-1][1][name.strip()] = value.strip()
    else:
        (lop, key, attrs) = (',', None, {})
        for (quoted, plain, tok) in _hsplit(h):
            value = plain.strip() if plain else quoted.replace('\\"', '"')
            if lop == ',':
                attrs = {}
                values.append((value, attrs))
            elif lop == ';':
                if tok == '=':
                    key = value
                else:
                    attrs[value] = ''
            elif lop == '=' and key:
                attrs[key] = value
                key = None
            lop = tok
    return values

def _parse_qsl(qs):
    if False:
        while True:
            i = 10
    r = []
    for pair in qs.split('&'):
        if not pair:
            continue
        nv = pair.split('=', 1)
        if len(nv) != 2:
            nv.append('')
        key = urlunquote(nv[0].replace('+', ' '))
        value = urlunquote(nv[1].replace('+', ' '))
        r.append((key, value))
    return r

def _lscmp(a, b):
    if False:
        print('Hello World!')
    ' Compares two strings in a cryptographically safe way:\n        Runtime is not affected by length of common prefix. '
    return not sum((0 if x == y else 1 for (x, y) in zip(a, b))) and len(a) == len(b)

def cookie_encode(data, key, digestmod=None):
    if False:
        i = 10
        return i + 15
    ' Encode and sign a pickle-able object. Return a (byte) string '
    depr(0, 13, 'cookie_encode() will be removed soon.', 'Do not use this API directly.')
    digestmod = digestmod or hashlib.sha256
    msg = base64.b64encode(pickle.dumps(data, -1))
    sig = base64.b64encode(hmac.new(tob(key), msg, digestmod=digestmod).digest())
    return tob('!') + sig + tob('?') + msg

def cookie_decode(data, key, digestmod=None):
    if False:
        return 10
    ' Verify and decode an encoded string. Return an object or None.'
    depr(0, 13, 'cookie_decode() will be removed soon.', 'Do not use this API directly.')
    data = tob(data)
    if cookie_is_encoded(data):
        (sig, msg) = data.split(tob('?'), 1)
        digestmod = digestmod or hashlib.sha256
        hashed = hmac.new(tob(key), msg, digestmod=digestmod).digest()
        if _lscmp(sig[1:], base64.b64encode(hashed)):
            return pickle.loads(base64.b64decode(msg))
    return None

def cookie_is_encoded(data):
    if False:
        print('Hello World!')
    ' Return True if the argument looks like a encoded cookie.'
    depr(0, 13, 'cookie_is_encoded() will be removed soon.', 'Do not use this API directly.')
    return bool(data.startswith(tob('!')) and tob('?') in data)

def html_escape(string):
    if False:
        return 10
    ' Escape HTML special characters ``&<>`` and quotes ``\'"``. '
    return string.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#039;')

def html_quote(string):
    if False:
        return 10
    ' Escape and quote a string to be used as an HTTP attribute.'
    return '"%s"' % html_escape(string).replace('\n', '&#10;').replace('\r', '&#13;').replace('\t', '&#9;')

def yieldroutes(func):
    if False:
        print('Hello World!')
    " Return a generator for routes that match the signature (name, args)\n    of the func parameter. This may yield more than one route if the function\n    takes optional keyword arguments. The output is best described by example::\n\n        a()         -> '/a'\n        b(x, y)     -> '/b/<x>/<y>'\n        c(x, y=5)   -> '/c/<x>' and '/c/<x>/<y>'\n        d(x=5, y=6) -> '/d' and '/d/<x>' and '/d/<x>/<y>'\n    "
    path = '/' + func.__name__.replace('__', '/').lstrip('/')
    spec = getargspec(func)
    argc = len(spec[0]) - len(spec[3] or [])
    path += '/<%s>' * argc % tuple(spec[0][:argc])
    yield path
    for arg in spec[0][argc:]:
        path += '/<%s>' % arg
        yield path

def path_shift(script_name, path_info, shift=1):
    if False:
        return 10
    ' Shift path fragments from PATH_INFO to SCRIPT_NAME and vice versa.\n\n        :return: The modified paths.\n        :param script_name: The SCRIPT_NAME path.\n        :param script_name: The PATH_INFO path.\n        :param shift: The number of path fragments to shift. May be negative to\n          change the shift direction. (default: 1)\n    '
    if shift == 0:
        return (script_name, path_info)
    pathlist = path_info.strip('/').split('/')
    scriptlist = script_name.strip('/').split('/')
    if pathlist and pathlist[0] == '':
        pathlist = []
    if scriptlist and scriptlist[0] == '':
        scriptlist = []
    if 0 < shift <= len(pathlist):
        moved = pathlist[:shift]
        scriptlist = scriptlist + moved
        pathlist = pathlist[shift:]
    elif 0 > shift >= -len(scriptlist):
        moved = scriptlist[shift:]
        pathlist = moved + pathlist
        scriptlist = scriptlist[:shift]
    else:
        empty = 'SCRIPT_NAME' if shift < 0 else 'PATH_INFO'
        raise AssertionError('Cannot shift. Nothing left from %s' % empty)
    new_script_name = '/' + '/'.join(scriptlist)
    new_path_info = '/' + '/'.join(pathlist)
    if path_info.endswith('/') and pathlist:
        new_path_info += '/'
    return (new_script_name, new_path_info)

def auth_basic(check, realm='private', text='Access denied'):
    if False:
        print('Hello World!')
    ' Callback decorator to require HTTP auth (basic).\n        TODO: Add route(check_auth=...) parameter. '

    def decorator(func):
        if False:
            while True:
                i = 10

        @functools.wraps(func)
        def wrapper(*a, **ka):
            if False:
                print('Hello World!')
            (user, password) = request.auth or (None, None)
            if user is None or not check(user, password):
                err = HTTPError(401, text)
                err.add_header('WWW-Authenticate', 'Basic realm="%s"' % realm)
                return err
            return func(*a, **ka)
        return wrapper
    return decorator

def make_default_app_wrapper(name):
    if False:
        while True:
            i = 10
    ' Return a callable that relays calls to the current default app. '

    @functools.wraps(getattr(Bottle, name))
    def wrapper(*a, **ka):
        if False:
            while True:
                i = 10
        return getattr(app(), name)(*a, **ka)
    return wrapper
route = make_default_app_wrapper('route')
get = make_default_app_wrapper('get')
post = make_default_app_wrapper('post')
put = make_default_app_wrapper('put')
delete = make_default_app_wrapper('delete')
patch = make_default_app_wrapper('patch')
error = make_default_app_wrapper('error')
mount = make_default_app_wrapper('mount')
hook = make_default_app_wrapper('hook')
install = make_default_app_wrapper('install')
uninstall = make_default_app_wrapper('uninstall')
url = make_default_app_wrapper('get_url')

class ServerAdapter(object):
    quiet = False

    def __init__(self, host='127.0.0.1', port=8080, **options):
        if False:
            print('Hello World!')
        self.options = options
        self.host = host
        self.port = int(port)

    def run(self, handler):
        if False:
            i = 10
            return i + 15
        pass

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        args = ', '.join(('%s=%s' % (k, repr(v)) for (k, v) in self.options.items()))
        return '%s(%s)' % (self.__class__.__name__, args)

class CGIServer(ServerAdapter):
    quiet = True

    def run(self, handler):
        if False:
            print('Hello World!')
        from wsgiref.handlers import CGIHandler

        def fixed_environ(environ, start_response):
            if False:
                return 10
            environ.setdefault('PATH_INFO', '')
            return handler(environ, start_response)
        CGIHandler().run(fixed_environ)

class FlupFCGIServer(ServerAdapter):

    def run(self, handler):
        if False:
            for i in range(10):
                print('nop')
        import flup.server.fcgi
        self.options.setdefault('bindAddress', (self.host, self.port))
        flup.server.fcgi.WSGIServer(handler, **self.options).run()

class WSGIRefServer(ServerAdapter):

    def run(self, app):
        if False:
            i = 10
            return i + 15
        from wsgiref.simple_server import make_server
        from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
        import socket

        class FixedHandler(WSGIRequestHandler):

            def address_string(self):
                if False:
                    while True:
                        i = 10
                return self.client_address[0]

            def log_request(*args, **kw):
                if False:
                    print('Hello World!')
                if not self.quiet:
                    return WSGIRequestHandler.log_request(*args, **kw)
        handler_cls = self.options.get('handler_class', FixedHandler)
        server_cls = self.options.get('server_class', WSGIServer)
        if ':' in self.host:
            if getattr(server_cls, 'address_family') == socket.AF_INET:

                class server_cls(server_cls):
                    address_family = socket.AF_INET6
        self.srv = make_server(self.host, self.port, app, server_cls, handler_cls)
        self.port = self.srv.server_port
        try:
            self.srv.serve_forever()
        except KeyboardInterrupt:
            self.srv.server_close()
            raise

class CherryPyServer(ServerAdapter):

    def run(self, handler):
        if False:
            for i in range(10):
                print('nop')
        depr(0, 13, "The wsgi server part of cherrypy was split into a new project called 'cheroot'.", "Use the 'cheroot' server adapter instead of cherrypy.")
        from cherrypy import wsgiserver
        self.options['bind_addr'] = (self.host, self.port)
        self.options['wsgi_app'] = handler
        certfile = self.options.get('certfile')
        if certfile:
            del self.options['certfile']
        keyfile = self.options.get('keyfile')
        if keyfile:
            del self.options['keyfile']
        server = wsgiserver.CherryPyWSGIServer(**self.options)
        if certfile:
            server.ssl_certificate = certfile
        if keyfile:
            server.ssl_private_key = keyfile
        try:
            server.start()
        finally:
            server.stop()

class CherootServer(ServerAdapter):

    def run(self, handler):
        if False:
            for i in range(10):
                print('nop')
        from cheroot import wsgi
        from cheroot.ssl import builtin
        self.options['bind_addr'] = (self.host, self.port)
        self.options['wsgi_app'] = handler
        certfile = self.options.pop('certfile', None)
        keyfile = self.options.pop('keyfile', None)
        chainfile = self.options.pop('chainfile', None)
        server = wsgi.Server(**self.options)
        if certfile and keyfile:
            server.ssl_adapter = builtin.BuiltinSSLAdapter(certfile, keyfile, chainfile)
        try:
            server.start()
        finally:
            server.stop()

class WaitressServer(ServerAdapter):

    def run(self, handler):
        if False:
            for i in range(10):
                print('nop')
        from waitress import serve
        serve(handler, host=self.host, port=self.port, _quiet=self.quiet, **self.options)

class PasteServer(ServerAdapter):

    def run(self, handler):
        if False:
            i = 10
            return i + 15
        from paste import httpserver
        from paste.translogger import TransLogger
        handler = TransLogger(handler, setup_console_handler=not self.quiet)
        httpserver.serve(handler, host=self.host, port=str(self.port), **self.options)

class MeinheldServer(ServerAdapter):

    def run(self, handler):
        if False:
            while True:
                i = 10
        from meinheld import server
        server.listen((self.host, self.port))
        server.run(handler)

class FapwsServer(ServerAdapter):
    """ Extremely fast webserver using libev. See https://github.com/william-os4y/fapws3 """

    def run(self, handler):
        if False:
            return 10
        depr(0, 13, 'fapws3 is not maintained and support will be dropped.')
        import fapws._evwsgi as evwsgi
        from fapws import base, config
        port = self.port
        if float(config.SERVER_IDENT[-2:]) > 0.4:
            port = str(port)
        evwsgi.start(self.host, port)
        if 'BOTTLE_CHILD' in os.environ and (not self.quiet):
            _stderr('WARNING: Auto-reloading does not work with Fapws3.')
            _stderr('         (Fapws3 breaks python thread support)')
        evwsgi.set_base_module(base)

        def app(environ, start_response):
            if False:
                while True:
                    i = 10
            environ['wsgi.multiprocess'] = False
            return handler(environ, start_response)
        evwsgi.wsgi_cb(('', app))
        evwsgi.run()

class TornadoServer(ServerAdapter):
    """ The super hyped asynchronous server by facebook. Untested. """

    def run(self, handler):
        if False:
            i = 10
            return i + 15
        import tornado.wsgi, tornado.httpserver, tornado.ioloop
        container = tornado.wsgi.WSGIContainer(handler)
        server = tornado.httpserver.HTTPServer(container)
        server.listen(port=self.port, address=self.host)
        tornado.ioloop.IOLoop.instance().start()

class AppEngineServer(ServerAdapter):
    """ Adapter for Google App Engine. """
    quiet = True

    def run(self, handler):
        if False:
            i = 10
            return i + 15
        depr(0, 13, 'AppEngineServer no longer required', 'Configure your application directly in your app.yaml')
        from google.appengine.ext.webapp import util
        module = sys.modules.get('__main__')
        if module and (not hasattr(module, 'main')):
            module.main = lambda : util.run_wsgi_app(handler)
        util.run_wsgi_app(handler)

class TwistedServer(ServerAdapter):
    """ Untested. """

    def run(self, handler):
        if False:
            while True:
                i = 10
        from twisted.web import server, wsgi
        from twisted.python.threadpool import ThreadPool
        from twisted.internet import reactor
        thread_pool = ThreadPool()
        thread_pool.start()
        reactor.addSystemEventTrigger('after', 'shutdown', thread_pool.stop)
        factory = server.Site(wsgi.WSGIResource(reactor, thread_pool, handler))
        reactor.listenTCP(self.port, factory, interface=self.host)
        if not reactor.running:
            reactor.run()

class DieselServer(ServerAdapter):
    """ Untested. """

    def run(self, handler):
        if False:
            return 10
        depr(0, 13, 'Diesel is not tested or supported and will be removed.')
        from diesel.protocols.wsgi import WSGIApplication
        app = WSGIApplication(handler, port=self.port)
        app.run()

class GeventServer(ServerAdapter):
    """ Untested. Options:

        * See gevent.wsgi.WSGIServer() documentation for more options.
    """

    def run(self, handler):
        if False:
            print('Hello World!')
        from gevent import pywsgi, local
        if not isinstance(threading.local(), local.local):
            msg = 'Bottle requires gevent.monkey.patch_all() (before import)'
            raise RuntimeError(msg)
        if self.quiet:
            self.options['log'] = None
        address = (self.host, self.port)
        server = pywsgi.WSGIServer(address, handler, **self.options)
        if 'BOTTLE_CHILD' in os.environ:
            import signal
            signal.signal(signal.SIGINT, lambda s, f: server.stop())
        server.serve_forever()

class GunicornServer(ServerAdapter):
    """ Untested. See http://gunicorn.org/configure.html for options. """

    def run(self, handler):
        if False:
            for i in range(10):
                print('nop')
        from gunicorn.app.base import BaseApplication
        if self.host.startswith('unix:'):
            config = {'bind': self.host}
        else:
            config = {'bind': '%s:%d' % (self.host, self.port)}
        config.update(self.options)

        class GunicornApplication(BaseApplication):

            def load_config(self):
                if False:
                    return 10
                for (key, value) in config.items():
                    self.cfg.set(key, value)

            def load(self):
                if False:
                    return 10
                return handler
        GunicornApplication().run()

class EventletServer(ServerAdapter):
    """ Untested. Options:

        * `backlog` adjust the eventlet backlog parameter which is the maximum
          number of queued connections. Should be at least 1; the maximum
          value is system-dependent.
        * `family`: (default is 2) socket family, optional. See socket
          documentation for available families.
    """

    def run(self, handler):
        if False:
            while True:
                i = 10
        from eventlet import wsgi, listen, patcher
        if not patcher.is_monkey_patched(os):
            msg = 'Bottle requires eventlet.monkey_patch() (before import)'
            raise RuntimeError(msg)
        socket_args = {}
        for arg in ('backlog', 'family'):
            try:
                socket_args[arg] = self.options.pop(arg)
            except KeyError:
                pass
        address = (self.host, self.port)
        try:
            wsgi.server(listen(address, **socket_args), handler, log_output=not self.quiet)
        except TypeError:
            wsgi.server(listen(address), handler)

class BjoernServer(ServerAdapter):
    """ Fast server written in C: https://github.com/jonashaag/bjoern """

    def run(self, handler):
        if False:
            for i in range(10):
                print('nop')
        from bjoern import run
        run(handler, self.host, self.port, reuse_port=True)

class AsyncioServerAdapter(ServerAdapter):
    """ Extend ServerAdapter for adding custom event loop """

    def get_event_loop(self):
        if False:
            return 10
        pass

class AiohttpServer(AsyncioServerAdapter):
    """ Asynchronous HTTP client/server framework for asyncio
        https://pypi.python.org/pypi/aiohttp/
        https://pypi.org/project/aiohttp-wsgi/
    """

    def get_event_loop(self):
        if False:
            for i in range(10):
                print('nop')
        import asyncio
        return asyncio.new_event_loop()

    def run(self, handler):
        if False:
            i = 10
            return i + 15
        import asyncio
        from aiohttp_wsgi.wsgi import serve
        self.loop = self.get_event_loop()
        asyncio.set_event_loop(self.loop)
        if 'BOTTLE_CHILD' in os.environ:
            import signal
            signal.signal(signal.SIGINT, lambda s, f: self.loop.stop())
        serve(handler, host=self.host, port=self.port)

class AiohttpUVLoopServer(AiohttpServer):
    """uvloop
       https://github.com/MagicStack/uvloop
    """

    def get_event_loop(self):
        if False:
            i = 10
            return i + 15
        import uvloop
        return uvloop.new_event_loop()

class AutoServer(ServerAdapter):
    """ Untested. """
    adapters = [WaitressServer, PasteServer, TwistedServer, CherryPyServer, CherootServer, WSGIRefServer]

    def run(self, handler):
        if False:
            i = 10
            return i + 15
        for sa in self.adapters:
            try:
                return sa(self.host, self.port, **self.options).run(handler)
            except ImportError:
                pass
server_names = {'cgi': CGIServer, 'flup': FlupFCGIServer, 'wsgiref': WSGIRefServer, 'waitress': WaitressServer, 'cherrypy': CherryPyServer, 'cheroot': CherootServer, 'paste': PasteServer, 'fapws3': FapwsServer, 'tornado': TornadoServer, 'gae': AppEngineServer, 'twisted': TwistedServer, 'diesel': DieselServer, 'meinheld': MeinheldServer, 'gunicorn': GunicornServer, 'eventlet': EventletServer, 'gevent': GeventServer, 'bjoern': BjoernServer, 'aiohttp': AiohttpServer, 'uvloop': AiohttpUVLoopServer, 'auto': AutoServer}

def load(target, **namespace):
    if False:
        print('Hello World!')
    " Import a module or fetch an object from a module.\n\n        * ``package.module`` returns `module` as a module object.\n        * ``pack.mod:name`` returns the module variable `name` from `pack.mod`.\n        * ``pack.mod:func()`` calls `pack.mod.func()` and returns the result.\n\n        The last form accepts not only function calls, but any type of\n        expression. Keyword arguments passed to this function are available as\n        local variables. Example: ``import_string('re:compile(x)', x='[a-z]')``\n    "
    (module, target) = target.split(':', 1) if ':' in target else (target, None)
    if module not in sys.modules:
        __import__(module)
    if not target:
        return sys.modules[module]
    if target.isalnum():
        return getattr(sys.modules[module], target)
    package_name = module.split('.')[0]
    namespace[package_name] = sys.modules[package_name]
    return eval('%s.%s' % (module, target), namespace)

def load_app(target):
    if False:
        print('Hello World!')
    ' Load a bottle application from a module and make sure that the import\n        does not affect the current default application, but returns a separate\n        application object. See :func:`load` for the target parameter. '
    global NORUN
    (NORUN, nr_old) = (True, NORUN)
    tmp = default_app.push()
    try:
        rv = load(target)
        return rv if callable(rv) else tmp
    finally:
        default_app.remove(tmp)
        NORUN = nr_old
_debug = debug

def run(app=None, server='wsgiref', host='127.0.0.1', port=8080, interval=1, reloader=False, quiet=False, plugins=None, debug=None, config=None, **kargs):
    if False:
        for i in range(10):
            print('nop')
    ' Start a server instance. This method blocks until the server terminates.\n\n        :param app: WSGI application or target string supported by\n               :func:`load_app`. (default: :func:`default_app`)\n        :param server: Server adapter to use. See :data:`server_names` keys\n               for valid names or pass a :class:`ServerAdapter` subclass.\n               (default: `wsgiref`)\n        :param host: Server address to bind to. Pass ``0.0.0.0`` to listens on\n               all interfaces including the external one. (default: 127.0.0.1)\n        :param port: Server port to bind to. Values below 1024 require root\n               privileges. (default: 8080)\n        :param reloader: Start auto-reloading server? (default: False)\n        :param interval: Auto-reloader interval in seconds (default: 1)\n        :param quiet: Suppress output to stdout and stderr? (default: False)\n        :param options: Options passed to the server adapter.\n     '
    if NORUN:
        return
    if reloader and (not os.environ.get('BOTTLE_CHILD')):
        import subprocess
        (fd, lockfile) = tempfile.mkstemp(prefix='bottle.', suffix='.lock')
        environ = os.environ.copy()
        environ['BOTTLE_CHILD'] = 'true'
        environ['BOTTLE_LOCKFILE'] = lockfile
        args = [sys.executable] + sys.argv
        if getattr(sys.modules.get('__main__'), '__package__', None):
            args[1:1] = ['-m', sys.modules['__main__'].__package__]
        try:
            os.close(fd)
            while os.path.exists(lockfile):
                p = subprocess.Popen(args, env=environ)
                while p.poll() is None:
                    os.utime(lockfile, None)
                    time.sleep(interval)
                if p.returncode == 3:
                    continue
                sys.exit(p.returncode)
        except KeyboardInterrupt:
            pass
        finally:
            if os.path.exists(lockfile):
                os.unlink(lockfile)
        return
    try:
        if debug is not None:
            _debug(debug)
        app = app or default_app()
        if isinstance(app, basestring):
            app = load_app(app)
        if not callable(app):
            raise ValueError('Application is not callable: %r' % app)
        for plugin in plugins or []:
            if isinstance(plugin, basestring):
                plugin = load(plugin)
            app.install(plugin)
        if config:
            app.config.update(config)
        if server in server_names:
            server = server_names.get(server)
        if isinstance(server, basestring):
            server = load(server)
        if isinstance(server, type):
            server = server(host=host, port=port, **kargs)
        if not isinstance(server, ServerAdapter):
            raise ValueError('Unknown or unsupported server: %r' % server)
        server.quiet = server.quiet or quiet
        if not server.quiet:
            _stderr('Bottle v%s server starting up (using %s)...' % (__version__, repr(server)))
            if server.host.startswith('unix:'):
                _stderr('Listening on %s' % server.host)
            else:
                _stderr('Listening on http://%s:%d/' % (server.host, server.port))
            _stderr('Hit Ctrl-C to quit.\n')
        if reloader:
            lockfile = os.environ.get('BOTTLE_LOCKFILE')
            bgcheck = FileCheckerThread(lockfile, interval)
            with bgcheck:
                server.run(app)
            if bgcheck.status == 'reload':
                sys.exit(3)
        else:
            server.run(app)
    except KeyboardInterrupt:
        pass
    except (SystemExit, MemoryError):
        raise
    except:
        if not reloader:
            raise
        if not getattr(server, 'quiet', quiet):
            print_exc()
        time.sleep(interval)
        sys.exit(3)

class FileCheckerThread(threading.Thread):
    """ Interrupt main-thread as soon as a changed module file is detected,
        the lockfile gets deleted or gets too old. """

    def __init__(self, lockfile, interval):
        if False:
            for i in range(10):
                print('nop')
        threading.Thread.__init__(self)
        self.daemon = True
        (self.lockfile, self.interval) = (lockfile, interval)
        self.status = None

    def run(self):
        if False:
            print('Hello World!')
        exists = os.path.exists
        mtime = lambda p: os.stat(p).st_mtime
        files = dict()
        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', '') or ''
            if path[-4:] in ('.pyo', '.pyc'):
                path = path[:-1]
            if path and exists(path):
                files[path] = mtime(path)
        while not self.status:
            if not exists(self.lockfile) or mtime(self.lockfile) < time.time() - self.interval - 5:
                self.status = 'error'
                thread.interrupt_main()
            for (path, lmtime) in list(files.items()):
                if not exists(path) or mtime(path) > lmtime:
                    self.status = 'reload'
                    thread.interrupt_main()
                    break
            time.sleep(self.interval)

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.start()

    def __exit__(self, exc_type, *_):
        if False:
            print('Hello World!')
        if not self.status:
            self.status = 'exit'
        self.join()
        return exc_type is not None and issubclass(exc_type, KeyboardInterrupt)

class TemplateError(BottleException):
    pass

class BaseTemplate(object):
    """ Base class and minimal API for template adapters """
    extensions = ['tpl', 'html', 'thtml', 'stpl']
    settings = {}
    defaults = {}

    def __init__(self, source=None, name=None, lookup=None, encoding='utf8', **settings):
        if False:
            print('Hello World!')
        ' Create a new template.\n        If the source parameter (str or buffer) is missing, the name argument\n        is used to guess a template filename. Subclasses can assume that\n        self.source and/or self.filename are set. Both are strings.\n        The lookup, encoding and settings parameters are stored as instance\n        variables.\n        The lookup parameter stores a list containing directory paths.\n        The encoding parameter should be used to decode byte strings or files.\n        The settings parameter contains a dict for engine-specific settings.\n        '
        self.name = name
        self.source = source.read() if hasattr(source, 'read') else source
        self.filename = source.filename if hasattr(source, 'filename') else None
        self.lookup = [os.path.abspath(x) for x in lookup] if lookup else []
        self.encoding = encoding
        self.settings = self.settings.copy()
        self.settings.update(settings)
        if not self.source and self.name:
            self.filename = self.search(self.name, self.lookup)
            if not self.filename:
                raise TemplateError('Template %s not found.' % repr(name))
        if not self.source and (not self.filename):
            raise TemplateError('No template specified.')
        self.prepare(**self.settings)

    @classmethod
    def search(cls, name, lookup=None):
        if False:
            return 10
        ' Search name in all directories specified in lookup.\n        First without, then with common extensions. Return first hit. '
        if not lookup:
            raise depr(0, 12, 'Empty template lookup path.', 'Configure a template lookup path.')
        if os.path.isabs(name):
            raise depr(0, 12, 'Use of absolute path for template name.', 'Refer to templates with names or paths relative to the lookup path.')
        for spath in lookup:
            spath = os.path.abspath(spath) + os.sep
            fname = os.path.abspath(os.path.join(spath, name))
            if not fname.startswith(spath):
                continue
            if os.path.isfile(fname):
                return fname
            for ext in cls.extensions:
                if os.path.isfile('%s.%s' % (fname, ext)):
                    return '%s.%s' % (fname, ext)

    @classmethod
    def global_config(cls, key, *args):
        if False:
            for i in range(10):
                print('nop')
        ' This reads or sets the global settings stored in class.settings. '
        if args:
            cls.settings = cls.settings.copy()
            cls.settings[key] = args[0]
        else:
            return cls.settings[key]

    def prepare(self, **options):
        if False:
            return 10
        ' Run preparations (parsing, caching, ...).\n        It should be possible to call this again to refresh a template or to\n        update settings.\n        '
        raise NotImplementedError

    def render(self, *args, **kwargs):
        if False:
            print('Hello World!')
        ' Render the template with the specified local variables and return\n        a single byte or unicode string. If it is a byte string, the encoding\n        must match self.encoding. This method must be thread-safe!\n        Local variables may be provided in dictionaries (args)\n        or directly, as keywords (kwargs).\n        '
        raise NotImplementedError

class MakoTemplate(BaseTemplate):

    def prepare(self, **options):
        if False:
            return 10
        from mako.template import Template
        from mako.lookup import TemplateLookup
        options.update({'input_encoding': self.encoding})
        options.setdefault('format_exceptions', bool(DEBUG))
        lookup = TemplateLookup(directories=self.lookup, **options)
        if self.source:
            self.tpl = Template(self.source, lookup=lookup, **options)
        else:
            self.tpl = Template(uri=self.name, filename=self.filename, lookup=lookup, **options)

    def render(self, *args, **kwargs):
        if False:
            print('Hello World!')
        for dictarg in args:
            kwargs.update(dictarg)
        _defaults = self.defaults.copy()
        _defaults.update(kwargs)
        return self.tpl.render(**_defaults)

class CheetahTemplate(BaseTemplate):

    def prepare(self, **options):
        if False:
            for i in range(10):
                print('nop')
        from Cheetah.Template import Template
        self.context = threading.local()
        self.context.vars = {}
        options['searchList'] = [self.context.vars]
        if self.source:
            self.tpl = Template(source=self.source, **options)
        else:
            self.tpl = Template(file=self.filename, **options)

    def render(self, *args, **kwargs):
        if False:
            print('Hello World!')
        for dictarg in args:
            kwargs.update(dictarg)
        self.context.vars.update(self.defaults)
        self.context.vars.update(kwargs)
        out = str(self.tpl)
        self.context.vars.clear()
        return out

class Jinja2Template(BaseTemplate):

    def prepare(self, filters=None, tests=None, globals={}, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from jinja2 import Environment, FunctionLoader
        self.env = Environment(loader=FunctionLoader(self.loader), **kwargs)
        if filters:
            self.env.filters.update(filters)
        if tests:
            self.env.tests.update(tests)
        if globals:
            self.env.globals.update(globals)
        if self.source:
            self.tpl = self.env.from_string(self.source)
        else:
            self.tpl = self.env.get_template(self.name)

    def render(self, *args, **kwargs):
        if False:
            print('Hello World!')
        for dictarg in args:
            kwargs.update(dictarg)
        _defaults = self.defaults.copy()
        _defaults.update(kwargs)
        return self.tpl.render(**_defaults)

    def loader(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name == self.filename:
            fname = name
        else:
            fname = self.search(name, self.lookup)
        if not fname:
            return
        with open(fname, 'rb') as f:
            return (f.read().decode(self.encoding), fname, lambda : False)

class SimpleTemplate(BaseTemplate):

    def prepare(self, escape_func=html_escape, noescape=False, syntax=None, **ka):
        if False:
            while True:
                i = 10
        self.cache = {}
        enc = self.encoding
        self._str = lambda x: touni(x, enc)
        self._escape = lambda x: escape_func(touni(x, enc))
        self.syntax = syntax
        if noescape:
            (self._str, self._escape) = (self._escape, self._str)

    @cached_property
    def co(self):
        if False:
            while True:
                i = 10
        return compile(self.code, self.filename or '<string>', 'exec')

    @cached_property
    def code(self):
        if False:
            for i in range(10):
                print('nop')
        source = self.source
        if not source:
            with open(self.filename, 'rb') as f:
                source = f.read()
        try:
            (source, encoding) = (touni(source), 'utf8')
        except UnicodeError:
            raise depr(0, 11, 'Unsupported template encodings.', 'Use utf-8 for templates.')
        parser = StplParser(source, encoding=encoding, syntax=self.syntax)
        code = parser.translate()
        self.encoding = parser.encoding
        return code

    def _rebase(self, _env, _name=None, **kwargs):
        if False:
            while True:
                i = 10
        _env['_rebase'] = (_name, kwargs)

    def _include(self, _env, _name=None, **kwargs):
        if False:
            while True:
                i = 10
        env = _env.copy()
        env.update(kwargs)
        if _name not in self.cache:
            self.cache[_name] = self.__class__(name=_name, lookup=self.lookup, syntax=self.syntax)
        return self.cache[_name].execute(env['_stdout'], env)

    def execute(self, _stdout, kwargs):
        if False:
            return 10
        env = self.defaults.copy()
        env.update(kwargs)
        env.update({'_stdout': _stdout, '_printlist': _stdout.extend, 'include': functools.partial(self._include, env), 'rebase': functools.partial(self._rebase, env), '_rebase': None, '_str': self._str, '_escape': self._escape, 'get': env.get, 'setdefault': env.setdefault, 'defined': env.__contains__})
        exec(self.co, env)
        if env.get('_rebase'):
            (subtpl, rargs) = env.pop('_rebase')
            rargs['base'] = ''.join(_stdout)
            del _stdout[:]
            return self._include(env, subtpl, **rargs)
        return env

    def render(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        ' Render the template using keyword arguments as local variables. '
        env = {}
        stdout = []
        for dictarg in args:
            env.update(dictarg)
        env.update(kwargs)
        self.execute(stdout, env)
        return ''.join(stdout)

class StplSyntaxError(TemplateError):
    pass

class StplParser(object):
    """ Parser for stpl templates. """
    _re_cache = {}
    _re_tok = '(\n        [urbURB]*\n        (?:  \'\'(?!\')\n            |""(?!")\n            |\'{6}\n            |"{6}\n            |\'(?:[^\\\\\']|\\\\.)+?\'\n            |"(?:[^\\\\"]|\\\\.)+?"\n            |\'{3}(?:[^\\\\]|\\\\.|\\n)+?\'{3}\n            |"{3}(?:[^\\\\]|\\\\.|\\n)+?"{3}\n        )\n    )'
    _re_inl = _re_tok.replace('|\\n', '')
    _re_tok += "\n        # 2: Comments (until end of line, but not the newline itself)\n        |(\\#.*)\n\n        # 3: Open and close (4) grouping tokens\n        |([\\[\\{\\(])\n        |([\\]\\}\\)])\n\n        # 5,6: Keywords that start or continue a python block (only start of line)\n        |^([\\ \\t]*(?:if|for|while|with|try|def|class)\\b)\n        |^([\\ \\t]*(?:elif|else|except|finally)\\b)\n\n        # 7: Our special 'end' keyword (but only if it stands alone)\n        |((?:^|;)[\\ \\t]*end[\\ \\t]*(?=(?:%(block_close)s[\\ \\t]*)?\\r?$|;|\\#))\n\n        # 8: A customizable end-of-code-block template token (only end of line)\n        |(%(block_close)s[\\ \\t]*(?=\\r?$))\n\n        # 9: And finally, a single newline. The 10th token is 'everything else'\n        |(\\r?\\n)\n    "
    _re_split = '(?m)^[ \\t]*(\\\\?)((%(line_start)s)|(%(block_start)s))'
    _re_inl = '%%(inline_start)s((?:%s|[^\'"\\n])*?)%%(inline_end)s' % _re_inl
    _re_tok = '(?mx)' + _re_tok
    _re_inl = '(?mx)' + _re_inl
    default_syntax = '<% %> % {{ }}'

    def __init__(self, source, syntax=None, encoding='utf8'):
        if False:
            while True:
                i = 10
        (self.source, self.encoding) = (touni(source, encoding), encoding)
        self.set_syntax(syntax or self.default_syntax)
        (self.code_buffer, self.text_buffer) = ([], [])
        (self.lineno, self.offset) = (1, 0)
        (self.indent, self.indent_mod) = (0, 0)
        self.paren_depth = 0

    def get_syntax(self):
        if False:
            for i in range(10):
                print('nop')
        ' Tokens as a space separated string (default: <% %> % {{ }}) '
        return self._syntax

    def set_syntax(self, syntax):
        if False:
            for i in range(10):
                print('nop')
        self._syntax = syntax
        self._tokens = syntax.split()
        if syntax not in self._re_cache:
            names = 'block_start block_close line_start inline_start inline_end'
            etokens = map(re.escape, self._tokens)
            pattern_vars = dict(zip(names.split(), etokens))
            patterns = (self._re_split, self._re_tok, self._re_inl)
            patterns = [re.compile(p % pattern_vars) for p in patterns]
            self._re_cache[syntax] = patterns
        (self.re_split, self.re_tok, self.re_inl) = self._re_cache[syntax]
    syntax = property(get_syntax, set_syntax)

    def translate(self):
        if False:
            for i in range(10):
                print('nop')
        if self.offset:
            raise RuntimeError('Parser is a one time instance.')
        while True:
            m = self.re_split.search(self.source, pos=self.offset)
            if m:
                text = self.source[self.offset:m.start()]
                self.text_buffer.append(text)
                self.offset = m.end()
                if m.group(1):
                    (line, sep, _) = self.source[self.offset:].partition('\n')
                    self.text_buffer.append(self.source[m.start():m.start(1)] + m.group(2) + line + sep)
                    self.offset += len(line + sep)
                    continue
                self.flush_text()
                self.offset += self.read_code(self.source[self.offset:], multiline=bool(m.group(4)))
            else:
                break
        self.text_buffer.append(self.source[self.offset:])
        self.flush_text()
        return ''.join(self.code_buffer)

    def read_code(self, pysource, multiline):
        if False:
            return 10
        (code_line, comment) = ('', '')
        offset = 0
        while True:
            m = self.re_tok.search(pysource, pos=offset)
            if not m:
                code_line += pysource[offset:]
                offset = len(pysource)
                self.write_code(code_line.strip(), comment)
                break
            code_line += pysource[offset:m.start()]
            offset = m.end()
            (_str, _com, _po, _pc, _blk1, _blk2, _end, _cend, _nl) = m.groups()
            if self.paren_depth > 0 and (_blk1 or _blk2):
                code_line += _blk1 or _blk2
                continue
            if _str:
                code_line += _str
            elif _com:
                comment = _com
                if multiline and _com.strip().endswith(self._tokens[1]):
                    multiline = False
            elif _po:
                self.paren_depth += 1
                code_line += _po
            elif _pc:
                if self.paren_depth > 0:
                    self.paren_depth -= 1
                code_line += _pc
            elif _blk1:
                code_line = _blk1
                self.indent += 1
                self.indent_mod -= 1
            elif _blk2:
                code_line = _blk2
                self.indent_mod -= 1
            elif _cend:
                if multiline:
                    multiline = False
                else:
                    code_line += _cend
            elif _end:
                self.indent -= 1
                self.indent_mod += 1
            else:
                self.write_code(code_line.strip(), comment)
                self.lineno += 1
                (code_line, comment, self.indent_mod) = ('', '', 0)
                if not multiline:
                    break
        return offset

    def flush_text(self):
        if False:
            i = 10
            return i + 15
        text = ''.join(self.text_buffer)
        del self.text_buffer[:]
        if not text:
            return
        (parts, pos, nl) = ([], 0, '\\\n' + '  ' * self.indent)
        for m in self.re_inl.finditer(text):
            (prefix, pos) = (text[pos:m.start()], m.end())
            if prefix:
                parts.append(nl.join(map(repr, prefix.splitlines(True))))
            if prefix.endswith('\n'):
                parts[-1] += nl
            parts.append(self.process_inline(m.group(1).strip()))
        if pos < len(text):
            prefix = text[pos:]
            lines = prefix.splitlines(True)
            if lines[-1].endswith('\\\\\n'):
                lines[-1] = lines[-1][:-3]
            elif lines[-1].endswith('\\\\\r\n'):
                lines[-1] = lines[-1][:-4]
            parts.append(nl.join(map(repr, lines)))
        code = '_printlist((%s,))' % ', '.join(parts)
        self.lineno += code.count('\n') + 1
        self.write_code(code)

    @staticmethod
    def process_inline(chunk):
        if False:
            return 10
        if chunk[0] == '!':
            return '_str(%s)' % chunk[1:]
        return '_escape(%s)' % chunk

    def write_code(self, line, comment=''):
        if False:
            return 10
        code = '  ' * (self.indent + self.indent_mod)
        code += line.lstrip() + comment + '\n'
        self.code_buffer.append(code)

def template(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Get a rendered template as a string iterator.\n    You can use a name, a filename or a template string as first parameter.\n    Template rendering arguments can be passed as dictionaries\n    or directly (as keyword arguments).\n    '
    tpl = args[0] if args else None
    for dictarg in args[1:]:
        kwargs.update(dictarg)
    adapter = kwargs.pop('template_adapter', SimpleTemplate)
    lookup = kwargs.pop('template_lookup', TEMPLATE_PATH)
    tplid = (id(lookup), tpl)
    if tplid not in TEMPLATES or DEBUG:
        settings = kwargs.pop('template_settings', {})
        if isinstance(tpl, adapter):
            TEMPLATES[tplid] = tpl
            if settings:
                TEMPLATES[tplid].prepare(**settings)
        elif '\n' in tpl or '{' in tpl or '%' in tpl or ('$' in tpl):
            TEMPLATES[tplid] = adapter(source=tpl, lookup=lookup, **settings)
        else:
            TEMPLATES[tplid] = adapter(name=tpl, lookup=lookup, **settings)
    if not TEMPLATES[tplid]:
        abort(500, 'Template (%s) not found' % tpl)
    return TEMPLATES[tplid].render(kwargs)
mako_template = functools.partial(template, template_adapter=MakoTemplate)
cheetah_template = functools.partial(template, template_adapter=CheetahTemplate)
jinja2_template = functools.partial(template, template_adapter=Jinja2Template)

def view(tpl_name, **defaults):
    if False:
        while True:
            i = 10
    ' Decorator: renders a template for a handler.\n        The handler can control its behavior like that:\n\n          - return a dict of template vars to fill out the template\n          - return something other than a dict and the view decorator will not\n            process the template, but return the handler result as is.\n            This includes returning a HTTPResponse(dict) to get,\n            for instance, JSON with autojson or other castfilters.\n    '

    def decorator(func):
        if False:
            i = 10
            return i + 15

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            result = func(*args, **kwargs)
            if isinstance(result, (dict, DictMixin)):
                tplvars = defaults.copy()
                tplvars.update(result)
                return template(tpl_name, **tplvars)
            elif result is None:
                return template(tpl_name, **defaults)
            return result
        return wrapper
    return decorator
mako_view = functools.partial(view, template_adapter=MakoTemplate)
cheetah_view = functools.partial(view, template_adapter=CheetahTemplate)
jinja2_view = functools.partial(view, template_adapter=Jinja2Template)
TEMPLATE_PATH = ['./', './views/']
TEMPLATES = {}
DEBUG = False
NORUN = False
HTTP_CODES = httplib.responses.copy()
HTTP_CODES[418] = "I'm a teapot"
HTTP_CODES[428] = 'Precondition Required'
HTTP_CODES[429] = 'Too Many Requests'
HTTP_CODES[431] = 'Request Header Fields Too Large'
HTTP_CODES[451] = 'Unavailable For Legal Reasons'
HTTP_CODES[511] = 'Network Authentication Required'
_HTTP_STATUS_LINES = dict(((k, '%d %s' % (k, v)) for (k, v) in HTTP_CODES.items()))
ERROR_PAGE_TEMPLATE = '\n%%try:\n    %%from %s import DEBUG, request\n    <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">\n    <html>\n        <head>\n            <title>Error: {{e.status}}</title>\n            <style type="text/css">\n              html {background-color: #eee; font-family: sans-serif;}\n              body {background-color: #fff; border: 1px solid #ddd;\n                    padding: 15px; margin: 15px;}\n              pre {background-color: #eee; border: 1px solid #ddd; padding: 5px;}\n            </style>\n        </head>\n        <body>\n            <h1>Error: {{e.status}}</h1>\n            <p>Sorry, the requested URL <tt>{{repr(request.url)}}</tt>\n               caused an error:</p>\n            <pre>{{e.body}}</pre>\n            %%if DEBUG and e.exception:\n              <h2>Exception:</h2>\n              %%try:\n                %%exc = repr(e.exception)\n              %%except:\n                %%exc = \'<unprintable %%s object>\' %% type(e.exception).__name__\n              %%end\n              <pre>{{exc}}</pre>\n            %%end\n            %%if DEBUG and e.traceback:\n              <h2>Traceback:</h2>\n              <pre>{{e.traceback}}</pre>\n            %%end\n        </body>\n    </html>\n%%except ImportError:\n    <b>ImportError:</b> Could not generate the error page. Please add bottle to\n    the import path.\n%%end\n' % __name__
request = LocalRequest()
response = LocalResponse()
local = threading.local()
apps = app = default_app = AppStack()
ext = _ImportRedirect('bottle.ext' if __name__ == '__main__' else __name__ + '.ext', 'bottle_%s').module

def _main(argv):
    if False:
        print('Hello World!')
    (args, parser) = _cli_parse(argv)

    def _cli_error(cli_msg):
        if False:
            return 10
        parser.print_help()
        _stderr('\nError: %s\n' % cli_msg)
        sys.exit(1)
    if args.version:
        print('Bottle %s' % __version__)
        sys.exit(0)
    if not args.app:
        _cli_error('No application entry point specified.')
    sys.path.insert(0, '.')
    sys.modules.setdefault('bottle', sys.modules['__main__'])
    (host, port) = (args.bind or 'localhost', 8080)
    if ':' in host and host.rfind(']') < host.rfind(':'):
        (host, port) = host.rsplit(':', 1)
    host = host.strip('[]')
    config = ConfigDict()
    for cfile in args.conf or []:
        try:
            if cfile.endswith('.json'):
                with open(cfile, 'rb') as fp:
                    config.load_dict(json_loads(fp.read()))
            else:
                config.load_config(cfile)
        except configparser.Error as parse_error:
            _cli_error(parse_error)
        except IOError:
            _cli_error('Unable to read config file %r' % cfile)
        except (UnicodeError, TypeError, ValueError) as error:
            _cli_error('Unable to parse config file %r: %s' % (cfile, error))
    for cval in args.param or []:
        if '=' in cval:
            config.update((cval.split('=', 1),))
        else:
            config[cval] = True
    run(args.app, host=host, port=int(port), server=args.server, reloader=args.reload, plugins=args.plugin, debug=args.debug, config=config)
if __name__ == '__main__':
    _main(sys.argv)