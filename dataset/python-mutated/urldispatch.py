import re
from zope.interface import implementer
from pyramid.exceptions import URLDecodeError
from pyramid.interfaces import IRoute, IRoutesMapper
from pyramid.traversal import PATH_SAFE, quote_path_segment, split_path_info
from pyramid.util import is_nonstr_iter, text_
_marker = object()

@implementer(IRoute)
class Route:

    def __init__(self, name, pattern, factory=None, predicates=(), pregenerator=None):
        if False:
            return 10
        self.pattern = pattern
        self.path = pattern
        (self.match, self.generate) = _compile_route(pattern)
        self.name = name
        self.factory = factory
        self.predicates = predicates
        self.pregenerator = pregenerator

@implementer(IRoutesMapper)
class RoutesMapper:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.routelist = []
        self.static_routes = []
        self.routes = {}

    def has_routes(self):
        if False:
            return 10
        return bool(self.routelist)

    def get_routes(self, include_static=False):
        if False:
            i = 10
            return i + 15
        if include_static is True:
            return self.routelist + self.static_routes
        return self.routelist

    def get_route(self, name):
        if False:
            print('Hello World!')
        return self.routes.get(name)

    def connect(self, name, pattern, factory=None, predicates=(), pregenerator=None, static=False):
        if False:
            for i in range(10):
                print('nop')
        if name in self.routes:
            oldroute = self.routes[name]
            if oldroute in self.routelist:
                self.routelist.remove(oldroute)
        route = Route(name, pattern, factory, predicates, pregenerator)
        if not static:
            self.routelist.append(route)
        else:
            self.static_routes.append(route)
        self.routes[name] = route
        return route

    def generate(self, name, kw):
        if False:
            return 10
        return self.routes[name].generate(kw)

    def __call__(self, request):
        if False:
            while True:
                i = 10
        try:
            path = request.path_info or '/'
        except KeyError:
            path = '/'
        except UnicodeDecodeError as e:
            raise URLDecodeError(e.encoding, e.object, e.start, e.end, e.reason)
        for route in self.routelist:
            match = route.match(path)
            if match is not None:
                preds = route.predicates
                info = {'match': match, 'route': route}
                if preds and (not all((p(info, request) for p in preds))):
                    continue
                return info
        return {'route': None, 'match': None}
old_route_re = re.compile('(\\:[_a-zA-Z]\\w*)')
star_at_end = re.compile('\\*(\\w*)$')
route_re = re.compile('(\\{[_a-zA-Z][^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\})')

def update_pattern(matchobj):
    if False:
        for i in range(10):
            print('nop')
    name = matchobj.group(0)
    return '{%s}' % name[1:]

def _compile_route(route):
    if False:
        while True:
            i = 10
    if route.__class__ is not str:
        try:
            route = text_(route, 'ascii')
        except UnicodeDecodeError:
            raise ValueError('The pattern value passed to add_route must be either a Unicode string or a plain string without any non-ASCII characters (you provided %r).' % route)
    if old_route_re.search(route) and (not route_re.search(route)):
        route = old_route_re.sub(update_pattern, route)
    if not route.startswith('/'):
        route = '/' + route
    remainder = None
    if star_at_end.search(route):
        (route, remainder) = route.rsplit('*', 1)
    pat = route_re.split(route)
    pat.reverse()
    rpat = []
    gen = []
    prefix = pat.pop()
    gen.append(quote_path_segment(prefix, safe='/').replace('%', '%%'))
    rpat.append(re.escape(prefix))
    while pat:
        name = pat.pop()
        name = name[1:-1]
        if ':' in name:
            (name, reg) = name.split(':', 1)
        else:
            reg = '[^/]+'
        gen.append('%%(%s)s' % name)
        name = f'(?P<{name}>{reg})'
        rpat.append(name)
        s = pat.pop()
        if s:
            rpat.append(re.escape(s))
            gen.append(quote_path_segment(s, safe='/').replace('%', '%%'))
    if remainder:
        rpat.append('(?P<%s>.*?)' % remainder)
        gen.append('%%(%s)s' % remainder)
    pattern = ''.join(rpat) + '$'
    match = re.compile(pattern).match

    def matcher(path):
        if False:
            for i in range(10):
                print('nop')
        m = match(path)
        if m is None:
            return None
        d = {}
        for (k, v) in m.groupdict().items():
            if k == remainder:
                d[k] = split_path_info(v)
            else:
                d[k] = v
        return d
    gen = ''.join(gen)

    def q(v):
        if False:
            print('Hello World!')
        return quote_path_segment(v, safe=PATH_SAFE)

    def generator(dict):
        if False:
            print('Hello World!')
        newdict = {}
        for (k, v) in dict.items():
            if v.__class__ is bytes:
                v = v.decode('utf-8')
            if k == remainder:
                if is_nonstr_iter(v):
                    v = '/'.join([q(x) for x in v])
                else:
                    if v.__class__ is not str:
                        v = str(v)
                    v = q(v)
            else:
                if v.__class__ is not str:
                    v = str(v)
                v = q(v)
            newdict[k] = v
        result = gen % newdict
        return result
    return (matcher, generator)