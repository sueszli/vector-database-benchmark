import collections
from http import cookiejar
from typing import Optional
from mitmproxy import ctx
from mitmproxy import exceptions
from mitmproxy import flowfilter
from mitmproxy import http
from mitmproxy.net.http import cookies
TOrigin = tuple[str, int, str]

def ckey(attrs: dict[str, str], f: http.HTTPFlow) -> TOrigin:
    if False:
        i = 10
        return i + 15
    '\n    Returns a (domain, port, path) tuple.\n    '
    domain = f.request.host
    path = '/'
    if 'domain' in attrs:
        domain = attrs['domain']
    if 'path' in attrs:
        path = attrs['path']
    return (domain, f.request.port, path)

def domain_match(a: str, b: str) -> bool:
    if False:
        print('Hello World!')
    if cookiejar.domain_match(a, b):
        return True
    elif cookiejar.domain_match(a, b.strip('.')):
        return True
    return False

class StickyCookie:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.jar: collections.defaultdict[TOrigin, dict[str, str]] = collections.defaultdict(dict)
        self.flt: flowfilter.TFilter | None = None

    def load(self, loader):
        if False:
            return 10
        loader.add_option('stickycookie', Optional[str], None, 'Set sticky cookie filter. Matched against requests.')

    def configure(self, updated):
        if False:
            for i in range(10):
                print('nop')
        if 'stickycookie' in updated:
            if ctx.options.stickycookie:
                try:
                    self.flt = flowfilter.parse(ctx.options.stickycookie)
                except ValueError as e:
                    raise exceptions.OptionsError(str(e)) from e
            else:
                self.flt = None

    def response(self, flow: http.HTTPFlow):
        if False:
            print('Hello World!')
        assert flow.response
        if self.flt:
            for (name, (value, attrs)) in flow.response.cookies.items(multi=True):
                dom_port_path = ckey(attrs, flow)
                if domain_match(flow.request.host, dom_port_path[0]):
                    if cookies.is_expired(attrs):
                        self.jar[dom_port_path].pop(name, None)
                        if not self.jar[dom_port_path]:
                            self.jar.pop(dom_port_path, None)
                    else:
                        self.jar[dom_port_path][name] = value

    def request(self, flow: http.HTTPFlow):
        if False:
            for i in range(10):
                print('nop')
        if self.flt:
            cookie_list: list[tuple[str, str]] = []
            if flowfilter.match(self.flt, flow):
                for ((domain, port, path), c) in self.jar.items():
                    match = [domain_match(flow.request.host, domain), flow.request.port == port, flow.request.path.startswith(path)]
                    if all(match):
                        cookie_list.extend(c.items())
            if cookie_list:
                flow.metadata['stickycookie'] = True
                flow.request.headers['cookie'] = cookies.format_cookie_header(cookie_list)