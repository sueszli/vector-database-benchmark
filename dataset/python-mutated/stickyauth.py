from typing import Optional
from mitmproxy import ctx
from mitmproxy import exceptions
from mitmproxy import flowfilter

class StickyAuth:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.flt = None
        self.hosts = {}

    def load(self, loader):
        if False:
            i = 10
            return i + 15
        loader.add_option('stickyauth', Optional[str], None, 'Set sticky auth filter. Matched against requests.')

    def configure(self, updated):
        if False:
            while True:
                i = 10
        if 'stickyauth' in updated:
            if ctx.options.stickyauth:
                try:
                    self.flt = flowfilter.parse(ctx.options.stickyauth)
                except ValueError as e:
                    raise exceptions.OptionsError(str(e)) from e
            else:
                self.flt = None

    def request(self, flow):
        if False:
            print('Hello World!')
        if self.flt:
            host = flow.request.host
            if 'authorization' in flow.request.headers:
                self.hosts[host] = flow.request.headers['authorization']
            elif flowfilter.match(self.flt, flow):
                if host in self.hosts:
                    flow.request.headers['authorization'] = self.hosts[host]