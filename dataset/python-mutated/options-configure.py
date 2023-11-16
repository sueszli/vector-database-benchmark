"""React to configuration changes."""
from typing import Optional
from mitmproxy import ctx
from mitmproxy import exceptions

class AddHeader:

    def load(self, loader):
        if False:
            i = 10
            return i + 15
        loader.add_option(name='addheader', typespec=Optional[int], default=None, help='Add a header to responses')

    def configure(self, updates):
        if False:
            while True:
                i = 10
        if 'addheader' in updates:
            if ctx.options.addheader is not None and ctx.options.addheader > 100:
                raise exceptions.OptionsError('addheader must be <= 100')

    def response(self, flow):
        if False:
            while True:
                i = 10
        if ctx.options.addheader is not None:
            flow.response.headers['addheader'] = str(ctx.options.addheader)
addons = [AddHeader()]