from mitmproxy import ctx

class AntiCache:

    def load(self, loader):
        if False:
            return 10
        loader.add_option('anticache', bool, False, '\n            Strip out request headers that might cause the server to return\n            304-not-modified.\n            ')

    def request(self, flow):
        if False:
            return 10
        if ctx.options.anticache:
            flow.request.anticache()