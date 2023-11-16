from mitmproxy import ctx

class AntiComp:

    def load(self, loader):
        if False:
            return 10
        loader.add_option('anticomp', bool, False, 'Try to convince servers to send us un-compressed data.')

    def request(self, flow):
        if False:
            print('Hello World!')
        if ctx.options.anticomp:
            flow.request.anticomp()