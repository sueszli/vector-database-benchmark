from typing import Optional
from mitmproxy import ctx
from mitmproxy import exceptions
from mitmproxy import flow
from mitmproxy import flowfilter

class Intercept:
    filt: flowfilter.TFilter | None = None

    def load(self, loader):
        if False:
            for i in range(10):
                print('nop')
        loader.add_option('intercept_active', bool, False, 'Intercept toggle')
        loader.add_option('intercept', Optional[str], None, 'Intercept filter expression.')

    def configure(self, updated):
        if False:
            while True:
                i = 10
        if 'intercept' in updated:
            if ctx.options.intercept:
                try:
                    self.filt = flowfilter.parse(ctx.options.intercept)
                except ValueError as e:
                    raise exceptions.OptionsError(str(e)) from e
                ctx.options.intercept_active = True
            else:
                self.filt = None
                ctx.options.intercept_active = False

    def should_intercept(self, f: flow.Flow) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(ctx.options.intercept_active and self.filt and self.filt(f) and (not f.is_replay))

    def process_flow(self, f: flow.Flow) -> None:
        if False:
            print('Hello World!')
        if self.should_intercept(f):
            f.intercept()

    def request(self, f):
        if False:
            return 10
        self.process_flow(f)

    def response(self, f):
        if False:
            print('Hello World!')
        self.process_flow(f)

    def tcp_message(self, f):
        if False:
            for i in range(10):
                print('nop')
        self.process_flow(f)

    def udp_message(self, f):
        if False:
            print('Hello World!')
        self.process_flow(f)

    def dns_request(self, f):
        if False:
            return 10
        self.process_flow(f)

    def dns_response(self, f):
        if False:
            for i in range(10):
                print('nop')
        self.process_flow(f)