from viztracer import VizTracer
from .base_tmpl import BaseTmpl

class TestVizEvent(BaseTmpl):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        tracer = VizTracer(verbose=0)
        tracer.start()
        with tracer.log_event('event'):
            a = []
            a.append(1)
        tracer.stop()
        tracer.parse()
        self.assertEventNumber(tracer.data, 2)