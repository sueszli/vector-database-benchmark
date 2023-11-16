from viztracer import VizCounter, VizTracer
from .base_tmpl import BaseTmpl

class Hello(VizCounter):

    def __init__(self, tracer, name):
        if False:
            print('Hello World!')
        super().__init__(tracer, name, trigger_on_change=False)

class TestCounterClass(BaseTmpl):

    def test_basic(self):
        if False:
            print('Hello World!')
        tracer = VizTracer(verbose=0)
        tracer.start()
        counter = VizCounter(tracer, 'name')
        counter.a = 1
        counter.b = 2
        tracer.stop()
        entries = tracer.parse()
        self.assertEqual(entries, 2)

    def test_exception(self):
        if False:
            i = 10
            return i + 15
        tracer = VizTracer(verbose=0)
        tracer.start()
        counter = VizCounter(tracer, 'name')
        with self.assertRaises(Exception) as _:
            counter.a = ''
        with self.assertRaises(Exception) as _:
            counter.b = {}
        with self.assertRaises(Exception) as _:
            counter.c = []
        tracer.stop()
        tracer.clear()

    def test_inherit(self):
        if False:
            i = 10
            return i + 15
        tracer = VizTracer(verbose=0)
        tracer.start()
        a = Hello(tracer, 'name')
        a.b = 1
        a.c = 2
        a.d = 3
        a.log()
        tracer.stop()
        entries = tracer.parse()
        tracer.save()
        self.assertEqual(entries, 2)

    def test_notracer(self):
        if False:
            for i in range(10):
                print('nop')
        counter = VizCounter(None, 'name')
        counter.a = 1
        counter.b = 2
        a = Hello(None, 'name')
        a.b = 1
        a.c = 2
        a.d = 3
        a.log()