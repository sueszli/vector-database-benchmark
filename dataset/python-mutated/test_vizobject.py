from viztracer import VizObject, VizTracer
from .base_tmpl import BaseTmpl

class Hello(VizObject):

    def __init__(self, tracer):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(tracer, 'name', trigger_on_change=False)
        self.a = 1
        self.b = 'lol'

    @VizObject.triggerlog
    def change_val(self):
        if False:
            return 10
        self.a += 1
        self.b += 'a'

    @VizObject.triggerlog(when='both')
    def change_val2(self):
        if False:
            print('Hello World!')
        self.a += 2
        self.b += 'b'

class TestVizObject(BaseTmpl):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        tracer = VizTracer(verbose=0)
        tracer.start()
        a = VizObject(tracer, 'my variable')
        a.hello = 1
        a.hello = 2
        tracer.stop()
        entries = tracer.parse()
        self.assertEqual(entries, 3)

    def test_include(self):
        if False:
            print('Hello World!')
        tracer = VizTracer(verbose=0)
        tracer.start()
        a = VizObject(tracer, 'my variable', include_attributes=['b', 'c'])
        a.hello = 1
        a.b = 2
        a.c = 3
        a.lol = 4
        tracer.stop()
        entries = tracer.parse()
        self.assertEqual(entries, 3)

    def test_exclude(self):
        if False:
            print('Hello World!')
        tracer = VizTracer(verbose=0)
        tracer.start()
        a = VizObject(tracer, 'my variable', exclude_attributes=['b', 'c'])
        a.hello = 1
        a.b = 2
        a.c = 3
        a.lol = 4
        tracer.stop()
        entries = tracer.parse()
        self.assertEqual(entries, 3)

    def test_trigger_on_change(self):
        if False:
            i = 10
            return i + 15
        tracer = VizTracer(verbose=0)
        tracer.stop()
        tracer.cleanup()
        tracer.start()
        a = VizObject(tracer, 'my variable', trigger_on_change=False)
        a.hello = 1
        a.b = 2
        a.c = 3
        a.lol = 4
        a.log()
        tracer.stop()
        entries = tracer.parse()
        self.assertEqual(entries, 2)

    def test_config(self):
        if False:
            print('Hello World!')
        tracer = VizTracer(verbose=0)
        tracer.start()
        a = VizObject(tracer, 'my variable')
        a.config('trigger_on_change', False)
        a.hello = 1
        a.b = 2
        a.c = 3
        a.lol = 4
        a.log()
        tracer.stop()
        entries = tracer.parse()
        self.assertEqual(entries, 2)
        with self.assertRaises(ValueError):
            a.config('invalid', 'value')

    def test_decorator(self):
        if False:
            print('Hello World!')
        tracer = VizTracer(verbose=0)
        tracer.start()
        a = Hello(tracer)
        a.config('include_attributes', ['a', 'b'])
        a.change_val()
        a.change_val2()
        b = Hello(tracer)
        b.config('include_attributes', ['a', 'b'])
        b.change_val()
        b.change_val2()
        tracer.stop()
        entries = tracer.parse()
        self.assertEqual(entries, 10)
        with self.assertRaises(ValueError):

            @VizObject.triggerlog(when='invalid')
            def change_invalid():
                if False:
                    for i in range(10):
                        print('nop')
                pass
            change_invalid()

    def test_buffer_wrap(self):
        if False:
            return 10
        tracer = VizTracer(tracer_entries=10, verbose=0)
        tracer.start()
        a = VizObject(tracer, 'my variable')
        for i in range(15):
            a.hello = i
        tracer.stop()
        entries = tracer.parse()
        tracer.save()
        self.assertEqual(entries, 10)

    def test_notracer(self):
        if False:
            while True:
                i = 10
        a = VizObject(None, 'my variable')
        a.hello = 1
        a.hello = 2
        a = Hello(None)
        a.config('include_attributes', ['a', 'b'])
        a.change_val()
        a.change_val2()
        b = Hello(None)
        b.config('include_attributes', ['a', 'b'])
        b.change_val()
        b.change_val2()