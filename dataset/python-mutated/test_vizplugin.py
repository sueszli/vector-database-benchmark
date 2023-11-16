import io
from contextlib import redirect_stdout
from viztracer import VizTracer
from viztracer.vizplugin import VizPluginBase, VizPluginError
from .cmdline_tmpl import CmdlineTmpl

class MyPlugin(VizPluginBase):

    def __init__(self, terminate_well=True):
        if False:
            print('Hello World!')
        self.event_counter = 0
        self.handler_triggered = False
        self.terminate_well = terminate_well

    def support_version(self):
        if False:
            return 10
        return '0.10.5'

    def message(self, m_type, payload):
        if False:
            return 10

        def f(data):
            if False:
                print('Hello World!')
            self.handler_triggered = True
        self.event_counter += 1
        if m_type == 'event' and payload['when'] == 'pre-save':
            return {'action': 'handle_data', 'handler': f}
        if m_type == 'command':
            if payload['cmd_type'] == 'terminate':
                return {'success': self.terminate_well}
        return {}

class MyPluginIncomplete(VizPluginBase):
    pass

class MyPluginFuture(VizPluginBase):

    def support_version(self):
        if False:
            print('Hello World!')
        return '9999.999.99'

class TestVizPlugin(CmdlineTmpl):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        pl = MyPlugin()
        tracer = VizTracer(plugins=[pl], verbose=0)
        tracer.start()
        tracer.stop()
        tracer.save()
        self.assertEqual(pl.event_counter, 4)
        self.assertEqual(pl.handler_triggered, True)

    def test_invalid(self):
        if False:
            return 10
        invalid_pl = []
        with self.assertRaises(TypeError):
            _ = VizTracer(plugins=[invalid_pl])
        with self.assertRaises(NotImplementedError):
            _ = VizTracer(plugins=[MyPluginIncomplete()])

    def test_terminate(self):
        if False:
            return 10
        pl = MyPlugin()
        with VizTracer(plugins=[pl], verbose=0):
            _ = []
        pl = MyPlugin(terminate_well=False)
        with self.assertRaises(VizPluginError):
            with VizTracer(plugins=[pl], verbose=0):
                _ = []

    def test_version(self):
        if False:
            print('Hello World!')
        pl = MyPluginFuture()
        s = io.StringIO()
        with redirect_stdout(s):
            with VizTracer(plugins=[pl], verbose=0):
                _ = []
        output = s.getvalue()
        self.assertEqual(output.count('support version is higher'), 1)

    def test_cmdline(self):
        if False:
            return 10
        self.template(['viztracer', '--plugin', 'tests.modules.dummy_vizplugin', '--', 'cmdline_test.py'])
        self.template(['viztracer', '--plugin', 'tests.modules.dummy_vizplugin_wrong', '--', 'cmdline_test.py'], success=False)
        self.template(['viztracer', '--plugin', 'tests.modules', '--', 'cmdline_test.py'], success=False)
        self.template(['viztracer', '--plugin', 'invalid', '--', 'cmdline_test.py'], success=False)