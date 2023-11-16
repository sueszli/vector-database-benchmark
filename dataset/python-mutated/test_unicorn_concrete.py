import unittest
import os
import io
import contextlib
from manticore.native import Manticore
from manticore.native.state import State
from manticore.core.plugin import Plugin

class ConcretePlugin(Plugin):

    def will_run_callback(self, ready_states):
        if False:
            i = 10
            return i + 15
        for state in ready_states:
            state.cpu.emulate_until(0)

class RegisterCapturePlugin(Plugin):

    def did_run_callback(self):
        if False:
            print('Hello World!')
        with self.manticore.locked_context('regs', dict) as context:
            for st in self.manticore.terminated_states:
                for reg in st.platform.current.canonical_registers:
                    context[reg] = getattr(st.platform.current, reg)

class ManticornTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        if False:
            while True:
                i = 10
        dirname = os.path.dirname(__file__)
        self.m = Manticore(os.path.join(dirname, 'binaries', 'arguments_linux_amd64'), argv=['argv', 'mc', 'argface'])
        self.concrete_instance = Manticore(os.path.join(dirname, 'binaries', 'arguments_linux_amd64'), argv=['argv', 'mc', 'argface'])
        self.concrete_instance.register_plugin(ConcretePlugin())
        self.concrete_instance.register_plugin(RegisterCapturePlugin())
        self.m.register_plugin(RegisterCapturePlugin())

    def test_register_comparison(self):
        if False:
            return 10
        self.m.run()
        self.concrete_instance.run()
        should_match = {'RAX', 'RCX', 'RDX', 'RBX', 'RSP', 'RBP', 'RSI', 'RDI', 'R8', 'R9', 'R10', 'R12', 'R13', 'R14', 'R15', 'RIP', 'CS', 'DS', 'ES', 'SS', 'FS', 'GS', 'AF', 'CF', 'DF', 'IF', 'OF', 'SF', 'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5', 'FP6', 'FP7', 'FPSW', 'FPCW'}
        concrete_regs = {}
        normal_regs = {}
        self.assertEqual(len(list(self.m.terminated_states)), len(list(self.concrete_instance.terminated_states)))
        self.assertGreater(len(list(self.m.terminated_states)), 0)
        for st in self.m.terminated_states:
            for reg in should_match:
                normal_regs[reg] = getattr(st.platform.current, reg)
        for st in self.concrete_instance.terminated_states:
            for reg in should_match:
                concrete_regs[reg] = getattr(st.platform.current, reg)
        concrete_regs_vals = {reg: val for (reg, val) in concrete_regs.items() if reg in should_match}
        normal_regs_vals = {reg: val for (reg, val) in normal_regs.items() if reg in should_match}
        self.maxDiff = None
        self.assertDictEqual(concrete_regs_vals, normal_regs_vals)

    def test_integration_basic_stdout(self):
        if False:
            print('Hello World!')
        self.m.run()
        self.concrete_instance.run()
        self.m.finalize()
        self.concrete_instance.finalize()
        with open(os.path.join(self.m.workspace, 'test_00000000.stdout'), 'r') as f:
            left = f.read().strip()
        with open(os.path.join(self.concrete_instance.workspace, 'test_00000000.stdout'), 'r') as f:
            right = f.read().strip()
        self.assertEqual(left, right)

class ResumeUnicornPlugin(Plugin):

    def will_run_callback(self, ready_states):
        if False:
            i = 10
            return i + 15
        for state in ready_states:
            state.cpu.emulate_until(UnicornResumeTest.MAIN)

class UnicornResumeTest(unittest.TestCase):
    _multiprocess_can_split_ = True
    MAIN = 4202880
    PRE_LOOP = 4203246
    POST_LOOP = 4203334
    DONE = 4203731
    FAIL = 4203644

    def hook_main(self, state: State):
        if False:
            return 10
        print('Reached main!!')

    def hook_pre_loop(self, state: State):
        if False:
            for i in range(10):
                print('nop')
        print('Resuming emulation')
        state.cpu.emulate_until(self.POST_LOOP)

    def hook_ret_good(self, state: State):
        if False:
            while True:
                i = 10
        print('We made it!')

    def hook_ret_fail(self, state: State):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(False, 'Target binary called `lose`!')

    def setUp(self):
        if False:
            return 10
        dirname = os.path.dirname(__file__)
        self.concrete_instance = Manticore(os.path.join(dirname, 'binaries', 'rusticorn'))
        self.concrete_instance.register_plugin(ResumeUnicornPlugin())
        self.concrete_instance.add_hook(self.MAIN, callback=self.hook_main)
        self.concrete_instance.add_hook(self.PRE_LOOP, callback=self.hook_pre_loop)
        self.concrete_instance.add_hook(self.DONE, callback=self.hook_ret_good)
        self.concrete_instance.add_hook(self.FAIL, callback=self.hook_ret_fail)

    def test_integration_resume(self):
        if False:
            return 10
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.concrete_instance.run()
            self.concrete_instance.finalize()
        output = f.getvalue()
        print(output)
        self.assertIn('Reached main!!', output)
        self.assertIn('Resuming emulation', output)
        self.assertIn('We made it!', output)
        path = self.concrete_instance.workspace + '/test_00000000.stdout'
        with open(path) as stdoutf:
            stdout = stdoutf.read()
        self.assertIn('If we were running under Python, that would have taken a really long time!', stdout)
        self.assertIn('You win!', stdout)
        self.assertIn('8031989549026', stdout)