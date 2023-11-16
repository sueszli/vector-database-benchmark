import unittest
from manticore.core.smtlib import ConstraintSet
from manticore.core.workspace import *
from manticore.native.state import State
from manticore.platforms import linux
from manticore.utils.event import Eventful

class FakeMemory:

    def __init__(self):
        if False:
            print('Hello World!')
        self._constraints = None

    @property
    def constraints(self):
        if False:
            i = 10
            return i + 15
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if False:
            for i in range(10):
                print('nop')
        self._constraints = constraints

class FakeCpu(Eventful):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._memory = FakeMemory()

    @property
    def memory(self):
        if False:
            i = 10
            return i + 15
        return self._memory

class FakePlatform:

    def __init__(self):
        if False:
            print('Hello World!')
        self._constraints = None
        self.procs = [FakeCpu()]

    @property
    def current(self):
        if False:
            i = 10
            return i + 15
        return self.procs[0]

    @property
    def constraints(self):
        if False:
            i = 10
            return i + 15
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if False:
            i = 10
            return i + 15
        self._constraints = constraints
        for proc in self.procs:
            proc.memory.constraints = constraints

class StateTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        if False:
            return 10
        dirname = os.path.dirname(__file__)
        l = linux.Linux(os.path.join(dirname, 'binaries', 'basic_linux_amd64'))
        self.state = State(ConstraintSet(), l)

    def test_workspace_save_load(self):
        if False:
            print('Hello World!')
        self.state.constraints.add(True)
        workspace = Workspace('mem:')
        id_ = workspace.save_state(self.state)
        state = workspace.load_state(id_)
        for (left, right) in zip(sorted(self.state.mem._maps), sorted(state.mem._maps)):
            self.assertEqual(left.start, right.start)
            self.assertEqual(left.end, right.end)
            self.assertEqual(left.name, right.name)
        self.assertEqual(str(state.constraints), str(self.state.constraints))

    def test_workspace_id_start_with_zero(self):
        if False:
            while True:
                i = 10
        workspace = Workspace('mem:')
        id_ = workspace.save_state(self.state)
        self.assertEqual(id_, 0)

    def test_output(self):
        if False:
            while True:
                i = 10
        out = ManticoreOutput('mem:')
        name = 'mytest'
        message = 'custom message'
        testcase = out.testcase(prefix=name)
        out.save_testcase(self.state, testcase, message)
        workspace = out._store._data
        for (entry, data) in workspace.items():
            if entry.startswith('.'):
                continue
            self.assertTrue(entry.startswith(name))
            if 'messages' in entry:
                self.assertTrue(message in data)
        keys = [x.split('.')[1] for x in workspace.keys()]
        for key in self.state.platform.generate_workspace_files():
            self.assertIn(key, keys)
        self.assertIn('smt', keys)
        self.assertIn('trace', keys)
        self.assertIn('messages', keys)
        self.assertIn('input', keys)