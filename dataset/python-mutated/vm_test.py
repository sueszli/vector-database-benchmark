"""Tests for vm.py.

To create test cases, you can disassemble source code with the help of the dis
module. For example, in Python 3.7, this snippet:

  import dis
  import opcode
  def f(): return None
  bytecode = dis.Bytecode(f)
  for x in bytecode.codeobj.co_code:
    print(f'{x} ({opcode.opname[x]})')

prints:

  100 (LOAD_CONST)
  0 (<0>)
  83 (RETURN_VALUE)
  0 (<0>)
"""
import textwrap
from pytype import context
from pytype import vm
from pytype.tests import test_base
from pytype.tests import test_utils

class TraceVM(vm.VirtualMachine):
    """Special VM that remembers which instructions it executed."""

    def __init__(self, ctx):
        if False:
            print('Hello World!')
        super().__init__(ctx)
        self.instructions_executed = set()
        self._call_trace = set()
        self._functions = set()
        self._classes = set()
        self._unknowns = []

    def run_instruction(self, op, state):
        if False:
            while True:
                i = 10
        self.instructions_executed.add(op.index)
        return super().run_instruction(op, state)

class VmTestBase(test_base.BaseTest, test_utils.MakeCodeMixin):
    """Base for VM tests."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.ctx = self.make_context()

    def make_context(self):
        if False:
            return 10
        return context.Context(options=self.options, loader=self.loader)

class TraceVmTestBase(VmTestBase):
    """Base for VM tests with a tracer vm."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.ctx.vm = TraceVM(self.ctx)

class TraceTest(TraceVmTestBase):
    """Tests for opcode tracing in the VM."""

    def test_empty_data(self):
        if False:
            while True:
                i = 10
        'Test that we can trace values without data.'
        op = test_utils.FakeOpcode('foo.py', 123, 'foo')
        self.ctx.vm.trace_opcode(op, 'x', 42)
        self.assertEqual(self.ctx.vm.opcode_traces, [(op, 'x', (None,))])

    def test_const(self):
        if False:
            return 10
        src = textwrap.dedent('\n      x = 1  # line 1\n      y = x  # line 2\n    ').lstrip()
        self.ctx.vm.run_program(src, '', maximum_depth=10)
        expected = [('LOAD_CONST', 1, 1), ('STORE_NAME', 1, 'x'), ('LOAD_NAME', 2, 'x'), ('STORE_NAME', 2, 'y'), ('LOAD_CONST', 2, None)]
        actual = [(op.name, op.line, symbol) for (op, symbol, _) in self.ctx.vm.opcode_traces]
        self.assertEqual(actual, expected)

class AnnotationsTest(VmTestBase):
    """Tests for recording annotations."""

    def test_record_local_ops(self):
        if False:
            for i in range(10):
                print('nop')
        self.ctx.vm.run_program('v: int = None', '', maximum_depth=10)
        self.assertEqual(self.ctx.vm.local_ops, {'<module>': [vm.LocalOp(name='v', op=vm.LocalOp.Op.ASSIGN), vm.LocalOp(name='v', op=vm.LocalOp.Op.ANNOTATE)]})
if __name__ == '__main__':
    test_base.main()