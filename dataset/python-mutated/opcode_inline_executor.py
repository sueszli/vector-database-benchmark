from __future__ import annotations
import contextlib
import inspect
import re
from typing import TYPE_CHECKING
from ...profiler import event_register
from ...utils import BreakGraphError, log
from ..instruction_utils import Instruction
from .guard import StringifyExpression, union_free_vars
from .opcode_executor import OpcodeExecutorBase, Stop
from .tracker import ConstTracker, DanglingTracker, DummyTracker, Tracker
from .variables import CellVariable, FunctionGlobalVariable, IterVariable, SequenceIterVariable, VariableBase
if TYPE_CHECKING:
    from .pycode_generator import PyCodeGen
    from .variables import FunctionVariable

class FunctionGlobalTracker(Tracker):
    """
    A tracker class that represents a function global variable.

    Args:
        fn: FunctionVariable object.
        name: The name of the global variable.

    """

    def __init__(self, fn: FunctionVariable, name: str):
        if False:
            print('Hello World!')
        super().__init__([fn])
        self.fn = fn
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen):
        if False:
            i = 10
            return i + 15
        '\n        Generate bytecode instructions in order to put the variables at the top of the stack.\n\n        Args:\n            codegen: The PyCodeGen object used to generate bytecode.\n\n        '
        self.fn.tracker.gen_instructions(codegen)
        codegen.gen_load_attr('__globals__')
        codegen.gen_load_const(self.name)
        codegen.gen_subscribe()

    def trace_value_from_frame(self) -> StringifyExpression:
        if False:
            i = 10
            return i + 15
        '\n        Trace the value of the function global variable from the frame.\n\n        Returns:\n            StringifyExpression: The traced value of the function global variable.\n\n        '
        fn_tracer = self.fn.tracker.trace_value_from_frame()
        return StringifyExpression(f"{{}}.__globals__['{self.name}']", [fn_tracer], union_free_vars(fn_tracer.free_vars))

    def __repr__(self) -> str:
        if False:
            return 10
        return f'FunctionGlobalTracker(fn={self.fn}, name={self.name})'

class FunctionClosureTracker(Tracker):
    """
    A tracker class that represents a function closure variable.

    Args:
        fn: The FunctionVariable object.
        idx: The index of the closure variable.

    """

    def __init__(self, fn: FunctionVariable, idx: int):
        if False:
            print('Hello World!')
        super().__init__([fn])
        self.fn = fn
        self.idx = idx

    def gen_instructions(self, codegen: PyCodeGen):
        if False:
            return 10
        '\n        Generate bytecode instructions to trace the value of the function closure variable.\n\n        Args:\n            codegen: The PyCodeGen object used to generate bytecode.\n\n        '
        self.fn.tracker.gen_instructions(codegen)
        codegen.gen_load_attr('__closure__')
        codegen.gen_load_const(self.idx)
        codegen.gen_subscribe()
        codegen.gen_load_attr('cell_contents')

    def trace_value_from_frame(self):
        if False:
            return 10
        '\n        Trace the value of the function closure variable from the frame.\n\n        Returns:\n            The traced value of the function closure variable.\n\n        '
        fn_tracer = self.fn.tracker.trace_value_from_frame()
        return StringifyExpression(f'{{}}.__closure__[{self.idx}].cell_contents', [fn_tracer], union_free_vars(fn_tracer.free_vars))

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'FunctionClosureTracker(fn={self.fn}, idx={self.idx})'

@contextlib.contextmanager
def signature_clear_guard(fn, name):
    if False:
        i = 10
        return i + 15
    if not hasattr(fn, name):
        yield
    else:
        saved_attr = getattr(fn, name)
        delattr(fn, name)
        yield
        setattr(fn, name, saved_attr)

class OpcodeInlineExecutor(OpcodeExecutorBase):
    """
    A class that represents an executor for inlined opcode operations.

    Args:
        fn_variable: The function variable.

    """

    def __init__(self, fn_variable: FunctionVariable, *args, **kwargs):
        if False:
            return 10
        self._fn_var = fn_variable
        self.return_value: VariableBase | None = None
        self._fn_value = fn_variable.value
        super().__init__(fn_variable.get_code(), fn_variable.graph)
        self._name = 'Inline'
        self._prepare_locals(*args, **kwargs)
        self._prepare_closure()

    def _handle_comps(self):
        if False:
            print('Hello World!')
        is_comp = any((x in self._fn_value.__name__ for x in ['<listcomp>', '<dictcomp>', '<genexpr>']))
        if not is_comp:
            return
        pattern = 'implicit\\d+'
        for name in list(self._locals.keys()):
            if re.match(pattern, name):
                self._locals[name.replace('implicit', '.')] = self._locals[name]

    def _prepare_locals(self, *args, **kwargs):
        if False:
            return 10
        '\n        Prepare local variables for execution by adding them to the locals dictionary.\n\n        '
        from .variables import VariableBase, VariableFactory
        with signature_clear_guard(self._fn_value, '__signature__'), signature_clear_guard(self._fn_value, '__wrapped__'):
            sig = inspect.signature(self._fn_value)
            bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for (name, value) in bound_args.arguments.items():
            assert name in sig.parameters
            if sig.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL:
                tracker = DummyTracker(value)
            elif sig.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
                tracker = DummyTracker(list(value.values()))
            elif not isinstance(value, VariableBase):
                tracker = ConstTracker(value)
            else:
                tracker = value.tracker
            value = VariableFactory.from_value(value, self._graph, tracker)
            self._locals[name] = value
        self._handle_comps()
        log(5, f'[INLINE CALL] {self._code.co_name} with locals: ', self._locals)

    def _prepare_closure(self):
        if False:
            while True:
                i = 10
        '\n        Prepare closure variables for execution by adding them to the closure list.\n\n        '
        from .variables import VariableFactory
        closure = self._fn_var.get_py_value().__closure__
        for name in self._code.co_cellvars + self._code.co_freevars:
            self._cells[name] = CellVariable()
            if name in self._locals:
                self._cells[name].set_value(self._locals[name])
        if closure is None:
            return
        assert len(closure) == len(self._code.co_freevars)
        for (idx, (name, cell)) in enumerate(zip(self._code.co_freevars, closure)):
            value = cell.cell_contents
            value = VariableFactory.from_value(value, self._graph, FunctionClosureTracker(self._fn_var, idx))
            if not isinstance(value, CellVariable):
                value = CellVariable(value)
            self._cells[name] = value

    @event_register('OpcodeInlineExecutor: _prepare_virtual_env', event_level=2)
    def _prepare_virtual_env(self):
        if False:
            while True:
                i = 10
        '\n        Prepare the virtual environment for execution by adding variables from globals, builtins, and constants.\n\n        '
        from .variables import VariableFactory
        self._globals = FunctionGlobalVariable(self._fn_var, self._fn_value.__globals__, self._graph, DanglingTracker())
        self._builtins = self._graph._builtins
        for value in self._code.co_consts:
            self._co_consts.append(VariableFactory.from_value(value, self._graph, ConstTracker(value)))

    def inline_call(self) -> VariableBase:
        if False:
            return 10
        '\n        Execute the inline call of the function.\n        '
        self.run()
        assert self.return_value is not None
        return self.return_value

    def RETURN_VALUE(self, instr: Instruction):
        if False:
            return 10
        assert len(self.stack) == 1, f'Stack must have one element, but get {len(self.stack)} elements.'
        self.return_value = self.stack.pop()
        return Stop(state='Return')

    def _break_graph_in_jump(self, result, instr: Instruction):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper method to raise a BreakGraphError when breaking the graph in a jump operation.\n\n        Args:\n            result: The result of the operation.\n            instr (Instruction): The jump instruction.\n        '
        raise BreakGraphError('OpcodeInlineExecutor want call _break_graph_in_jump.')

    def _create_resume_fn(self, index: int, stack_size: int=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper method to create a resume function for the executor.\n\n        Args:\n            index (int): The index of the instruction to resume execution from.\n            stack_size (int, optional): The size of the stack. Defaults to 0.\n        '
        raise BreakGraphError('_create_resume_fn.')

    def FOR_ITER(self, instr: Instruction):
        if False:
            print('Hello World!')
        iterator = self.stack.top
        assert isinstance(iterator, IterVariable)
        self._graph.add_global_guarded_variable(iterator)
        if isinstance(iterator, SequenceIterVariable):
            try:
                self.stack.push(iterator.next())
            except StopIteration:
                self.stack.pop()
                assert isinstance(instr.jump_to, Instruction)
                self._lasti = self.indexof(instr.jump_to)
        else:
            self._graph.remove_global_guarded_variable(iterator)
            raise BreakGraphError(f'Found {iterator.__class__.__name__} as iterator.')