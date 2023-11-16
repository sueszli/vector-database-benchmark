import collections
import contextlib
import copy
import dataclasses
import dis
import functools
import importlib
import inspect
import itertools
import linecache
import logging
import operator
import sys
import textwrap
import threading
import traceback
import types
import typing
import weakref
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Type
from unittest.mock import patch
import torch
import torch._logging
from torch._guards import Checkpointable, tracing, TracingContext
from . import allowed_functions, config, exc, logging as torchdynamo_logging, side_effects, skipfiles, variables
from .allowed_functions import is_allowed, is_builtin_constant, is_forbidden
from .bytecode_analysis import get_indexof, JUMP_OPNAMES, livevars_analysis, propagate_line_nums
from .bytecode_transformation import cleaned_instructions, create_call_function, create_instruction, create_jump_absolute, Instruction, is_generator, unique_id
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import current_scope_id
from .exc import ArgsMismatchError, BackendCompilerFailed, unimplemented, Unsupported
from .funcname_cache import get_funcname
from .guards import GuardBuilder, install_guard
from .output_graph import GraphCompileReason, OutputGraph, OutputGraphState
from .replay_record import DummyModule, ExecutionRecorder
from .resume_execution import ContinueExecutionCache, ReenterWith
from .source import AttrSource, GetItemSource, GlobalSource, GlobalWeakRefSource, LocalSource, Source
from .utils import counters, get_fake_value, get_instruction_source_311, graph_break_dup_warning_checker, istype, LazyString, proxy_args_kwargs
from .variables.base import _is_top_level_scope, is_side_effect_safe, MutableLocal, typestr, VariableTracker
from .variables.builder import VariableBuilder, wrap_fx_proxy
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable, EnumVariable
from .variables.ctx_manager import ContextWrappingVariable, GenericContextWrappingVariable, WithExitFunctionVariable
from .variables.dicts import ConstDictVariable, SetVariable
from .variables.functions import BaseUserFunctionVariable, NestedUserFunctionVariable, UserFunctionVariable, UserMethodVariable
from .variables.lists import BaseListVariable, ListIteratorVariable, ListVariable, SliceVariable, TupleVariable
from .variables.misc import ClosureVariable, GetAttrVariable, InlinedClosureVariable, NullVariable, PythonModuleVariable, UnknownVariable
from .variables.nn_module import NNModuleVariable
from .variables.tensor import supported_const_comparison_ops, supported_tensor_comparison_ops, SymNodeVariable, TensorVariable
from .variables.torch import TorchVariable
from .variables.user_defined import RemovableHandleVariable, UserDefinedClassVariable, UserDefinedObjectVariable, UserDefinedVariable
log = logging.getLogger(__name__)
graph_break_log = torch._logging.getArtifactLogger(__name__, 'graph_breaks')
trace_call_log = torch._logging.getArtifactLogger(__name__, 'trace_call')
trace_source_log = torch._logging.getArtifactLogger(__name__, 'trace_source')
tls = threading.local()

@dataclasses.dataclass
class SpeculationEntry:
    filename: str
    lineno: int
    instruction_pointer: int
    failed: bool = False
    reason: Optional[GraphCompileReason] = None

    def fail_and_restart_analysis(self):
        if False:
            i = 10
            return i + 15
        "\n        Start tracing of the current frame over again, and don't take this branch.\n        "
        self.failed = True
        raise exc.SpeculationRestartAnalysis()

@dataclasses.dataclass
class SpeculationLog:
    """
    SpeculationLog replaces the prior copy_graphstate/restore_graphstate
    checkpointing.  Rather than saving/restoring state, we restart the
    dynamo conversion process over from the beginning -- but when we
    hit the start of the speculation that failed, we instead generate
    a graph break.
    """
    entries: List[SpeculationEntry] = dataclasses.field(default_factory=list)
    index: int = 0

    def restart(self):
        if False:
            return 10
        self.index = 0

    def clear(self):
        if False:
            return 10
        self.entries.clear()
        self.index = 0

    def next(self, filename: str, lineno: int, instruction_pointer) -> SpeculationEntry:
        if False:
            for i in range(10):
                print('nop')
        '\n        Lookup or create a SpeculationEntry() that is shared across\n        RestartAnalysis calls.  Args are used only for debug checks.\n        '
        if len(self.entries) == self.index:
            self.entries.append(SpeculationEntry(filename, lineno, instruction_pointer))
        entry = self.entries[self.index]
        self.index += 1
        assert entry.instruction_pointer == instruction_pointer and entry.filename == filename and (entry.lineno == lineno), textwrap.dedent(f'\n            SpecuationLog diverged at {self.index} of {len(self.entries)}:\n            - Run1: {entry.filename}:{entry.lineno} (ip={entry.instruction_pointer})\n            - Run2: {filename}:{lineno} (ip={instruction_pointer})\n            Please submit a bug report.\n            ')
        return entry

@functools.lru_cache(None)
def _step_logger():
    if False:
        while True:
            i = 10
    return torchdynamo_logging.get_step_logger(log)

@dataclasses.dataclass
class BlockStackEntry:
    target: Instruction
    stack_index: Optional[int] = None
    with_context: Optional[ContextWrappingVariable] = None

    def can_restore(self):
        if False:
            print('Hello World!')
        return self.with_context is not None

    def resume_fn(self):
        if False:
            return 10
        assert self.stack_index is not None
        if self.with_context and self.with_context.target_values:
            return ReenterWith(self.stack_index, tuple(self.with_context.target_values))
        else:
            return ReenterWith(self.stack_index)

    def exit(self, tx):
        if False:
            while True:
                i = 10
        assert self.with_context is not None
        return self.with_context.exit(tx)

class InstructionTranslatorGraphState(NamedTuple):
    output: OutputGraphState
    symbolic_locals: Dict[str, VariableTracker]
    stack: List[VariableTracker]
    block_stack: List[BlockStackEntry]
    instruction_pointer: Optional[int]
    current_instruction: Instruction
    next_instruction: Optional[Instruction]
    lineno: int

    def diff(self, other: 'InstructionTranslatorGraphState') -> Optional[str]:
        if False:
            i = 10
            return i + 15
        for k in self._fields:
            if k == 'output':
                return self.output.diff(other.output, prefix=f'{k}.')
            sv = getattr(self, k)
            ov = getattr(other, k)
            if sv != ov:
                return f'{k} mismatch: {sv} != {ov}'
        return None

def stack_op(fn: typing.Callable[..., object]):
    if False:
        for i in range(10):
            print('nop')
    nargs = len(inspect.signature(fn).parameters)
    fn_var = BuiltinVariable(fn)

    @functools.wraps(fn)
    def impl(self: 'InstructionTranslatorBase', inst: Instruction):
        if False:
            return 10
        self.push(fn_var.call_function(self, self.popn(nargs), {}))
    return impl

def _detect_and_normalize_assert_statement(self: 'InstructionTranslatorBase', truth_fn: typing.Callable[[object], bool], push: bool):
    if False:
        for i in range(10):
            print('nop')
    if truth_fn is not operator.truth or push:
        return False
    assert isinstance(self.instruction_pointer, int)
    current_instruction_pointer = self.instruction_pointer
    inst = self.instructions[current_instruction_pointer]
    if sys.version_info < (3, 9):
        if inst.opname != 'LOAD_GLOBAL' or inst.argval != 'AssertionError':
            return False
    elif inst.opname != 'LOAD_ASSERTION_ERROR':
        return False
    current_instruction_pointer += 1
    error_msg = 'assertion error'
    inst = self.instructions[current_instruction_pointer]
    if inst.opname == 'LOAD_CONST':
        if not isinstance(inst.argval, str):
            return False
        error_msg = inst.argval
        current_instruction_pointer += 1
        inst = self.instructions[current_instruction_pointer]
        if inst.opname not in ('CALL_FUNCTION', 'PRECALL'):
            return False
        current_instruction_pointer += 1
        if inst.opname == 'PRECALL':
            current_instruction_pointer += 1
        inst = self.instructions[current_instruction_pointer]
    if inst.opname != 'RAISE_VARARGS':
        return False
    self.push(ConstantVariable.create(error_msg))
    return True

def generic_jump(truth_fn: typing.Callable[[object], bool], push: bool):
    if False:
        for i in range(10):
            print('nop')

    def inner(self: 'InstructionTranslatorBase', inst: Instruction):
        if False:
            print('Hello World!')
        value: VariableTracker = self.pop()
        if config.rewrite_assert_with_torch_assert and _detect_and_normalize_assert_statement(self, truth_fn, push):
            error_msg: VariableTracker = self.pop()
            if value.is_python_constant() and bool(value.as_python_constant()):
                self.jump(inst)
                return
            if isinstance(value, TensorVariable):
                self.output.create_proxy('call_function', torch._assert_async, *proxy_args_kwargs((value, error_msg), {}))
                self.jump(inst)
                return
            scalar_to_tensor_proxy = self.output.create_proxy('call_function', torch.scalar_tensor, *proxy_args_kwargs((value,), {}))
            scalar_to_tensor = wrap_fx_proxy(self, scalar_to_tensor_proxy, example_value=get_fake_value(scalar_to_tensor_proxy.node, self))
            self.output.create_proxy('call_function', torch._assert_async, *proxy_args_kwargs((scalar_to_tensor, error_msg), {}))
            self.jump(inst)
            return
        if value.is_python_constant():
            if truth_fn(value.as_python_constant()):
                push and self.push(value)
                self.jump(inst)
        elif isinstance(value, TensorVariable) and self.should_compile_partial_graph():
            if self.has_backedge():
                msg = f'Skipping frame because there is a graph break in a for/while loop\n{self.frame_summary()}'
                log.info(msg)
                raise exc.SkipFrame(msg)
            self.push(value)
            log.debug('generic_jump triggered compile')
            self.output.compile_subgraph(self, reason=GraphCompileReason(f'generic_jump {typestr(value)}', [self.frame_summary()]))
            self.pop()
            if_next = self.create_call_resume_at(self.next_instruction)
            push and self.push(value)
            if_jump = self.create_call_resume_at(inst.target)
            self.output.add_output_instructions([create_instruction(inst.opname, target=if_jump[0])] + if_next + if_jump)
        elif isinstance(value, NNModuleVariable):
            mod = self.output.get_submodule(value.module_key)
            if truth_fn(mod):
                push and self.push(value)
                self.jump(inst)
        elif isinstance(value, UserDefinedObjectVariable):
            x = value.var_getattr(self, '__bool__')
            if isinstance(x, GetAttrVariable):
                x = value.var_getattr(self, '__len__')
            if isinstance(x, UserMethodVariable):
                result = x.call_function(self, [], {})
                if isinstance(result, ConstantVariable) and isinstance(result.value, (bool, int)):
                    if truth_fn(result.value):
                        push and self.push(value)
                        self.jump(inst)
                else:
                    unimplemented('generic_jump on UserDefined with __bool__ returning non-constant')
            elif truth_fn(True):
                push and self.push(value)
                self.jump(inst)
        elif not isinstance(value, TensorVariable) and value.has_unpack_var_sequence(self):
            if truth_fn(len(value.unpack_var_sequence(self))):
                push and self.push(value)
                self.jump(inst)
        elif isinstance(value, SymNodeVariable):
            eval_result = value.evaluate_expr(self.output)
            if truth_fn(eval_result):
                push and self.push(value)
                self.jump(inst)
        else:
            raise exc.UserError(exc.UserErrorType.DYNAMIC_CONTROL_FLOW, 'Dynamic control flow is not supported at the moment. Please use functorch.experimental.control_flow.cond to explicitly capture the control flow.', case_name='cond_operands')
    return inner
explain = False

def break_graph_if_unsupported(*, push):
    if False:
        print('Hello World!')

    def decorator(inner_fn):
        if False:
            while True:
                i = 10

        @functools.wraps(inner_fn)
        def wrapper(self: 'InstructionTranslatorBase', inst: Instruction):
            if False:
                for i in range(10):
                    print('nop')
            speculation = self.speculate()
            if speculation.failed:
                assert speculation.reason is not None
                return handle_graph_break(self, inst, speculation.reason)
            try:
                TracingContext.set_current_loc(self.f_code.co_filename, self.lineno, self.f_code.co_name)
                return inner_fn(self, inst)
            except Unsupported as excp:
                if self.should_compile_partial_graph() and self.has_backedge():
                    msg = f'Skipping frame because there is a graph break in a for/while loop\n{self.frame_summary()}'
                    log.info(msg)
                    raise exc.SkipFrame(msg) from excp
                if self.generic_context_manager_depth > 0:
                    excp.remove_from_stats()
                    unimplemented('Graph break under GenericContextWrappingVariable')
                if isinstance(excp, exc.UncapturedHigherOrderOpError):
                    raise
                if not self.should_compile_partial_graph():
                    raise
                log.debug('break_graph_if_unsupported triggered compile', exc_info=True)
                user_stack = excp.real_stack
                user_stack_formatted = ''.join(traceback.format_list(user_stack))
                frame_loc = (user_stack[-1].filename, user_stack[-1].lineno)
                if graph_break_log.isEnabledFor(logging.DEBUG) and (not explain) and graph_break_dup_warning_checker.add(frame_loc):
                    graph_break_log.debug('Graph break: %s from user code at:\n%s', excp, user_stack_formatted)
                excp.remove_from_stats()
                excp.add_to_stats('graph_break')
                speculation.reason = GraphCompileReason(excp.msg, user_stack)
            speculation.fail_and_restart_analysis()

        def handle_graph_break(self: 'InstructionTranslatorBase', inst: Instruction, reason: GraphCompileReason):
            if False:
                print('Hello World!')
            self.output.compile_subgraph(self, reason=reason)
            cg = PyCodegen(self)
            cleanup: List[Instruction] = []
            for b in self.block_stack:
                assert b.with_context is not None
                self.output.add_output_instructions([*b.with_context.reconstruct(cg), *b.resume_fn().try_except(cg.code_options, cleanup)])
            if sys.version_info >= (3, 11) and inst.opname == 'CALL':
                kw_names = self.kw_names.as_python_constant() if self.kw_names is not None else ()
                if len(kw_names) > 0:
                    self.output.add_output_instructions([create_instruction('KW_NAMES', argval=kw_names)])
                self.output.add_output_instructions(create_call_function(inst.arg, False))
            else:
                assert inst.target is None
                inst_copy = copy.copy(inst)
                inst_copy.exn_tab_entry = None
                self.output.add_output_instructions([inst_copy])
            self.output.add_output_instructions(cleanup)
            if sys.version_info >= (3, 11) and inst.opname == 'CALL':
                stack_effect = dis.stack_effect(dis.opmap['PRECALL'], inst.arg) + dis.stack_effect(dis.opmap['CALL'], inst.arg)
            else:
                stack_effect = dis.stack_effect(inst.opcode, inst.arg)
            self.popn(push - stack_effect)
            for _ in range(push):
                self.push(UnknownVariable())
            self.output.add_output_instructions(self.create_call_resume_at(self.next_instruction))
        return wrapper
    return decorator

class InstructionTranslatorBase(Checkpointable[InstructionTranslatorGraphState]):
    output: OutputGraph
    symbolic_locals: Dict[str, VariableTracker]
    symbolic_globals: Dict[str, VariableTracker]
    stack: List[VariableTracker]
    instruction_pointer: Optional[int]
    current_instruction: Instruction
    next_instruction: Optional[Instruction]
    block_stack: List[BlockStackEntry]
    lineno: int
    kw_names: Optional[ConstantVariable]
    accept_prefix_inst: bool
    prefix_insts: List[Instruction]
    inline_depth: int
    inconsistent_side_effects: bool
    current_speculation: Optional[SpeculationEntry]
    random_calls: List[Tuple[Callable[..., object], Tuple[object, ...], Dict[str, object]]]

    def mark_inconsistent_side_effects(self):
        if False:
            while True:
                i = 10
        '\n        InstructionTranslator has encountered instructions which may cause\n        dynamo to see a different version of history from eager\n        See: https://github.com/pytorch/pytorch/issues/110765\n        '
        self.inconsistent_side_effects = True

    def has_backedge(self):
        if False:
            while True:
                i = 10
        cur_offset = self.current_instruction.offset
        assert self.instruction_pointer is not None
        for inst in self.instructions[self.instruction_pointer:]:
            if inst.opname in JUMP_OPNAMES:
                jump_offset = inst.argval
                if jump_offset < cur_offset:
                    return True
        return False

    def cell_and_freevars(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, '_cell_and_freevars'):
            self._cell_and_freevars = tuple(self.code_options['co_cellvars'] or []) + tuple(self.code_options['co_freevars'] or [])
        return self._cell_and_freevars

    def prune_dead_locals(self):
        if False:
            print('Hello World!')
        reads = livevars_analysis(self.instructions, self.current_instruction)
        reads = reads | set(self.cell_and_freevars())
        self.symbolic_locals = {k: v for (k, v) in self.symbolic_locals.items() if k in reads}
        self.output.side_effects.prune_dead_object_new(self)

    def call_function(self, fn: VariableTracker, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]):
        if False:
            while True:
                i = 10
        assert isinstance(fn, VariableTracker)
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        assert all((isinstance(x, VariableTracker) for x in itertools.chain(args, kwargs.values())))
        inner_fn = None
        if hasattr(fn, 'value'):
            inner_fn = fn.value
        if hasattr(fn, 'fn'):
            inner_fn = fn.fn
        if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
            raise AssertionError(f'Attempt to trace forbidden callable {inner_fn}')
        self.push(fn.call_function(self, args, kwargs))

    def update_locals_and_stack(self, oldvar: VariableTracker, newvar: VariableTracker):
        if False:
            for i in range(10):
                print('nop')

        def repl(v: VariableTracker):
            if False:
                while True:
                    i = 10
            if v.mutable_local is oldvar.mutable_local:
                return newvar
            return v
        recursive_parents = oldvar.parents_tracker.recursive_parents()

        def skip(v: VariableTracker):
            if False:
                i = 10
                return i + 15
            return v.parents_tracker not in recursive_parents
        cache: Dict[int, Tuple[object, object]] = dict()
        self.output.side_effects.apply(repl, cache, skip_fn=skip)
        self.stack = [VariableTracker.apply(repl, x, cache, skip_fn=skip) for x in self.stack]
        for (k, x) in self.symbolic_locals.items():
            self.symbolic_locals[k] = VariableTracker.apply(repl, x, cache, skip_fn=skip)

    def replace_all(self, oldvar: VariableTracker, newvar: VariableTracker):
        if False:
            print('Hello World!')
        if isinstance(oldvar.mutable_local, side_effects.MutableSideEffects):
            newvar = self.output.side_effects.mutation(oldvar, newvar)
        else:
            assert isinstance(oldvar.mutable_local, variables.base.MutableLocal)
            newvar = newvar.clone(mutable_local=variables.base.MutableLocal())
        self.update_locals_and_stack(oldvar, newvar)
        return newvar

    def inline_user_function_return(self, fn, args, kwargs):
        if False:
            print('Hello World!')
        '\n        A call to some user defined function by inlining it.\n        '
        return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

    def get_line_of_code_header(self, lineno=None):
        if False:
            for i in range(10):
                print('nop')
        if lineno is None:
            lineno = self.lineno
        inline_depth_str = f' (inline depth: {self.inline_depth})' if self.inline_depth > 0 else ''
        funcname = get_funcname(self.f_code.co_filename, lineno)
        funcname_str = '' if funcname is None else f' ({funcname})'
        return f'{self.f_code.co_filename}:{lineno} in {self.f_code.co_name}{funcname_str}{inline_depth_str}'

    def get_log_starts_line_log_str(self):
        if False:
            print('Hello World!')
        log_str = f'TRACE starts_line {self.get_line_of_code_header()}\n'
        line = linecache.getline(self.f_code.co_filename, self.lineno).rstrip()
        log_str += f'    {line}'
        return log_str

    def log_starts_line(self):
        if False:
            while True:
                i = 10
        trace_source_log.debug('%s', LazyString(self.get_log_starts_line_log_str))

    def step(self):
        if False:
            while True:
                i = 10
        'Process exactly one instruction, return False we should exit'
        assert isinstance(self.instruction_pointer, int)
        inst = self.instructions[self.instruction_pointer]
        self.current_instruction = inst
        self.instruction_pointer += 1
        if self.instruction_pointer < len(self.instructions):
            self.next_instruction = self.instructions[self.instruction_pointer]
        else:
            self.instruction_pointer = None
            self.next_instruction = None
        if inst.starts_line and self.lineno != inst.starts_line:
            self.lineno = inst.starts_line
            self.log_starts_line()
        if len(self.stack) == 0 and self.should_compile_partial_graph() and self.is_non_empty_graph():
            self.current_speculation = self.speculate()
            if self.current_speculation.failed:
                return self.step_graph_break(inst)
        log.debug('TRACE %s %s %s', inst.opname, inst.argval, self.stack)
        if sys.version_info >= (3, 11):
            entry = inst.exn_tab_entry
            if not (self.block_stack and entry and (self.block_stack[-1].target is entry.target)):
                if not entry:
                    if self.block_stack and inst.opname != 'NOP':
                        assert len(self.block_stack) == 1
                        self.block_stack.pop()
                elif len(self.block_stack) > 1 and self.block_stack[-2].target is entry.target:
                    self.block_stack.pop()
                else:
                    self.block_stack.append(BlockStackEntry(entry.target, len(self.stack)))
        try:
            if not hasattr(self, inst.opname):
                unimplemented(f'missing: {inst.opname}')
            TracingContext.set_current_loc(self.f_code.co_filename, self.lineno, self.f_code.co_name)
            getattr(self, inst.opname)(inst)
            return inst.opname != 'RETURN_VALUE'
        except Unsupported:
            if self.current_speculation is None:
                log.debug('empty checkpoint')
                raise
            log.debug('step triggered compile', exc_info=True)
        self.current_speculation.fail_and_restart_analysis()

    def step_graph_break(self, continue_inst):
        if False:
            for i in range(10):
                print('nop')
        assert not self.output.output_instructions
        assert self.current_speculation is not None
        self.output.compile_subgraph(self, partial_convert=True, reason=GraphCompileReason('step_unsupported', [self.frame_summary()]))
        self.output.add_output_instructions([create_jump_absolute(continue_inst)] + self.instructions)

    def run_ctx_mgr(self):
        if False:
            i = 10
            return i + 15
        return TracingContext.current_frame(None)

    def run(self):
        if False:
            while True:
                i = 10
        with self.run_ctx_mgr():
            try:
                self.output.push_tx(self)
                while self.instruction_pointer is not None and (not self.output.should_exit) and self.step():
                    pass
            except BackendCompilerFailed:
                raise
            except Exception as e:
                if config.replay_record_enabled:
                    e.exec_record = self.exec_recorder.get_record()
                raise
            finally:
                self.output.pop_tx()
                if isinstance(self, InstructionTranslator):
                    self.output.cleanup()

    def push(self, val: Optional[VariableTracker]):
        if False:
            print('Hello World!')
        assert val is None or isinstance(val, VariableTracker), f'push expects VariableTracker, got {typestr(val)}'
        self.stack.append(val)

    def push_many(self, vals: List[VariableTracker]):
        if False:
            for i in range(10):
                print('nop')
        for val in vals:
            self.push(val)

    def pop(self) -> VariableTracker:
        if False:
            while True:
                i = 10
        return self.stack.pop()

    def popn(self, n: int) -> List[VariableTracker]:
        if False:
            for i in range(10):
                print('nop')
        assert n >= 0
        return list(reversed([self.pop() for _ in range(n)]))

    def LOAD_FAST(self, inst):
        if False:
            while True:
                i = 10
        name = inst.argval
        if name in self.f_locals and config.replay_record_enabled:
            self.exec_recorder.add_local_var(name, self.f_locals[name])
        if name.startswith('.') and name not in self.symbolic_locals:
            name = name.replace('.', 'implicit')
        assert name not in self.cell_and_freevars()
        if name not in self.symbolic_locals:
            unimplemented('undefined LOAD_FAST')
        self.push(self.symbolic_locals[name])
        if name.startswith('___stack'):
            self.symbolic_locals.pop(name)

    def LOAD_DEREF(self, inst):
        if False:
            return 10
        assert inst.argval in self.cell_and_freevars()
        if inst.argval in self.f_locals and config.replay_record_enabled:
            self.exec_recorder.add_local_var(inst.argval, self.f_locals[inst.argval])
        if inst.argval not in self.symbolic_locals:
            unimplemented(f'undefined LOAD_DEREF {inst.argval}')
        self.push(self.symbolic_locals[inst.argval])

    def STORE_FAST(self, inst):
        if False:
            while True:
                i = 10
        loaded_vt = self.pop()
        name = inst.argval
        if _is_top_level_scope(current_scope_id()):
            loaded_vt = loaded_vt.rename(self, name)
        self.symbolic_locals[name] = loaded_vt

    def DELETE_FAST(self, inst):
        if False:
            while True:
                i = 10
        del self.symbolic_locals[inst.argval]
    STORE_DEREF = STORE_FAST

    def LOAD_CLOSURE(self, inst):
        if False:
            i = 10
            return i + 15
        self.push(ClosureVariable(name=inst.argval))

    def LOAD_CONST(self, inst):
        if False:
            i = 10
            return i + 15
        if isinstance(inst.argval, tuple) and (not inst.argval):
            self.push(TupleVariable([]))
        else:
            self.push(ConstantVariable.create(value=inst.argval))

    def get_global_source(self, name):
        if False:
            for i in range(10):
                print('nop')
        source: Source
        if self.output.global_scope is self.f_globals:
            source = GlobalSource(name)
        elif '__name__' in self.f_globals:
            source = AttrSource(self.import_source(self.f_globals['__name__']), name)
        else:
            mangled_name = f'___unnamed_scope_{id(self.f_globals)}'
            if mangled_name not in self.output.global_scope:
                self.output.install_global(mangled_name, self.f_globals)
            source = GetItemSource(GlobalSource(mangled_name), name)
        return source

    def LOAD_GLOBAL(self, inst):
        if False:
            for i in range(10):
                print('nop')
        if sys.version_info >= (3, 11):
            if inst.arg % 2:
                self.PUSH_NULL(inst)
        name = inst.argval
        if config.replay_record_enabled:
            if name in self.f_globals:
                self.exec_recorder.add_global_var(name, self.f_globals[name])
            else:
                assert name in self.f_builtins
                self.exec_recorder.builtins[name] = self.f_builtins[name]
        if inst.argval == 'AssertionError':
            unimplemented('assert with non-string message')
        if name in self.symbolic_globals:
            variable = self.output.side_effects[self.symbolic_globals[name]]
            self.push(self.output.side_effects.load_global(variable, name))
            return
        try:
            value = self.f_globals[name]
        except KeyError:
            return self.load_builtin(inst)
        source = self.get_global_source(name)
        self.push(VariableBuilder(self, source)(value))

    def STORE_GLOBAL(self, inst):
        if False:
            return 10
        value = self.pop()
        name = inst.argval
        source = self.get_global_source(name)
        if name not in self.symbolic_globals:
            self.symbolic_globals[name] = object()
        variable = self.output.side_effects.track_global_existing(source, self.symbolic_globals[name])
        if isinstance(value, RemovableHandleVariable):
            unimplemented('Storing handles in globals - NYI')
        self.output.side_effects.store_global(variable, name, value)

    def import_source(self, module_name):
        if False:
            while True:
                i = 10
        'Create an alias to a module for use in guards'
        if 'torch_package' in module_name:
            value = torch.package.package_importer._package_imported_modules[module_name]
            alias = module_name.replace('>', '_').replace('<', '_').replace('.', '_dot_')
        else:
            value = importlib.import_module(module_name)
            alias = f"__import_{module_name.replace('.', '_dot_')}"
        f_globals = self.output.global_scope
        assert alias not in f_globals or f_globals[alias] is value
        f_globals[alias] = value
        self.output.update_co_names(alias)
        return GlobalSource(alias)

    def resolve_name(self, name, package, level):
        if False:
            i = 10
            return i + 15
        '\n        Copied from the Cpython implementation of __import__\n        Resolve a relative module name to an absolute one.\n        https://github.com/python/cpython/blob/5a094f0255eea1db58fb2cf14c200971e64ec36e/Lib/importlib/_bootstrap.py#L902\n        '
        bits = package.rsplit('.', level - 1)
        if len(bits) < level:
            raise ImportError('attempted relative import beyond top-level package')
        base = bits[0]
        return f'{base}.{name}' if name else base

    def calc_package(self):
        if False:
            print('Hello World!')
        '\n        Copied from the Cpython implementation of __import__\n        https://github.com/python/cpython/blob/5a094f0255eea1db58fb2cf14c200971e64ec36e/Lib/importlib/_bootstrap.py#L1090\n        '
        package = self.f_globals.get('__package__')
        spec = self.f_globals.get('__spec__')
        if package is not None:
            if spec is not None and package != spec.parent:
                log.warning('__package__ != __spec__.parent (%r != %r)', package, spec.parent, stacklevel=3)
            return package
        elif spec is not None:
            return spec.parent
        else:
            log.warning("can't resolve package from __spec__ or __package__, falling back on __name__ and __path__", stacklevel=3)
            package = self.f_globals['__name__']
            if '__path__' not in self.f_globals:
                package = package.rpartition('.')[0]
        return package

    def IMPORT_NAME(self, inst):
        if False:
            while True:
                i = 10
        (level, fromlist) = self.popn(2)
        level = level.as_python_constant()
        fromlist = fromlist.as_python_constant()
        module_name = inst.argval
        recorded_name = f'{ExecutionRecorder.LOCAL_MOD_PREFIX}_{level}_{fromlist}_{module_name}'
        if recorded_name in self.f_globals:
            value = self.f_globals[recorded_name]
            source = GlobalSource(recorded_name)
        else:
            value = __import__(module_name, fromlist=fromlist, level=level, globals=self.f_globals)
            if level != 0:
                pkg = self.calc_package()
                module_name = self.resolve_name(module_name, pkg, level)
            if not fromlist:
                top_level_module_name = module_name.partition('.')[0]
                source = self.import_source(top_level_module_name)
            else:
                source = self.import_source(module_name)
        if config.replay_record_enabled:
            self.exec_recorder.add_local_mod(recorded_name, value)
        if is_allowed(value):
            self.push(TorchVariable(value, source=source))
        elif istype(value, (types.ModuleType, DummyModule)):
            self.push(PythonModuleVariable(value, source=source))
        else:
            unimplemented(f'IMPORT_NAME {typestr(value)}')

    def IMPORT_FROM(self, inst):
        if False:
            return 10
        self.DUP_TOP(inst)
        self.LOAD_ATTR(inst)

    def load_builtin(self, inst):
        if False:
            print('Hello World!')
        if inst.argval not in self.f_builtins:
            raise NameError(f"name '{inst.argval}' is not defined")
        val = self.f_builtins[inst.argval]
        if callable(val):
            self.push(VariableBuilder(self, GlobalSource(inst.argval))(val))
        else:
            assert is_builtin_constant(val)
            self.push(ConstantVariable.create(value=val))

    def jump(self, inst):
        if False:
            for i in range(10):
                print('nop')
        self.instruction_pointer = self.indexof[inst.target]
    JUMP_FORWARD = jump
    JUMP_ABSOLUTE = jump
    POP_JUMP_IF_FALSE = generic_jump(operator.not_, False)
    POP_JUMP_IF_TRUE = generic_jump(operator.truth, False)
    JUMP_IF_FALSE_OR_POP = generic_jump(operator.not_, True)
    JUMP_IF_TRUE_OR_POP = generic_jump(operator.truth, True)

    def SETUP_LOOP(self, inst):
        if False:
            print('Hello World!')
        self.block_stack.append(BlockStackEntry(inst.target))

    def SETUP_EXCEPT(self, inst):
        if False:
            while True:
                i = 10
        self.block_stack.append(BlockStackEntry(inst.target))

    def POP_BLOCK(self, inst):
        if False:
            while True:
                i = 10
        self.block_stack.pop()

    def SETUP_WITH(self, inst):
        if False:
            while True:
                i = 10
        self.setup_or_before_with(inst)

    def SETUP_FINALLY(self, inst):
        if False:
            i = 10
            return i + 15
        self.block_stack.append(BlockStackEntry(inst.target))

    def BEGIN_FINALLY(self, inst):
        if False:
            i = 10
            return i + 15
        self.push(None)

    def WITH_CLEANUP_START(self, inst):
        if False:
            print('Hello World!')
        (exit, exc) = self.popn(2)
        assert exc is None
        self.push(exc)
        self.push(exit.call_function(self, [ConstantVariable.create(None)] * 3, {}))

    def WITH_CLEANUP_FINISH(self, inst):
        if False:
            print('Hello World!')
        self.popn(2)
        self.push(None)

    def END_FINALLY(self, inst):
        if False:
            return 10
        tos = self.pop()
        assert tos is None

    def POP_FINALLY(self, inst):
        if False:
            print('Hello World!')
        preserve_tos = inst.argval
        if preserve_tos:
            tos = self.pop()
        assert self.pop() is None
        if preserve_tos:
            self.push(tos)

    def FOR_ITER(self, inst):
        if False:
            return 10
        it = self.pop().realize()
        if isinstance(it, (variables.ListIteratorVariable, variables.IteratorVariable)):
            try:
                (val, next_iter) = it.next_variables(self)
                self.push(next_iter)
                self.push(val)
            except StopIteration:
                self.jump(inst)
        else:
            unimplemented(f'FOR_ITER {typestr(it)}')

    def COMPARE_OP(self, inst):
        if False:
            for i in range(10):
                print('nop')
        (left, right) = self.popn(2)
        op = inst.argval
        supported_any = dict(itertools.chain(supported_tensor_comparison_ops.items(), supported_const_comparison_ops.items()))
        if isinstance(left, (TensorVariable, SymNodeVariable, NNModuleVariable, BaseListVariable, UserDefinedVariable, BaseUserFunctionVariable, ConstDictVariable)) and isinstance(right, ConstantVariable) and (right.value is None) and (op in supported_const_comparison_ops):
            self.push(ConstantVariable.create(supported_const_comparison_ops[op](object(), right.value)))
        elif left.is_python_constant() and right.is_python_constant() and (op in supported_any):
            self.push(ConstantVariable.create(supported_any[op](left.as_python_constant(), right.as_python_constant())))
        elif op in ('in', 'not in'):
            self.push(right.call_method(self, '__contains__', [left], {}))
            if op == 'not in':
                self.UNARY_NOT(inst)
        else:
            self.push(BuiltinVariable(supported_any[op]).call_function(self, [left, right], {}))

    def GET_ITER(self, inst):
        if False:
            for i in range(10):
                print('nop')
        self.call_function(BuiltinVariable(iter), [self.pop()], {})

    @break_graph_if_unsupported(push=1)
    def CALL_FUNCTION(self, inst):
        if False:
            return 10
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

    @break_graph_if_unsupported(push=1)
    def CALL_FUNCTION_EX(self, inst):
        if False:
            while True:
                i = 10
        kwargsvars: VariableTracker
        if inst.argval == 0:
            kwargsvars = ConstDictVariable({}, dict)
            argsvars = self.pop()
        elif inst.argval == 1:
            kwargsvars = self.pop()
            argsvars = self.pop()
        else:
            unimplemented('CALL_FUNCTION_EX')
        fn = self.pop()
        if sys.version_info >= (3, 11):
            null = self.pop()
            assert isinstance(null, NullVariable)
        if isinstance(fn, GetAttrVariable) and isinstance(fn.obj, TensorVariable) and (fn.name == 'view') and isinstance(argsvars, (ConstantVariable, TensorVariable)):
            argsvars = TupleVariable([argsvars])
        if not isinstance(argsvars, BaseListVariable) and argsvars.has_unpack_var_sequence(self):
            argsvars = TupleVariable(argsvars.unpack_var_sequence(self))
        if not isinstance(argsvars, BaseListVariable) or not isinstance(kwargsvars, ConstDictVariable):
            unimplemented(f'non-static call {typestr(argsvars)} {typestr(kwargsvars)}')
        self.call_function(fn, argsvars.items, kwargsvars.items)

    @break_graph_if_unsupported(push=1)
    def CALL_FUNCTION_KW(self, inst):
        if False:
            for i in range(10):
                print('nop')
        argnames = self.pop()
        args = self.popn(inst.argval)
        fn = self.pop()
        assert isinstance(argnames, TupleVariable) and argnames.is_python_constant()
        argnames = argnames.as_python_constant()
        (args, kwargs_list) = (args[:-len(argnames)], args[-len(argnames):])
        kwargs = dict(zip(argnames, kwargs_list))
        assert len(kwargs) == len(argnames)
        self.call_function(fn, args, kwargs)

    def LOAD_METHOD_SUPER(self, inst):
        if False:
            return 10
        self.CALL_FUNCTION(dataclasses.replace(inst, argval=2))
        arg = inst.argval[0]
        argval = self.code_options['co_names'][arg]
        if sys.version_info < (3, 11):
            self.LOAD_ATTR(dataclasses.replace(inst, argval=argval))
        else:
            self.LOAD_METHOD(dataclasses.replace(inst, argval=argval))

    def LOAD_ATTR_SUPER(self, inst):
        if False:
            print('Hello World!')
        self.CALL_FUNCTION(dataclasses.replace(inst, argval=2))
        arg = inst.argval[0]
        argval = self.code_options['co_names'][arg]
        self.LOAD_ATTR(dataclasses.replace(inst, argval=argval))

    def LOAD_METHOD(self, inst):
        if False:
            i = 10
            return i + 15
        self.LOAD_ATTR(inst)
        obj = self.pop()
        if sys.version_info >= (3, 11):
            self.PUSH_NULL(inst)
            self.push(obj)
        else:
            self.push(obj)
            self.push(None)

    def CALL_METHOD(self, inst):
        if False:
            return 10
        args = self.popn(inst.argval)
        dummy = self.pop()
        assert dummy is None
        fn = self.pop()
        self.call_function(fn, args, {})

    def LOAD_ATTR(self, inst):
        if False:
            for i in range(10):
                print('nop')
        obj = self.pop()
        result = BuiltinVariable(getattr).call_function(self, [obj, ConstantVariable.create(inst.argval)], {})
        self.push(result)

    def STORE_ATTR(self, inst):
        if False:
            for i in range(10):
                print('nop')
        speculation = self.speculate()
        if speculation.failed:
            return self.store_attr_graph_break(inst)
        (val, obj) = self.popn(2)
        if isinstance(obj, NNModuleVariable):
            assert not self.export, f'Mutating module attribute {inst.argval} during export.'
        try:
            BuiltinVariable(setattr).call_function(self, [obj, ConstantVariable.create(inst.argval), val], {})
            return
        except Unsupported as e:
            if not self.should_compile_partial_graph():
                raise
            log.debug('STORE_ATTR triggered compile', exc_info=True)
            e.remove_from_stats()
            e.add_to_stats('graph_break')
        speculation.fail_and_restart_analysis()

    def store_attr_graph_break(self, inst):
        if False:
            while True:
                i = 10
        self.output.compile_subgraph(self, reason=GraphCompileReason('store_attr', [self.frame_summary()]))
        self.output.add_output_instructions([copy.copy(inst)])
        self.popn(2)
        self.output.add_output_instructions(self.create_call_resume_at(self.next_instruction))

    def DELETE_ATTR(self, inst):
        if False:
            print('Hello World!')
        obj = self.pop()
        BuiltinVariable(delattr).call_function(self, [obj, ConstantVariable.create(inst.argval)], {})

    def create_call_resume_at(self, offset):
        if False:
            return 10
        raise AssertionError(f'create_call_resume_at not overridden by subclass {type(self)}')

    def should_compile_partial_graph(self) -> bool:
        if False:
            i = 10
            return i + 15
        raise AssertionError(f'should_compile_partial_graph not overridden by subclass {type(self)}')

    @break_graph_if_unsupported(push=0)
    def STORE_SUBSCR(self, inst):
        if False:
            for i in range(10):
                print('nop')
        (val, obj, key) = self.popn(3)
        result = obj.call_method(self, '__setitem__', [key, val], {})

    def BUILD_TUPLE(self, inst):
        if False:
            print('Hello World!')
        items = self.popn(inst.argval)
        self.push(TupleVariable(items))

    def BUILD_SLICE(self, inst):
        if False:
            for i in range(10):
                print('nop')
        items = self.popn(inst.argval)
        self.push(SliceVariable(items))

    def BUILD_LIST(self, inst):
        if False:
            return 10
        items = self.popn(inst.argval)
        self.push(ListVariable(items, mutable_local=MutableLocal()))

    def BUILD_SET(self, inst):
        if False:
            print('Hello World!')
        if config.inject_BUILD_SET_unimplemented_TESTING_ONLY:
            unimplemented('missing: BUILD_SET')
        items = self.popn(inst.argval)
        new_set = SetVariable(items, mutable_local=MutableLocal())
        self.push(new_set)

    def BUILD_LIST_UNPACK(self, inst, cls=ListVariable):
        if False:
            for i in range(10):
                print('nop')
        seqs = self.popn(inst.argval)
        items = list()
        for seq in seqs:
            try:
                items.extend(seq.unpack_var_sequence(self))
            except NotImplementedError:
                unimplemented(f'BUILD_LIST_UNPACK {seq}')
        self.push(cls(items, mutable_local=MutableLocal()))

    def BUILD_TUPLE_UNPACK(self, inst):
        if False:
            for i in range(10):
                print('nop')
        self.BUILD_LIST_UNPACK(inst, cls=TupleVariable)
    BUILD_TUPLE_UNPACK_WITH_CALL = BUILD_TUPLE_UNPACK

    def BUILD_MAP(self, inst):
        if False:
            print('Hello World!')
        items = self.popn(inst.argval * 2)
        result = dict()
        for (k, v) in zip(items[::2], items[1::2]):
            assert isinstance(k, (ConstantVariable, EnumVariable, BuiltinVariable)) or (isinstance(k, TensorVariable) and k.specialized_value is not None) or k.is_python_constant()
            result[ConstDictVariable.get_key(k)] = v
        assert len(result) == len(items) / 2
        self.push(ConstDictVariable(result, dict, mutable_local=MutableLocal()))

    def BUILD_MAP_UNPACK(self, inst):
        if False:
            print('Hello World!')
        items = self.popn(inst.argval)
        items = [BuiltinVariable(dict).call_function(self, [x], {}) for x in items]
        result = dict()
        for x in items:
            assert isinstance(x, ConstDictVariable)
            result.update(x.items)
        self.push(ConstDictVariable(result, dict, mutable_local=MutableLocal()))
    BUILD_MAP_UNPACK_WITH_CALL = BUILD_MAP_UNPACK

    def BUILD_CONST_KEY_MAP(self, inst):
        if False:
            for i in range(10):
                print('nop')
        keys = self.pop()
        values = self.popn(inst.argval)
        assert isinstance(keys, TupleVariable)
        assert keys.is_python_constant()
        keys = keys.as_python_constant()
        assert istype(keys, tuple)
        assert len(keys) == len(values)
        self.push(ConstDictVariable(dict(zip(keys, values)), dict, mutable_local=MutableLocal()))

    def MAP_ADD(self, inst):
        if False:
            print('Hello World!')
        (k, v) = self.popn(2)
        assert inst.argval > 0
        obj = self.stack[-inst.arg].realize()
        assert isinstance(obj, ConstDictVariable)
        assert obj.mutable_local
        items = dict(obj.items)
        items[k.as_python_constant()] = v
        self.replace_all(obj, ConstDictVariable(items, obj.user_cls))

    def SET_ADD(self, inst):
        if False:
            return 10
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, SetVariable)
        assert obj.mutable_local
        return obj.call_method(self, 'add', [v], {})

    def LIST_APPEND(self, inst):
        if False:
            for i in range(10):
                print('nop')
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg].realize()
        assert isinstance(obj, ListVariable)
        assert obj.mutable_local
        self.replace_all(obj, ListVariable(obj.items + [v]))

    def MAKE_FUNCTION(self, inst):
        if False:
            return 10
        flags = inst.arg
        old_stack = list(self.stack)
        if sys.version_info < (3, 11):
            fn_name = self.pop()
        code = self.pop()
        if sys.version_info >= (3, 11):
            assert hasattr(code.value, 'co_qualname')
            fn_name = ConstantVariable.create(value=code.value.co_qualname)
        defaults = None
        closure = None
        annotations = None
        kwdefaults = None
        if flags & 8:
            closure = self.pop()
        if flags & 4:
            annotations = self.pop()
        if flags & 2:
            kwdefaults = self.pop()
        if flags & 1:
            defaults = self.pop()
        self.push(NestedUserFunctionVariable(fn_name, code, self.f_globals, defaults, kwdefaults, annotations, closure, closure_scope=self))

    def UNPACK_SEQUENCE(self, inst):
        if False:
            i = 10
            return i + 15
        seq = self.pop()
        if isinstance(seq, (BaseListVariable, SetVariable)):
            val = seq.unpack_var_sequence(self)
        elif seq.is_python_constant() and isinstance(seq, ConstantVariable):
            val = seq.unpack_var_sequence(self)
        elif isinstance(seq, TensorVariable):
            val = seq.unpack_var_sequence(self, idxes=range(inst.argval))
        elif isinstance(seq, GetAttrVariable) and isinstance(seq.obj, TensorVariable):
            proxy = getattr(seq.obj.as_proxy(), seq.name)
            val = [wrap_fx_proxy(self, proxy[i]) for i in range(inst.argval)]
        else:
            unimplemented(f'UNPACK_SEQUENCE {seq}')
        assert len(val) == inst.argval
        for i in reversed(val):
            self.push(i)

    def UNPACK_EX(self, inst):
        if False:
            for i in range(10):
                print('nop')
        assert 0 <= inst.argval <= 65535
        prefix = inst.argval & 255
        suffix = inst.argval >> 8
        seq = self.pop()
        if seq.has_unpack_var_sequence(self):
            vals = list(seq.unpack_var_sequence(self))
            assert len(vals) >= prefix + suffix
            vals_prefix = vals[:prefix]
            vals_list = vals[prefix:len(vals) - suffix]
            vals_suffix = vals[len(vals) - suffix:]
            for item in reversed(vals_suffix):
                self.push(item)
            self.push(TupleVariable(vals_list))
            for item in reversed(vals_prefix):
                self.push(item)
        else:
            unimplemented(f'UNPACK_EX {seq}')

    def NOP(self, inst):
        if False:
            print('Hello World!')
        pass

    def POP_TOP(self, inst):
        if False:
            print('Hello World!')
        self.pop()

    def ROT_TWO(self, inst):
        if False:
            return 10
        a = self.pop()
        b = self.pop()
        self.push(a)
        self.push(b)

    def ROT_THREE(self, inst):
        if False:
            return 10
        a = self.pop()
        b = self.pop()
        c = self.pop()
        self.push(a)
        self.push(c)
        self.push(b)

    def ROT_FOUR(self, inst):
        if False:
            i = 10
            return i + 15
        a = self.pop()
        b = self.pop()
        c = self.pop()
        d = self.pop()
        self.push(a)
        self.push(d)
        self.push(c)
        self.push(b)

    def DUP_TOP(self, inst):
        if False:
            return 10
        a = self.pop()
        self.push(a)
        self.push(a)

    def DUP_TOP_TWO(self, inst):
        if False:
            while True:
                i = 10
        a = self.pop()
        b = self.pop()
        self.push(b)
        self.push(a)
        self.push(b)
        self.push(a)

    def FORMAT_VALUE(self, inst):
        if False:
            for i in range(10):
                print('nop')
        flags = inst.arg
        if flags & 4 == 4:
            fmt_spec = self.pop()
        else:
            fmt_spec = ConstantVariable.create('')
        value = self.pop()
        if isinstance(value, SymNodeVariable):
            value = ConstantVariable.create(str(value.sym_num))
        if flags & 3 == 1:
            value = BuiltinVariable(str).call_function(self, [value], {})
        elif flags & 3 == 2:
            value = BuiltinVariable(repr).call_function(self, [value], {})
        elif flags & 3 == 3:
            value = BuiltinVariable(ascii).call_function(self, [value], {})
        fmt_var = ConstantVariable.create('{:' + fmt_spec.as_python_constant() + '}')
        self.call_function(BuiltinVariable(str.format), [fmt_var, value], {})

    def BUILD_STRING(self, inst):
        if False:
            print('Hello World!')
        result = ''
        for _ in range(inst.arg):
            str_var = self.pop()
            assert isinstance(str_var, ConstantVariable)
            result = str_var.value + result
        self.push(ConstantVariable.create(value=result))

    def IS_OP(self, inst):
        if False:
            print('Hello World!')
        assert inst.argval == 0 or inst.argval == 1
        if inst.argval == 0:
            new_argval = 'is'
        else:
            new_argval = 'is not'
        new_inst = create_instruction('COMPARE_OP', argval=new_argval)
        self.COMPARE_OP(new_inst)

    def CONTAINS_OP(self, inst):
        if False:
            print('Hello World!')
        assert inst.argval == 0 or inst.argval == 1
        (left, right) = self.popn(2)
        op = inst.argval
        self.push(right.call_method(self, '__contains__', [left], {}))
        if op == 1:
            self.UNARY_NOT(inst)

    def LIST_EXTEND(self, inst):
        if False:
            return 10
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, ListVariable)
        assert obj.mutable_local
        obj.call_method(self, 'extend', [v], {})

    def LIST_TO_TUPLE(self, inst):
        if False:
            while True:
                i = 10
        self.push(BuiltinVariable(tuple).call_function(self, [self.pop()], {}))

    def DICT_MERGE(self, inst):
        if False:
            return 10
        v = self.pop()
        assert inst.argval > 0
        obj = self.stack[-inst.arg]
        assert isinstance(obj, ConstDictVariable)
        assert obj.mutable_local
        obj.call_method(self, 'update', [v], {})
    DICT_UPDATE = DICT_MERGE

    def GEN_START(self, inst):
        if False:
            i = 10
            return i + 15
        self.pop()

    def GET_LEN(self, inst):
        if False:
            i = 10
            return i + 15
        tos = self.stack[-1]
        if tos.is_python_constant():
            self.push(ConstantVariable.create(len(tos.as_python_constant())))
        else:
            self.push(tos.call_method(self, '__len__', [], {}))

    def MATCH_MAPPING(self, inst):
        if False:
            while True:
                i = 10
        tos = self.stack[-1]
        assert isinstance(tos, ConstDictVariable)
        if isinstance(tos.items, collections.abc.Mapping):
            self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(False))

    def MATCH_SEQUENCE(self, inst):
        if False:
            i = 10
            return i + 15
        tos = self.stack[-1]
        assert tos.is_python_constant()
        tos_value = tos.as_python_constant()
        if isinstance(tos_value, collections.abc.Sequence) and (not isinstance(tos_value, (str, bytes, bytearray))):
            self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(False))

    def MATCH_KEYS(self, inst):
        if False:
            while True:
                i = 10
        tos = self.stack[-1]
        assert tos.is_python_constant()
        keys = tos.as_python_constant()
        tos1 = self.stack[-2]
        assert isinstance(tos1, ConstDictVariable)
        match_obj = tos1.items
        if all((key in match_obj for key in keys)):
            self.push(TupleVariable([match_obj[key] for key in keys]))
            if sys.version_info < (3, 11):
                self.push(ConstantVariable.create(True))
        else:
            self.push(ConstantVariable.create(None))
            if sys.version_info < (3, 11):
                self.push(ConstantVariable.create(False))

    def LOAD_ASSERTION_ERROR(self, inst):
        if False:
            i = 10
            return i + 15
        unimplemented('assert with non-string message')
    UNARY_POSITIVE = stack_op(operator.pos)
    UNARY_NEGATIVE = stack_op(operator.neg)
    UNARY_NOT = stack_op(operator.not_)
    UNARY_INVERT = stack_op(operator.invert)
    BINARY_POWER = stack_op(operator.pow)
    BINARY_MULTIPLY = stack_op(operator.mul)
    BINARY_MATRIX_MULTIPLY = stack_op(operator.matmul)
    BINARY_FLOOR_DIVIDE = stack_op(operator.floordiv)
    BINARY_TRUE_DIVIDE = stack_op(operator.truediv)
    BINARY_MODULO = stack_op(operator.mod)
    BINARY_REMAINDER = stack_op(operator.mod)
    BINARY_ADD = stack_op(operator.add)
    BINARY_SUBTRACT = stack_op(operator.sub)
    BINARY_SUBSCR = break_graph_if_unsupported(push=1)(stack_op(operator.getitem))
    BINARY_LSHIFT = stack_op(operator.lshift)
    BINARY_RSHIFT = stack_op(operator.rshift)
    BINARY_AND = stack_op(operator.and_)
    BINARY_OR = stack_op(operator.or_)
    BINARY_XOR = stack_op(operator.xor)
    INPLACE_POWER = stack_op(operator.ipow)
    INPLACE_MULTIPLY = stack_op(operator.imul)
    INPLACE_MATRIX_MULTIPLY = stack_op(operator.imatmul)
    INPLACE_FLOOR_DIVIDE = stack_op(operator.ifloordiv)
    INPLACE_TRUE_DIVIDE = stack_op(operator.itruediv)
    INPLACE_MODULO = stack_op(operator.imod)
    INPLACE_REMAINDER = stack_op(operator.imod)
    INPLACE_ADD = stack_op(operator.iadd)
    INPLACE_SUBTRACT = stack_op(operator.isub)
    INPLACE_LSHIFT = stack_op(operator.ilshift)
    INPLACE_RSHIFT = stack_op(operator.irshift)
    INPLACE_AND = stack_op(operator.iand)
    INPLACE_XOR = stack_op(operator.ixor)
    INPLACE_OR = stack_op(operator.ior)

    def RESUME(self, inst):
        if False:
            print('Hello World!')
        if inst.arg == 0:
            self.append_prefix_inst(inst)
            self.accept_prefix_inst = False
        else:
            assert not self.accept_prefix_inst

    def BINARY_OP(self, inst):
        if False:
            i = 10
            return i + 15
        if sys.version_info >= (3, 11):
            opname = dis._nb_ops[inst.arg][0][3:]
            if opname.startswith('INPLACE'):
                return getattr(self, 'INPLACE_' + opname[8:])(inst)
            return getattr(self, 'BINARY_' + opname)(inst)
        else:
            unimplemented('BINARY_OP requires Python 3.11+')

    def PRECALL(self, inst):
        if False:
            print('Hello World!')
        pass

    def KW_NAMES(self, inst):
        if False:
            print('Hello World!')
        kw_names = self.code_options['co_consts'][inst.arg]
        assert isinstance(kw_names, tuple)
        for name in kw_names:
            assert isinstance(name, str)
        assert self.kw_names is None
        self.kw_names = ConstantVariable.create(value=kw_names)

    def PUSH_NULL(self, inst):
        if False:
            for i in range(10):
                print('nop')
        self.push(NullVariable())

    @break_graph_if_unsupported(push=1)
    def CALL(self, inst):
        if False:
            print('Hello World!')
        contents = self.popn(inst.arg + 2)
        if isinstance(contents[0], NullVariable):
            fn = contents[1]
            args = []
        else:
            fn = contents[0]
            args = [contents[1]]
        kw_names = self.kw_names.value if self.kw_names else ()
        if kw_names:
            args = args + contents[2:-len(kw_names)]
            kwargs_list = contents[-len(kw_names):]
            kwargs = dict(zip(kw_names, kwargs_list))
            assert len(kwargs) == len(kw_names)
        else:
            args = args + contents[2:]
            kwargs = {}
        self.call_function(fn, args, kwargs)
        self.kw_names = None

    def COPY(self, inst):
        if False:
            return 10
        self.push(self.stack[-inst.arg])

    def SWAP(self, inst):
        if False:
            i = 10
            return i + 15
        (self.stack[-1], self.stack[-inst.arg]) = (self.stack[-inst.arg], self.stack[-1])
    JUMP_BACKWARD = jump
    JUMP_BACKWARD_NO_INTERRUPT = jump
    POP_JUMP_FORWARD_IF_TRUE = generic_jump(operator.truth, False)
    POP_JUMP_BACKWARD_IF_TRUE = generic_jump(operator.truth, False)
    POP_JUMP_FORWARD_IF_FALSE = generic_jump(operator.not_, False)
    POP_JUMP_BACKWARD_IF_FALSE = generic_jump(operator.not_, False)

    def CACHE(self, inst):
        if False:
            print('Hello World!')
        pass

    def BEFORE_WITH(self, inst):
        if False:
            i = 10
            return i + 15
        self.setup_or_before_with(inst)

    def setup_or_before_with(self, inst):
        if False:
            print('Hello World!')
        ctx = self.pop()
        if not isinstance(ctx, ContextWrappingVariable):
            unimplemented(f'{inst.opname} {ctx}')
        if isinstance(ctx, GenericContextWrappingVariable):
            self.generic_context_manager_depth += 1
        exit = WithExitFunctionVariable(ctx, inst.target)
        if sys.version_info >= (3, 11):
            assert self.next_instruction
            assert self.next_instruction.exn_tab_entry
            target = self.next_instruction.exn_tab_entry.target
        else:
            target = inst.target
        if isinstance(self, InstructionTranslator):
            self.block_stack.append(BlockStackEntry(target, len(self.stack), ctx))
        else:
            self.block_stack.append(BlockStackEntry(target))
        self.push(exit)
        self.push(ctx.enter(self))

    def append_prefix_inst(self, inst):
        if False:
            return 10
        assert self.accept_prefix_inst
        self.prefix_insts.append(inst)

    def MAKE_CELL(self, inst):
        if False:
            print('Hello World!')
        self.append_prefix_inst(inst)

    def COPY_FREE_VARS(self, inst):
        if False:
            i = 10
            return i + 15
        self.append_prefix_inst(inst)

    def RETURN_GENERATOR(self, inst):
        if False:
            print('Hello World!')
        self.append_prefix_inst(inst)

    def copy_graphstate(self) -> InstructionTranslatorGraphState:
        if False:
            i = 10
            return i + 15
        'Create a checkpoint of the current state by copying everything'
        return InstructionTranslatorGraphState(self.output.copy_graphstate(), dict(self.symbolic_locals), list(self.stack), list(self.block_stack), self.instruction_pointer, self.current_instruction, self.next_instruction, self.lineno)

    def restore_graphstate(self, state: InstructionTranslatorGraphState):
        if False:
            i = 10
            return i + 15
        'Restore a checkpoint created by self.copy_graphstate()'
        (output_state, self.symbolic_locals, self.stack, self.block_stack, self.instruction_pointer, self.current_instruction, self.next_instruction, self.lineno) = state
        self.output.restore_graphstate(output_state)

    def is_non_empty_graph(self):
        if False:
            for i in range(10):
                print('nop')
        if self.output.count_calls() > 1:
            self.is_non_empty_graph = lambda : True
            return True
        return False

    def format_frame_summary(self, additional_stack_frames=None):
        if False:
            i = 10
            return i + 15
        if additional_stack_frames is None:
            additional_stack_frames = []
        return ''.join(traceback.format_list([self.frame_summary()] + list(reversed(additional_stack_frames))))

    def frame_summary(self):
        if False:
            while True:
                i = 10
        return traceback.FrameSummary(getattr(self.f_code, 'co_filename', '<unknown>'), self.lineno, getattr(self.f_code, 'co_name', '<unknown>'), lookup_line=False)

    def store_global_weakref(self, name, value):
        if False:
            i = 10
            return i + 15
        install_guard(GlobalWeakRefSource(name).make_guard(GuardBuilder.WEAKREF_ALIVE))
        if name not in self.output.global_scope:
            self.output.install_global(name, weakref.ref(value))

    @property
    def fake_mode(self):
        if False:
            i = 10
            return i + 15
        return self._fake_mode

    def find_symbolic_locals_name(self, tensor_variable):
        if False:
            for i in range(10):
                print('nop')
        for (key, value) in self.symbolic_locals.items():
            if value is tensor_variable:
                return key
        return None

    @contextlib.contextmanager
    def strict_translation_mode(self):
        if False:
            while True:
                i = 10
        self.strict_checks_enabled = True
        try:
            yield
        finally:
            self.strict_checks_enabled = False

    def speculate(self) -> SpeculationEntry:
        if False:
            for i in range(10):
                print('nop')
        return self.speculation_log.next(self.f_code.co_filename, self.lineno, self.instruction_pointer)

    def __init__(self, output: OutputGraph, instructions: List[Instruction], f_locals: Dict[str, Any], f_globals: Dict[str, Any], f_builtins: Dict[str, Any], code_options: Dict[str, Any], symbolic_locals: Dict[str, VariableTracker], symbolic_globals: Dict[str, VariableTracker], f_code: types.CodeType, export: bool, inline_depth: int, speculation_log: SpeculationLog):
        if False:
            print('Hello World!')
        super().__init__()
        self.speculation_log = speculation_log
        self.output = output
        self.symbolic_locals = symbolic_locals
        self.symbolic_globals = symbolic_globals
        self.stack = []
        self.instruction_pointer = 0
        self.current_instruction = create_instruction('NOP')
        self.next_instruction = None
        self.block_stack = []
        self.generic_context_manager_depth = 0
        self.lineno = code_options['co_firstlineno']
        self.kw_names = None
        self.accept_prefix_inst = True
        self.prefix_insts = []
        self.instructions: List[Instruction] = instructions
        self.indexof: Dict[Instruction, int] = get_indexof(self.instructions)
        self.f_locals: Dict[str, Any] = f_locals
        self.f_globals: Dict[str, Any] = f_globals
        self.f_builtins: Dict[str, Any] = f_builtins
        self.code_options: Dict[str, Any] = code_options
        self.f_code: types.CodeType = f_code
        self.exec_recorder = ExecutionRecorder(code=f_code, code_options=code_options)
        self.nn_module_stack: Dict[str, Tuple[str, Type[Any]]] = {}
        self.export = export
        self._fake_mode = output.tracing_context.fake_mode
        self.current_speculation = None
        self.random_calls = []
        self.strict_checks_enabled = False
        if sys.version_info >= (3, 10):
            from .resume_execution import CO_ASYNC_GENERATOR, CO_COROUTINE, CO_GENERATOR, CO_ITERABLE_COROUTINE
            if f_code.co_flags & (CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR):
                self.push(BuiltinVariable(None))
        self.inline_depth = inline_depth
        self.inconsistent_side_effects = False
        linecache.lazycache(f_code.co_filename, f_globals)
        self.log_starts_line()

class InstructionTranslator(InstructionTranslatorBase):
    mutated_closure_cell_contents: Set[str]

    @staticmethod
    def current_tx() -> 'InstructionTranslator':
        if False:
            while True:
                i = 10
        return tls.current_tx

    @contextlib.contextmanager
    def set_current_tx(self):
        if False:
            return 10
        prior = getattr(tls, 'current_tx', None)
        tls.current_tx = self
        try:
            yield
        finally:
            tls.current_tx = prior

    def __init__(self, instructions: List[Instruction], f_code, f_locals, f_globals, f_builtins, code_options, compiler_fn, one_graph, export, export_constraints, mutated_closure_cell_contents: Set[str], frame_state, speculation_log: SpeculationLog):
        if False:
            while True:
                i = 10
        _step_logger()(logging.INFO, f"torchdynamo start tracing {f_code.co_name} {code_options['co_filename']}:{code_options['co_firstlineno']}")
        super().__init__(output=OutputGraph(code_options, compiler_fn, self, export, export_constraints, frame_state, local_scope=f_locals, global_scope=f_globals, f_code=f_code), instructions=instructions, f_locals=f_locals, f_globals=f_globals, f_builtins=f_builtins, code_options=code_options, symbolic_locals={}, symbolic_globals={}, f_code=f_code, export=export, inline_depth=0, speculation_log=speculation_log)
        with tracing(self.output.tracing_context), self.set_current_tx():
            self.one_graph: bool = one_graph
            self.export = export
            self.mutated_closure_cell_contents = mutated_closure_cell_contents
            if self.export:
                assert self.one_graph, 'Export without one graph - something has gone wrong.'
            vars = list(code_options['co_varnames'])
            cells_and_freevars = [x for x in self.cell_and_freevars() if x not in vars]
            vars.extend(cells_and_freevars)
            cells_and_freevars_set = set(cells_and_freevars)
            self.symbolic_locals = {k: variables.LazyVariableTracker.create(f_locals[k], source=LocalSource(k, cell_or_freevar=k in cells_and_freevars_set)) for k in vars if k in f_locals}
            if export:
                self.symbolic_locals = VariableTracker.apply(lambda x: x.realize(), self.symbolic_locals)
            self._freevars_ids = dict()
            for name in self.code_options['co_freevars']:
                if name in f_locals:
                    self._freevars_ids[name] = id(f_locals[name])

    def run(self):
        if False:
            while True:
                i = 10
        super().run()

    def match_nested_cell(self, name, cell):
        if False:
            for i in range(10):
                print('nop')
        'Match a cell in this method to one in a function we are inlining'
        value = cell.cell_contents
        if id(value) != self._freevars_ids.get(name):
            return None
        return self.symbolic_locals[name]

    def should_compile_partial_graph(self):
        if False:
            while True:
                i = 10
        return all((b.can_restore() for b in self.block_stack)) and (not self.one_graph) and (self.generic_context_manager_depth == 0)

    def create_call_resume_at(self, inst):
        if False:
            while True:
                i = 10
        self.instruction_pointer = None
        if inst.opname == 'RETURN_VALUE':
            return [create_instruction('RETURN_VALUE')]
        reads = livevars_analysis(self.instructions, inst)
        argnames = tuple((k for k in self.symbolic_locals.keys() if k in reads and k not in self.cell_and_freevars()))
        cg = PyCodegen(self)
        null_idxes: List[int] = []
        if sys.version_info >= (3, 11):
            for (i, var) in enumerate(self.stack):
                if isinstance(var, NullVariable):
                    null_idxes.append(i)
            null_cnt = 0
            for (i, var) in enumerate(reversed(self.stack)):
                if isinstance(var, NullVariable):
                    for j in range(2, i + 2 - null_cnt):
                        cg.append_output(create_instruction('SWAP', arg=j))
                    cg.extend_output(cg.pop_null())
                    null_cnt += 1
        stack_len = len(self.stack) - len(null_idxes)
        nargs = stack_len + len(argnames)
        name = unique_id(f'__resume_at_{inst.offset}')
        new_code: types.CodeType = ContinueExecutionCache.lookup(self.f_code, self.lineno, inst.offset, tuple((b.target.offset for b in self.block_stack)), stack_len, argnames, tuple((b.resume_fn() for b in self.block_stack)), tuple(null_idxes))
        orig_graphmodule_maybe = code_context.get_context(self.f_code).get('orig_graphmodule', None)
        if orig_graphmodule_maybe is not None:
            code_context.get_context(new_code)['orig_graphmodule'] = orig_graphmodule_maybe
        if new_code.co_freevars:
            cg.make_function_with_closure(name, new_code, True, stack_len)
        else:
            self.output.install_global(name, types.FunctionType(new_code, self.f_globals, name))
            cg.extend_output(cg.load_function_name(name, True, stack_len))
        cg.extend_output([cg.create_load(k) for k in argnames])
        cg.extend_output(create_call_function(nargs, False))
        cg.append_output(create_instruction('RETURN_VALUE'))
        return cg.get_instructions()

    def symbolic_locals_contain_module_class(self):
        if False:
            return 10
        for v in self.symbolic_locals.values():
            if isinstance(v, UserDefinedClassVariable) and issubclass(v.as_python_constant(), torch.nn.Module):
                return True
        return False

    def RETURN_VALUE(self, inst):
        if False:
            print('Hello World!')
        if self.output.count_calls() == 0 and (not self.inconsistent_side_effects) and (not self.symbolic_locals_contain_module_class()) and (not self.export):
            raise exc.SkipFrame('because no content in function call')
        self.instruction_pointer = None
        _step_logger()(logging.INFO, f'torchdynamo done tracing {self.f_code.co_name} (RETURN_VALUE)')
        log.debug('RETURN_VALUE triggered compile')
        self.output.compile_subgraph(self, reason=GraphCompileReason('return_value', [self.frame_summary()], graph_break=False))
        self.output.add_output_instructions([create_instruction('RETURN_VALUE')])

class InliningInstructionTranslator(InstructionTranslatorBase):
    """Trace and inline a called method"""
    symbolic_result: Optional[TensorVariable]

    @classmethod
    def inline_call(cls, parent, func, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        with patch.dict(counters, {'unimplemented': counters['inline_call']}):
            return cls.inline_call_(parent, func, args, kwargs)

    @staticmethod
    def check_inlineable(func):
        if False:
            print('Hello World!')
        if func.has_self():
            unimplemented('inline with __self__')
        if func.get_name() == 'patched_init':
            unimplemented('Patched init cannot be inlined.')
        try:
            func_value = func.get_function()
        except NotImplementedError:
            func_value = None
        if func.get_name() == '__torch_function__' or func_value is torch._tensor._convert:
            return skipfiles.SkipResult(False, 'Allow __torch_function__')
        if func_value and id(func_value) in allowed_functions._disallowed_function_ids:
            unimplemented(f'inlining disallowed: {func_value}')
        result = skipfiles.check_verbose(func, allow_torch=True)
        if result.skipped:
            from torch._dynamo.variables.misc import produce_trampoline_autograd_apply, produce_trampoline_autograd_bwd, produce_trampoline_autograd_fwd
            if hasattr(func.fn, '_origin') and func.fn._origin in [produce_trampoline_autograd_fwd, produce_trampoline_autograd_apply, produce_trampoline_autograd_bwd]:
                return skipfiles.SkipResult(False, 'allowlist in dynamo known function')
            unimplemented(f"'inline in skipfiles: {func.fn.__qualname__} | {func.get_name()} {func.get_filename()}, {result.reason}'")
        if isinstance(func, UserFunctionVariable) and inspect.getattr_static(func.get_function(), '_torchdynamo_disable', False):
            unimplemented(f'call torch._dynamo.disable() wrapped function {func.get_function()}')
        else:
            return result

    @staticmethod
    def inline_call_(parent, func: VariableTracker, args: List[VariableTracker], kwargs):
        if False:
            i = 10
            return i + 15
        assert isinstance(func, (UserFunctionVariable, NestedUserFunctionVariable))
        result = InliningInstructionTranslator.check_inlineable(func)
        assert result.skipped is False
        try:
            (sub_locals, closure_cells) = func.bind_args(parent, args, kwargs)
        except TypeError as e:
            raise ArgsMismatchError('{reason}.\n  func = {func}, args = {args}, kwargs = {kwargs}'.format(reason=str(e), func=f"'{func.get_name()}' {func.get_filename()}:{func.get_code().co_firstlineno}", args=[arg.python_type() for arg in args], kwargs=kwargs))
        for v in itertools.chain(sub_locals.values(), closure_cells.values()):
            if not isinstance(v, VariableTracker):
                unimplemented(f'unconverted arg {v}')
        code: types.CodeType = func.get_code()
        if code.co_name in ('__setitem__', '__setattr__') and (not (args is not None and len(args) > 0 and isinstance(args[0], variables.CustomizedDictVariable))):
            unimplemented(f'inline {code.co_name}')
        suffix = ''
        if torch._logging._internal.log_state.is_artifact_enabled('output_code'):
            suffix = f'\n{dis.Bytecode(code).dis()}'
        if sys.version_info >= (3, 11):
            cur_inst = parent.current_instruction
            parent_code = parent.f_code
            header = parent.get_line_of_code_header(lineno=cur_inst.positions.lineno)

            def get_trace_call_log_str():
                if False:
                    while True:
                        i = 10
                line = get_instruction_source_311(parent_code, cur_inst).rstrip()
                return f'TRACE inlined call {code.co_name} from {header}\n{line}'
            trace_call_log.debug('%s', LazyString(get_trace_call_log_str))
        log.debug('INLINING %s%s, %s', code, suffix, result.reason)
        if args and isinstance(args[0], NNModuleVariable):
            module = parent.output.get_submodule(args[0].module_key)
            if isinstance(module, torch.fx.GraphModule):
                code_context.get_context(module.forward.__code__)['orig_graphmodule'] = module
        tracer: InliningInstructionTranslator
        if is_generator(code):
            tracer = InliningGeneratorInstructionTranslator(parent, code, sub_locals, parent.symbolic_globals, closure_cells, func)
        else:
            tracer = InliningInstructionTranslator(parent, code, sub_locals, parent.symbolic_globals, closure_cells, func)
        strict_ctx: Any = contextlib.nullcontext()
        if parent.strict_checks_enabled:
            strict_ctx = tracer.strict_translation_mode()
        try:
            with strict_ctx:
                tracer.run()
        except exc.SkipFrame as e:
            msg = f'SKIPPED INLINING {code}: {e}'
            log.debug(msg)
            raise Unsupported(msg) from e
        except Exception as e:
            log.debug('FAILED INLINING %s', code)
            raise
        assert tracer.symbolic_result is not None
        func.export_freevars(parent, tracer)
        if tracer.f_globals is parent.f_globals:
            parent.symbolic_globals.update(tracer.symbolic_globals)
        parent.inconsistent_side_effects |= tracer.inconsistent_side_effects
        log.debug('DONE INLINING %s', code)
        if is_generator(code):
            assert isinstance(tracer, InliningGeneratorInstructionTranslator)
            assert tracer.symbolic_result.as_python_constant() is None
            return ListIteratorVariable(tracer.generated_items, mutable_local=MutableLocal())
        else:
            return tracer.symbolic_result

    def __init__(self, parent: InstructionTranslatorBase, code: types.CodeType, symbolic_locals: Dict[str, VariableTracker], symbolic_globals: Dict[str, VariableTracker], closure_cells: Dict[str, VariableTracker], funcvar: BaseUserFunctionVariable):
        if False:
            for i in range(10):
                print('nop')
        f_globals = funcvar.get_globals()
        f_builtins = f_globals['__builtins__']
        if not isinstance(f_builtins, dict):
            f_builtins = f_builtins.__dict__
        instructions = cleaned_instructions(code)
        propagate_line_nums(instructions)
        super().__init__(output=parent.output, f_locals={}, f_globals=f_globals, f_builtins=f_builtins, symbolic_locals=symbolic_locals, symbolic_globals=symbolic_globals, instructions=instructions, code_options={k: getattr(code, k) for k in dir(code)}, f_code=code, export=parent.export, inline_depth=parent.inline_depth + 1, speculation_log=parent.speculation_log)
        self.parent = parent
        self.symbolic_result = None
        self.closure_cells = closure_cells
        self.nn_module_stack = parent.nn_module_stack.copy()

    @property
    def fake_mode(self):
        if False:
            print('Hello World!')
        return self.parent.fake_mode

    def run_ctx_mgr(self):
        if False:
            return 10
        return TracingContext.current_frame(self.parent.frame_summary())

    def STORE_DEREF(self, inst):
        if False:
            print('Hello World!')
        if inst.argval in self.closure_cells:
            cell = self.closure_cells[inst.argval]
            val = self.pop()
            if isinstance(cell, ClosureVariable):
                if not self.output.is_root_tracer():
                    unimplemented('HigherOrderOperator: Mutating a variable not in the current scope (ClosureVariable)')
                self.output.root_tx.symbolic_locals[cell.name] = val
            else:
                self.output.side_effects.store_cell(cell, val)
        else:
            maybe_cell = self.symbolic_locals.get(inst.argval)
            if isinstance(maybe_cell, variables.NewCellVariable):
                self.output.side_effects.store_cell(self.symbolic_locals[inst.argval], self.pop())
            else:
                if maybe_cell is not None and maybe_cell.source.name() not in self.output.root_tx.mutated_closure_cell_contents:
                    self.output.root_tx.mutated_closure_cell_contents.add(maybe_cell.source.name())
                    raise exc.UnspecializeRestartAnalysis()
                unimplemented('write to __closure__ while inlining')

    def LOAD_DEREF(self, inst):
        if False:
            while True:
                i = 10
        if inst.argval in self.closure_cells:
            cell = self.closure_cells[inst.argval]
            if isinstance(cell, ClosureVariable):
                self.push(self.output.root_tx.symbolic_locals[cell.name])
            else:
                self.push(self.output.side_effects.load_cell(cell))
        else:
            maybe_sym_local = self.symbolic_locals.get(inst.argval, None)
            if isinstance(maybe_sym_local, variables.NewCellVariable):
                self.push(self.output.side_effects.load_cell(maybe_sym_local))
            else:
                super().LOAD_DEREF(inst)

    def LOAD_CLOSURE(self, inst):
        if False:
            i = 10
            return i + 15
        assert inst.argval in self.cell_and_freevars()
        if inst.argval in self.closure_cells:
            self.push(self.closure_cells[inst.argval])
        else:
            self.push(InlinedClosureVariable(name=inst.argval))

    def check_replace_is_safe(self, oldvar):
        if False:
            while True:
                i = 10
        if not is_side_effect_safe(oldvar.mutable_local):
            unimplemented('HigherOrderOperator: Mutating a variable not in the current scope (replace_all)')

    def replace_all(self, oldvar: VariableTracker, newvar: VariableTracker):
        if False:
            for i in range(10):
                print('nop')
        self.check_replace_is_safe(oldvar)
        newvar = super().replace_all(oldvar, newvar)
        translator: InstructionTranslatorBase = self
        while hasattr(translator, 'parent'):
            translator = translator.parent
            translator.update_locals_and_stack(oldvar, newvar)
        return newvar

    def should_compile_partial_graph(self):
        if False:
            i = 10
            return i + 15
        return False

    def create_call_resume_at(self, offset):
        if False:
            while True:
                i = 10
        unimplemented('cant resume while inlining')

    def RETURN_VALUE(self, inst):
        if False:
            i = 10
            return i + 15
        self.symbolic_result = self.pop()
        self.instruction_pointer = None

class InliningGeneratorInstructionTranslator(InliningInstructionTranslator):
    generated_items: List[VariableTracker]

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.generated_items = []

    def YIELD_VALUE(self, inst: Instruction):
        if False:
            i = 10
            return i + 15
        self.generated_items.append(self.pop())
        self.push(ConstantVariable.create(None))

    def GET_YIELD_FROM_ITER(self, inst):
        if False:
            i = 10
            return i + 15
        tos = self.stack[-1]
        if not isinstance(tos, ListIteratorVariable):
            self.pop()
            res = BuiltinVariable(iter).call_function(self, [tos], {})
            self.push(res)
        return self.YIELD_FROM(inst)

    def YIELD_FROM(self, inst):
        if False:
            i = 10
            return i + 15
        while True:
            tos = self.stack[-1].realize()
            if isinstance(tos, ConstantVariable) and tos.value is None:
                self.pop()
                return
            if isinstance(tos, (variables.ListIteratorVariable, variables.IteratorVariable)):
                try:
                    (val, next_iter) = tos.next_variables(self)
                    self.push(val)
                    self.YIELD_VALUE(inst)
                    self.pop()
                    self.push(next_iter)
                except StopIteration:
                    return
            else:
                unimplemented(f'YIELD_FROM {typestr(tos)}')

    def SEND(self, inst):
        if False:
            i = 10
            return i + 15
        assert len(self.stack) >= 2
        val = self.pop()
        tos = self.stack[-1]
        if isinstance(tos, ListIteratorVariable):
            if isinstance(val, ConstantVariable) and val.value is None:
                self.push(val)
                self.instruction_pointer = self.indexof[inst.target]
            else:
                unimplemented('Unreachable sub-generator code')
        else:
            unimplemented(f'SEND {typestr(tos)}')