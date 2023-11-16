from __future__ import annotations
import builtins
import inspect
from collections import namedtuple
from copy import deepcopy
from functools import cached_property
from typing import Any, Callable
from ...infer_meta import InferMetaCache, LayerInferMetaCache, MetaInfo
from ...profiler import EventGuard, event_register
from ...symbolic.statement_ir import Reference, Symbol
from ...symbolic.symbolic_context import SymbolicTraceContext
from ...utils import ENV_SHOW_TRACKERS, NameGenerator, OrderedSet, inner_error_default_handler, is_inplace_api, is_paddle_api, log, log_do, map_if, tmp_name_guard
from ..instruction_utils import get_instructions
from .guard import Guard, StringifyExpression, make_guard
from .mutable_data import MutationDel, MutationNew, MutationSet
from .pycode_generator import PyCodeGen
from .side_effects import DictSideEffectRestorer, GlobalDelSideEffectRestorer, GlobalSetSideEffectRestorer, ListSideEffectRestorer, ObjDelSideEffectRestorer, ObjSetSideEffectRestorer, SideEffectRestorer, SideEffects
from .tracker import BuiltinTracker, DummyTracker
from .variables import DictVariable, GlobalVariable, ListVariable, NullVariable, PaddleLayerVariable, TensorVariable, VariableBase, VariableFactory, find_traceable_vars, map_variables

def convert_to_meta(inputs: Any):
    if False:
        i = 10
        return i + 15
    '\n    Convert the input variables to meta if it is TensorVariable.\n    '

    def func(x):
        if False:
            print('Hello World!')
        if isinstance(x, TensorVariable):
            return x.meta
        if isinstance(x, VariableBase):
            return x.get_py_value()
        return x
    return map_variables(func, inputs)

def convert_to_symbol(inputs: Any):
    if False:
        print('Hello World!')
    '\n    Convert the input variables to symbol if it can be symbolic.\n    '

    def func(x):
        if False:
            while True:
                i = 10
        if isinstance(x, (TensorVariable, PaddleLayerVariable)):
            return x.get_symbol()
        if isinstance(x, VariableBase):
            return x.get_py_value()
        return x
    return map_variables(func, inputs)

class FunctionGraph:
    """
    A Graph representation corresponding to each FunctionFrame
    The input binding diagram containing the current call represents three parts of output settings,
    This Graph can be compiled as a f_locals dependency function which produce the same outputs.
    """
    OUT_VAR_PREFIX = '___SIR_out_'
    Memo = namedtuple('function_graph_memo', ['inner_out', 'input_variables', 'stmt_ir', 'global_guards', 'side_effects_state', 'print_variables', 'inplace_tensors'])

    def __init__(self, frame, **kwargs):
        if False:
            print('Hello World!')
        self.sir_ctx = SymbolicTraceContext()
        self.inner_out = set()
        self.input_variables = []
        self.pycode_gen = PyCodeGen(frame, disable_eval_frame=True)
        self.side_effects = SideEffects()
        self._global_guarded_variables: OrderedSet[VariableBase] = OrderedSet()
        self._print_variables = []
        self._inplace_tensors = OrderedSet()
        self.build_strategy = kwargs.get('build_strategy', None)
        self._kwargs = kwargs

    @cached_property
    def _builtins(self):
        if False:
            while True:
                i = 10
        builtins_ = {}
        for (name, value) in builtins.__dict__.items():
            builtins_[name] = VariableFactory.from_value(value, self, BuiltinTracker(name), debug_name=name)
        return builtins_

    def add_print_variables(self, variable):
        if False:
            while True:
                i = 10
        '\n        Used to support psdb_print\n        '
        self._print_variables.append(variable)

    def add_inplace_tensors(self, variable):
        if False:
            print('Hello World!')
        '\n        Used to support psdb_print\n        '
        self._inplace_tensors.add(variable)

    def need_add_input(self, var):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if it is the input of graph.\n\n        Args:\n            var: The input variable.\n\n        '
        if var.id in self.inner_out:
            return False
        for v in self.input_variables:
            if v.id == var.id:
                return False
        return True

    def save_memo(self) -> FunctionGraph.Memo:
        if False:
            i = 10
            return i + 15
        "\n        Save the state of the current FunctionGraph, for future state recovery, it is used for state recovery during inline call error reporting\n\n        NOTE:\n            Why don't use __deepcopy__, because memo is not a deepcopy, i.e inner_out is only a shallow copy, SIR is a deepcopy.\n        "
        saved_stmt_ir = deepcopy(self.sir_ctx.TOS)
        return FunctionGraph.Memo(inner_out=set(self.inner_out), input_variables=list(self.input_variables), stmt_ir=saved_stmt_ir, global_guards=OrderedSet(self._global_guarded_variables), side_effects_state=self.side_effects.get_state(), print_variables=list(self._print_variables), inplace_tensors=OrderedSet(self._inplace_tensors))

    def restore_memo(self, memo: FunctionGraph.Memo):
        if False:
            while True:
                i = 10
        '\n        Restore the state of graph to memo.\n\n        Args:\n            memo: Previously recorded memo\n\n        '
        self.inner_out = memo.inner_out
        self.input_variables = memo.input_variables
        self.sir_ctx.replace_TOS(memo.stmt_ir)
        self._global_guarded_variables = memo.global_guards
        self.side_effects.restore_state(memo.side_effects_state)
        self._print_variables = memo.print_variables
        self._inplace_tensors = memo.inplace_tensors

    def collect_input_variables(self, inputs: list[VariableBase]):
        if False:
            while True:
                i = 10
        '\n        Variables required within the method\n\n        Args:\n            inputs: Required VariableBase\n        '

        def collect(inp):
            if False:
                return 10
            if isinstance(inp, VariableBase) and self.need_add_input(inp):
                self.input_variables.append(inp)
        map_variables(collect, inputs)

    @property
    @event_register('guard_fn')
    def guard_fn(self) -> Guard:
        if False:
            while True:
                i = 10
        with tmp_name_guard():
            guards = []
            with EventGuard('guard_fn: find vars and make stringify guard', event_level=1):
                for variable in find_traceable_vars(self.input_variables + list(self._global_guarded_variables)):
                    guards.extend(variable.make_stringify_guard())
            guards = OrderedSet(guards)
            for guard in guards:
                assert isinstance(guard, StringifyExpression), 'guard must be StringifyExpression.'
            return make_guard(guards)

    def _restore_origin_opcode(self, stack_vars, store_var_info, instr_idx):
        if False:
            i = 10
            return i + 15

        class VariableLoader:

            def __init__(self, store_var_info, pycode_gen):
                if False:
                    while True:
                        i = 10
                self._store_var_info = store_var_info
                self._pycode_gen: PyCodeGen = pycode_gen

            def load(self, var, allow_push_null=True):
                if False:
                    return 10
                if isinstance(var, NullVariable):
                    if allow_push_null:
                        var.reconstruct(self._pycode_gen)
                    else:
                        self._pycode_gen.gen_load_null_variable()
                    return
                self._pycode_gen.gen_load(self._store_var_info[var])
        origin_instrs = get_instructions(self.pycode_gen._origin_code)
        restore_instrs = origin_instrs[:instr_idx]
        restore_instr_names = [instr.opname for instr in restore_instrs[:instr_idx]]
        if restore_instr_names[-2:] == ['KW_NAMES', 'PRECALL']:
            restore_instrs = restore_instrs[:-2]
        for instr in restore_instrs:
            if instr.opname == 'LOAD_FAST' and instr.argval in self.pycode_gen._frame.f_locals.keys() and isinstance(self.pycode_gen._frame.f_locals[instr.argval], NullVariable):
                self.pycode_gen._frame.f_locals[instr.argval].reconstruct(self.pycode_gen)
            elif instr.opname == 'LOAD_GLOBAL' and instr.argval in self.pycode_gen._frame.f_globals.keys() and isinstance(self.pycode_gen._frame.f_globals[instr.argval], NullVariable):
                self.pycode_gen._frame.f_globals[instr.argval].reconstruct(self.pycode_gen)
            else:
                self.pycode_gen.extend_instrs([instr])
        nop = self.pycode_gen._add_instr('NOP')
        for instr in origin_instrs:
            if instr.jump_to == origin_instrs[instr_idx]:
                instr.jump_to = nop
        self.pycode_gen.hooks.append(lambda : self.pycode_gen.extend_instrs(iter(origin_instrs[instr_idx + 1:])))
        self.pycode_gen.gen_enable_eval_frame()
        name_gen = NameGenerator('__start_compile_saved_orig_')
        for var in stack_vars[::-1]:
            store_var_info[var] = name_gen.next()
            self.pycode_gen.gen_store_fast(store_var_info[var])
        return VariableLoader(store_var_info, self.pycode_gen)

    def _build_compile_fn_with_name_store(self, ret_vars, to_store_vars):
        if False:
            print('Hello World!')

        class VariableLoader:

            def __init__(self, index_for_load, pycode_gen):
                if False:
                    return 10
                self._index_for_load = index_for_load
                self._pycode_gen: PyCodeGen = pycode_gen

            def load(self, var, allow_push_null=True):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(var, NullVariable):
                    if allow_push_null:
                        var.reconstruct(self._pycode_gen)
                    else:
                        self._pycode_gen.gen_load_null_variable()
                    return
                self._pycode_gen.gen_load_fast(self._index_for_load[var.id])
        index_for_load = {}
        to_store_vars = list(filter(lambda x: not isinstance(x, NullVariable), to_store_vars))
        self.start_compile(*ret_vars + to_store_vars)
        name_gen = NameGenerator('__start_compile_saved_')
        for var in to_store_vars[::-1]:
            index_for_load[var.id] = name_gen.next()

            def _log_fn():
                if False:
                    while True:
                        i = 10
                print(f'[StartCompile] saved var: {index_for_load[var.id]} = ', var)
            log_do(4, _log_fn)
            self.pycode_gen.gen_store_fast(index_for_load[var.id])
        return VariableLoader(index_for_load, self.pycode_gen)

    @event_register('start_compile', event_level=2)
    def start_compile(self, *ret_vars: VariableBase):
        if False:
            while True:
                i = 10
        '\n        Generate bytecode based on the information collected by the simulation execution.\n\n        This consists of the following steps:\n        - Compile the FunctionGraph into a dy2st StaticFunction and load it in the generated bytecode\n        - Load the group network input\n        - Calling the generated dy2st StaticFunction\n        - Restore the side effects\n        - Restore the output\n        - Return the top of the stack\n        '
        from ..breakpoint import BreakpointManager
        BreakpointManager().on_event('start_compile')
        ret_items = [ret_item for ret_var in ret_vars for ret_item in ret_var.flatten_items()]
        tensor_items = self._find_tensor_outputs(ret_items)
        (compiled_fn, statment_ir) = self.sir_ctx.compile_fn([Symbol(tensor_var.var_name) for tensor_var in tensor_items], **self._kwargs)
        input_names = statment_ir.inputs
        compiled_fn_name = f'__compiled_fn_{statment_ir.name}'
        self.pycode_gen.gen_load_object(compiled_fn, compiled_fn_name)
        for name in input_names:
            found = False
            for variable in self.input_variables:
                if isinstance(variable, TensorVariable) and variable.get_symbol().name == name:
                    variable.tracker.gen_instructions(self.pycode_gen)
                    found = True
                    break
            assert found, f"can't find input {name} in SIR."
        self.pycode_gen.gen_build_tuple(count=len(input_names))
        self.pycode_gen.gen_call_function(argc=1)
        self.pycode_gen.gen_unpack_sequence(count=len(tensor_items))
        for tensor_var in tensor_items:
            self.pycode_gen.gen_store_fast(tensor_var.out_var_name)
        for ret_var in ret_vars:
            ret_var.reconstruct(self.pycode_gen)
        self.restore_inplace_tensor(self._inplace_tensors)
        self.restore_print_stmts(self._print_variables)
        self.restore_side_effects(self.side_effects.proxy_variables)
        self.pycode_gen.gen_enable_eval_frame()
        tracker_output_path = ENV_SHOW_TRACKERS.get()
        if tracker_output_path:
            from .tracker_viewer import view_tracker
            view_tracker(list(ret_vars), tracker_output_path, format='png')

    def call_paddle_api(self, func: Callable[..., Any], *args: VariableBase, **kwargs: VariableBase):
        if False:
            return 10
        '\n        Record Paddle Networking API to SIR\n\n        Args:\n            func: paddle api\n        '
        assert is_paddle_api(func)
        log(3, f'call paddle.api : {func.__name__}', '\n')

        def message_handler(*args, **kwargs):
            if False:
                return 10
            return f'Call paddle_api error: {func.__name__}, may be not a operator api ?'
        return inner_error_default_handler(self.symbolic_call, message_handler)(InferMetaCache(), self.sir_ctx.call_API, func, *args, **kwargs)

    def call_tensor_method(self, method_name: str, *args: VariableBase, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        call tensor method, start symbolic trace.\n\n        Args:\n            method_name: tensor method name\n        '

        def message_handler(*args, **kwargs):
            if False:
                print('Hello World!')
            return f'Call tensor_method error: Tensor.{method_name}, may be not a valid operator api ?'
        return inner_error_default_handler(self.symbolic_call, message_handler)(InferMetaCache(), self.sir_ctx.call_METHOD, method_name, *args, **kwargs)

    @staticmethod
    def get_opcode_executor_stack():
        if False:
            i = 10
            return i + 15
        from .opcode_executor import OpcodeExecutorBase
        if len(OpcodeExecutorBase.call_stack) == 0:
            return []
        current_executor = OpcodeExecutorBase.call_stack[-1]
        current_line = current_executor._current_line
        filename = current_executor._code.co_filename
        (source_lines, start_line) = inspect.getsourcelines(current_executor._code)
        line_idx = min(current_line - start_line, len(source_lines) - 1)
        code_line = source_lines[line_idx]
        stack = []
        stack.append('  File "{}", line {}, in {}'.format(filename, current_line, current_executor._code.co_name))
        stack.append(f'    {code_line}')
        return stack

    def call_layer(self, layer: PaddleLayerVariable, weak_ref: bool, *args: VariableBase, **kwargs: VariableBase):
        if False:
            i = 10
            return i + 15
        '\n        call paddle layer, start symbolic trace.\n\n        Args:\n            layer: paddle layer\n        '

        def infer_meta_fn(layer, *metas, **kwmetas):
            if False:
                print('Hello World!')
            metas = LayerInferMetaCache()(layer.value, *metas, **kwmetas)
            return metas

        def compute_fn(layer, inputs, outputs, stacks):
            if False:
                i = 10
                return i + 15
            self.sir_ctx.call_LAYER(Reference(layer.value, weak_ref), inputs=inputs, outputs=outputs, stacks=stacks)

        def message_handler(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return f'Call paddle layer error: {layer}, may be not a valid paddle layer ?'
        return inner_error_default_handler(self.symbolic_call, message_handler)(infer_meta_fn, compute_fn, layer, *args, **kwargs)

    def symbolic_call(self, infer_meta_fn, compute_fn, func, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Using infer_meta_fn and compute_fn convert func to symbolic function.\n\n        Args:\n            infer_meta_fn: function for infer meta, (func, metas, kwmetas) -> output_metas\n            compute_fn   : function for sir compile, (func, input_symbols, outputs_symbols) -> None\n            func         : symbolic function\n        '
        self.collect_input_variables(list(args))
        self.collect_input_variables(list(kwargs.values()))
        metas = convert_to_meta(args)
        kwmetas = convert_to_meta(kwargs)
        out_metas = infer_meta_fn(func, *metas, **kwmetas)
        inputs_symbols = (convert_to_symbol(args), convert_to_symbol(kwargs))
        log(3, f'         inputs : {inputs_symbols}', '\n')
        outputs = map_if(out_metas, pred=lambda x: isinstance(x, MetaInfo), true_fn=lambda x: TensorVariable(x, self, tracker=DummyTracker(list(args) + list(kwargs.values()))), false_fn=lambda x: x)
        stmt_stacks = []
        log_do(3, lambda : stmt_stacks.extend(FunctionGraph.get_opcode_executor_stack()))
        if outputs is not None:
            if is_inplace_api(func):
                compute_fn(func, inputs_symbols, convert_to_symbol(args[0]), stmt_stacks)
            else:
                compute_fn(func, inputs_symbols, convert_to_symbol(outputs), stmt_stacks)
                self._put_inner(outputs)
            return VariableFactory.from_value(outputs, self, DummyTracker(list(args) + list(kwargs.values())))
        else:
            return None

    def _put_inner(self, vars: VariableBase):
        if False:
            return 10
        '\n        put inner variable to inner_out\n        '
        map_if(vars, pred=lambda x: isinstance(x, VariableBase), true_fn=lambda x: self.inner_out.add(x.id), false_fn=lambda x: None)

    def add_global_guarded_variable(self, variable: VariableBase):
        if False:
            i = 10
            return i + 15
        '\n        Add variable to global guarded variable\n        '
        self._global_guarded_variables.add(variable)

    def remove_global_guarded_variable(self, variable: VariableBase):
        if False:
            while True:
                i = 10
        '\n        Remove variable to global guarded variable\n        '
        if variable in self._global_guarded_variables:
            self._global_guarded_variables.remove(variable)

    def _find_tensor_outputs(self, outputs: list[VariableBase]) -> OrderedSet[TensorVariable]:
        if False:
            while True:
                i = 10
        '\n        Return all TensorVariable. find TensorVariables participating in networking from the output Variables\n\n        Args:\n            outputs: output variables\n        '

        def collect_related_dummy_tensor(var):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(var.tracker, DummyTracker):
                if isinstance(var, TensorVariable):
                    return [var]
                else:
                    retval = []
                    for inp in var.tracker.inputs:
                        retval.extend(collect_related_dummy_tensor(inp))
                    return retval
            return []
        output_tensors: OrderedSet[TensorVariable] = OrderedSet()
        for output in outputs:
            if isinstance(output.tracker, DummyTracker):
                if isinstance(output, TensorVariable):
                    output_tensors.add(output)
                else:
                    for inp in output.tracker.inputs:
                        for _var in collect_related_dummy_tensor(inp):
                            output_tensors.add(_var)
                    self.add_global_guarded_variable(output)
        for side_effect_var in self.side_effects.proxy_variables:
            if isinstance(side_effect_var, (ListVariable, DictVariable)):
                for var in side_effect_var.flatten_items():
                    if isinstance(var.tracker, DummyTracker) and isinstance(var, TensorVariable) and side_effect_var.tracker.is_traceable():
                        output_tensors.add(var)
            else:
                if isinstance(side_effect_var, GlobalVariable):
                    proxy_records = side_effect_var.proxy.records
                elif side_effect_var.tracker.is_traceable():
                    proxy_records = side_effect_var.attr_proxy.records
                else:
                    continue
                for record in proxy_records:
                    if isinstance(record, (MutationSet, MutationNew)):
                        for var in record.value.flatten_items():
                            if isinstance(var.tracker, DummyTracker) and isinstance(var, TensorVariable):
                                output_tensors.add(var)
        for print_stmt in self._print_variables:
            for var in print_stmt.flatten_items():
                if isinstance(var.tracker, DummyTracker) and isinstance(var, TensorVariable):
                    output_tensors.add(var)
        for inplace_tensor in self._inplace_tensors:
            output_tensors.add(inplace_tensor)
        return output_tensors

    def restore_print_stmts(self, variables: list[VariableBase]):
        if False:
            while True:
                i = 10
        for var in variables:
            var.reconstruct(self.pycode_gen, use_tracker=False, add_to_global_guarded_vars=False)

    def restore_inplace_tensor(self, variables: list[VariableBase]):
        if False:
            print('Hello World!')
        for var in variables:
            if not var.tracker.is_traceable():
                continue
            var.reconstruct(self.pycode_gen, use_tracker=True, add_to_global_guarded_vars=False)
            self.pycode_gen.gen_load_method('_inplace_assign')
            var.reconstruct(self.pycode_gen, use_tracker=False, add_to_global_guarded_vars=True)
            self.pycode_gen.gen_call_method(1)
            self.pycode_gen.gen_pop_top()

    def restore_side_effects(self, variables: list[VariableBase]):
        if False:
            return 10
        '\n        Generate side effect recovery code for variables with side effects\n\n        Args:\n            variables: Variables that may have side effects.\n        '
        restorers: list[SideEffectRestorer] = []
        for var in variables:
            if not var.tracker.is_traceable() and (not isinstance(var, GlobalVariable)):
                continue
            if isinstance(var, DictVariable):
                restorers.append(DictSideEffectRestorer(var))
            elif isinstance(var, ListVariable):
                restorers.append(ListSideEffectRestorer(var))
            elif isinstance(var, GlobalVariable):
                for record in var.proxy.records[::-1]:
                    if isinstance(record, (MutationSet, MutationNew)):
                        restorers.append(GlobalSetSideEffectRestorer(record.key, record.value))
                    elif isinstance(record, MutationDel):
                        restorers.append(GlobalDelSideEffectRestorer(record.key))
            else:
                for record in var.attr_proxy.records[::-1]:
                    if isinstance(record, (MutationSet, MutationNew)):
                        restorers.append(ObjSetSideEffectRestorer(var, record.key, record.value))
                    elif isinstance(record, MutationDel):
                        restorers.append(ObjDelSideEffectRestorer(var, record.key))
        for restorer in restorers:
            restorer.pre_gen(self.pycode_gen)
        for restorer in restorers[::-1]:
            restorer.post_gen(self.pycode_gen)