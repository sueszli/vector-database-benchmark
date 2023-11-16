from copy import deepcopy
import numpy as np
import paddle
from paddle import _legacy_C_ops
from paddle.amp.auto_cast import _in_amp_guard, _in_pure_fp16_guard
from paddle.base import backward, core, framework, program_guard
from paddle.base.compiler import BuildStrategy
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.base.framework import _apply_pass, get_flags
from paddle.base.unique_name import guard as UniqueNameGuard
from paddle.optimizer.lr import LRScheduler
from . import logging_utils
from .utils import RETURN_NO_VALUE_MAGIC_NUM, backend_guard, construct_grad_names, tensor_name_guard
__all__ = []

class NestSequence:
    """
    A wrapper class that easily to flatten and restore the nest structure of
    given sequence.
    """

    def __init__(self, raw_input, need_check=False):
        if False:
            print('Hello World!')
        self.__raw_input = raw_input
        self.__input_list = self.tolist()
        self.__var_ids = self._get_var_ids()
        self._check_non_variable(need_check)

    def tolist(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Flattens the nested sequences into single list.\n        '
        return paddle.utils.flatten(self.__raw_input)

    def restore(self, value_list):
        if False:
            return 10
        '\n        Restores the nested sequence from value list.\n        '
        assert len(self.__input_list) == len(value_list)
        return paddle.utils.pack_sequence_as(self.__raw_input, value_list)

    def _get_var_ids(self):
        if False:
            while True:
                i = 10
        var_ids = []
        for (idx, var) in enumerate(self.__input_list):
            if isinstance(var, (framework.Variable, core.eager.Tensor)):
                var_ids.append(idx)
        return var_ids

    def _check_non_variable(self, need_check):
        if False:
            return 10
        '\n        Raises warning if output of traced function contains non-tensor type values.\n        '
        if need_check:
            warning_types = set()
            for var in self.__input_list:
                if not isinstance(var, (framework.Variable, core.eager.Tensor)):
                    warning_types.add(type(var))
            if warning_types:
                logging_utils.warn("Output of traced function contains non-tensor type values: {}. Currently, We don't support to update them while training and will return what we first saw. Please try to return them as tensor.".format(list(warning_types)))

    @property
    def var_ids(self):
        if False:
            i = 10
            return i + 15
        return self.__var_ids

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return self.__input_list[item]

class LazyInitialized:
    """
    Descriptor to implement lazy initialization of property.
    """

    def __init__(self, function):
        if False:
            while True:
                i = 10
        self.function = function

    def __get__(self, instance, cls):
        if False:
            for i in range(10):
                print('nop')
        val = self.function(instance)
        setattr(instance, self.function.__name__, val)
        return val

class ProgramInfo:
    """
    A helper class to recoder Program information
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_size = {'fp32': -1, 'amp': -1, 'fp16': -1}
        self.programs = {}
        self.mode = 'infer'

    def __call__(self, key, prog_creator):
        if False:
            i = 10
            return i + 15
        '\n        Recoder infer program and op size.\n        '
        assert key in ['fp32', 'amp', 'fp16']
        if key not in self.programs:
            infer_prog = prog_creator(is_infer_mode=True)
            self.programs[key] = infer_prog
            self.op_size[key] = infer_prog.desc.block(0).op_size()
        return (self.programs[key], self.op_size[key])

class PartialProgramLayerHook:

    def before_append_backward(self, forward_program):
        if False:
            while True:
                i = 10
        ...

    def after_append_backward(self, whole_program, backward_start_idx):
        if False:
            print('Hello World!')
        ...

    def after_infer(self, infer_program):
        if False:
            for i in range(10):
                print('nop')
        ...

class PartialProgramLayer:
    """
    PartialProgramLayer wraps all the ops from layers decorated by `@to_static`
    and execute them as a static subgraph.

    .. note::
        **1. This is a very low level API. Users should not use this API
             directly. Please use `partial_program_from(concrete_program)`
             to create it.
        **2. LoDTensorArray is not currently supported in the output.

    Args:
        main_program(Program): The main program that contains ops need to be executed.
        inputs(list[Variable]): The input list of the decorated function by `@to_static`.
        outputs(list[Variable]): The output list of the decorated function by `@to_static`.
        parameters(list[Tensor]|None): All trainable parameters included in the program. Default None.

    Returns:
        Layer: A Layer object that run all ops internally in static graph mode.
    """

    def __init__(self, main_program, inputs, outputs, name_generator, parameters=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._inputs = NestSequence(inputs)
        self._outputs = NestSequence(outputs, need_check=True)
        self._params = parameters if parameters is not None else []
        self._name_generator = name_generator
        self._build_strategy = kwargs.get('build_strategy', BuildStrategy())
        assert isinstance(self._build_strategy, BuildStrategy)
        self._origin_main_program = self._verify_program(main_program)
        with paddle.base.framework._dygraph_guard(paddle.base.dygraph.Tracer()):
            self._cuda_graph_vec = self._create_cuda_graph_vec()
        self._cuda_graph_capture_mode = ''
        self._cuda_graph_pool_id = 0
        self.training = True
        self._infer_info = ProgramInfo()
        self._forward_end_index_map = {}
        (amp_dtype, custom_white_list, custom_black_list) = (None, None, None)
        tracer = framework._dygraph_tracer()
        if tracer:
            (custom_white_list, custom_black_list) = tracer._get_amp_op_list()
            amp_dtype = tracer._amp_dtype
        if amp_dtype is not None and amp_dtype in ['float16', 'bfloat16']:
            self._amp_list = paddle.static.amp.fp16_lists.AutoMixedPrecisionLists(custom_white_list=custom_white_list, custom_black_list=custom_black_list, dtype=amp_dtype)
        self._pir_scope_cache = {}
        self._legacy_scope_cache = {}
        self._hooker = None
        self._backend = kwargs.get('backend', None)
        self._grad_var_names = {}

    def __call__(self, inputs):
        if False:
            i = 10
            return i + 15
        '\n        Execute static graph by Interpreter and Return dynamic Tensors.\n        '
        with UniqueNameGuard(self._name_generator):
            (in_vars, out_vars, in_var_names) = self._prepare(inputs)
            self._cast_fp16_if_pure_fp16(in_vars)
            attrs = self._prepare_attributes()
            attrs.extend(['x_names', in_var_names])
            self._sync_lr_value_with_scheduler()
            with tensor_name_guard(in_vars, in_var_names):
                _legacy_C_ops.run_program(self._valid_vars(in_vars), self._valid_vars(self._params), self._valid_vars(out_vars), self._create_scope_vec(program_id=self.program_id, use_scope_cache=True), self._cuda_graph_vec, *attrs)
            self._update_stop_gradient(out_vars)
            restored_nest_out = self._restore_out(out_vars)
            return self._remove_no_value(restored_nest_out)

    def _sync_lr_value_with_scheduler(self):
        if False:
            while True:
                i = 10
        'Update lr_var value with calculated by lr_scheduler.'
        main_program = self._origin_main_program
        if hasattr(main_program, 'lr_scheduler') and hasattr(main_program, 'lr_var'):
            lr_scheduler = main_program.lr_scheduler
            lr_var = main_program.lr_var
            assert isinstance(lr_scheduler, LRScheduler), 'must be LRScheduler'
            lr_scheduler = self._origin_main_program.lr_scheduler
            lr_value = lr_scheduler()
            data = np.array(lr_value).astype(convert_dtype(lr_var.dtype))
            lr_var.set_value(data)

    def set_hooker(self, hooker):
        if False:
            i = 10
            return i + 15
        self._hooker = hooker

    def _get_scope(self, program_id=None, use_scope_cache=False):
        if False:
            return 10
        if get_flags('FLAGS_enable_pir_in_executor')['FLAGS_enable_pir_in_executor']:
            _scope_cache = self._pir_scope_cache
        else:
            _scope_cache = self._legacy_scope_cache
        if not use_scope_cache:
            return core.Scope()
        if program_id not in _scope_cache:
            _scope_cache[program_id] = []
        cached_scopes = _scope_cache[program_id]
        for scope in cached_scopes:
            if scope._can_reused:
                return scope
        scope = core.Scope()
        cached_scopes.append(scope)
        return scope

    @switch_to_static_graph
    def _create_program(self, is_infer_mode=False):
        if False:
            while True:
                i = 10
        if is_infer_mode:
            infer_program = self._origin_main_program.clone(for_test=is_infer_mode)
            if self._hooker:
                infer_program = self._hooker.after_infer(infer_program)
            return infer_program
        else:
            train_program = self._append_backward_desc(self._origin_main_program)
            self._set_grad_type(self._params, train_program)
            return train_program

    @switch_to_static_graph
    def _create_amp_program(self, is_infer_mode=False):
        if False:
            return 10
        amp_program = self._origin_main_program.clone(for_test=is_infer_mode)
        with program_guard(amp_program):
            paddle.static.amp.fp16_utils.cast_model_to_fp16(amp_program, self._amp_list, use_fp16_guard=False, level='O1')
        if is_infer_mode:
            if self._hooker:
                amp_program = self._hooker.after_infer(amp_program)
            return amp_program
        else:
            train_amp_program = self._append_backward_desc(amp_program)
            self._set_grad_type(self._params, train_amp_program)
            return train_amp_program

    @switch_to_static_graph
    def _create_pure_fp16_program(self, is_infer_mode=False):
        if False:
            for i in range(10):
                print('nop')
        pure_fp16_program = self._origin_main_program.clone(for_test=is_infer_mode)
        with program_guard(pure_fp16_program):
            paddle.static.amp.fp16_utils.cast_model_to_fp16(pure_fp16_program, self._amp_list, use_fp16_guard=False)
        if is_infer_mode:
            if self._hooker:
                pure_fp16_program = self._hooker.after_infer(pure_fp16_program)
            return pure_fp16_program
        else:
            train_pure_fp16_program = self._append_backward_desc(pure_fp16_program)
            self._set_grad_type(self._params, train_pure_fp16_program)
            return train_pure_fp16_program

    @switch_to_static_graph
    def _create_forward_backward_train_program(self):
        if False:
            return 10
        whole_program = self._train_program
        forward_end_op_index = self.get_forward_end_op_idx(whole_program)
        assert forward_end_op_index >= 0
        return self._get_forward_backward_program_form(whole_program, forward_end_op_index)

    @switch_to_static_graph
    def _create_forward_backward_train_amp_program(self):
        if False:
            return 10
        whole_program = self._train_amp_program
        forward_end_op_index = self.get_forward_end_op_idx(whole_program)
        assert forward_end_op_index >= 0
        return self._get_forward_backward_program_form(whole_program, forward_end_op_index)

    @switch_to_static_graph
    def _create_forward_backward_train_pure_fp16_program(self):
        if False:
            for i in range(10):
                print('nop')
        whole_program = self._train_pure_fp16_program
        forward_end_op_index = self.get_forward_end_op_idx(whole_program)
        assert forward_end_op_index >= 0
        return self._get_forward_backward_program_form(whole_program, forward_end_op_index)

    @LazyInitialized
    def _train_program(self):
        if False:
            print('Hello World!')
        return self._create_program()

    @LazyInitialized
    def _infer_program(self):
        if False:
            i = 10
            return i + 15
        (program, op_size) = self._infer_info('fp32', self._create_program)
        return self._build_infer_program(program, op_size)

    @LazyInitialized
    def _train_amp_program(self):
        if False:
            i = 10
            return i + 15
        return self._create_amp_program()

    @LazyInitialized
    def _infer_amp_program(self):
        if False:
            while True:
                i = 10
        (program, op_size) = self._infer_info('amp', self._create_amp_program)
        return self._build_infer_program(program, op_size)

    @LazyInitialized
    def _train_pure_fp16_program(self):
        if False:
            return 10
        return self._create_pure_fp16_program()

    @LazyInitialized
    def _infer_pure_fp16_program(self):
        if False:
            while True:
                i = 10
        (program, op_size) = self._infer_info('fp16', self._create_pure_fp16_program)
        return self._build_infer_program(program, op_size)

    @LazyInitialized
    def _train_forward_backward_program(self):
        if False:
            i = 10
            return i + 15
        program = self._create_forward_backward_train_program()
        return program

    @LazyInitialized
    def _train_amp_forward_backward_program(self):
        if False:
            i = 10
            return i + 15
        program = self._create_forward_backward_train_amp_program()
        return program

    @LazyInitialized
    def _empty_backward_program_for_eval(self):
        if False:
            i = 10
            return i + 15
        return paddle.static.Program()

    @LazyInitialized
    def _train_pure_fp16_forward_backward_program(self):
        if False:
            return 10
        program = self._create_forward_backward_train_pure_fp16_program()
        return program

    @LazyInitialized
    def _train_program_id(self):
        if False:
            i = 10
            return i + 15
        program_id = paddle.utils._hash_with_id(self._train_program, self)
        core._set_cached_executor_build_strategy(program_id, self._build_strategy)
        return program_id

    @LazyInitialized
    def _infer_program_id(self):
        if False:
            i = 10
            return i + 15
        return paddle.utils._hash_with_id(self._infer_program, self)

    @LazyInitialized
    def _train_amp_program_id(self):
        if False:
            print('Hello World!')
        program_id = paddle.utils._hash_with_id(self._train_amp_program, self)
        core._set_cached_executor_build_strategy(program_id, self._build_strategy)
        return program_id

    @LazyInitialized
    def _infer_amp_program_id(self):
        if False:
            return 10
        return paddle.utils._hash_with_id(self._infer_amp_program, self)

    @LazyInitialized
    def _train_pure_fp16_program_id(self):
        if False:
            i = 10
            return i + 15
        program_id = paddle.utils._hash_with_id(self._train_pure_fp16_program, self)
        core._set_cached_executor_build_strategy(program_id, self._build_strategy)
        return program_id

    @LazyInitialized
    def _infer_pure_fp16_program_id(self):
        if False:
            i = 10
            return i + 15
        return paddle.utils._hash_with_id(self._infer_pure_fp16_program, self)

    def get_forward_end_op_idx(self, program):
        if False:
            while True:
                i = 10
        return self._forward_end_index_map[paddle.utils._hash_with_id(program, self)]

    @property
    def program(self):
        if False:
            while True:
                i = 10
        '\n        Return current train or eval program.\n        '
        if self.training:
            return self.train_program
        else:
            return self.infer_program

    @property
    def program_id(self):
        if False:
            return 10
        '\n        Return current train or eval program hash id.\n        '
        if self.training:
            if _in_amp_guard():
                return self._train_amp_program_id
            elif _in_pure_fp16_guard():
                return self._train_pure_fp16_program_id
            else:
                return self._train_program_id
        elif _in_amp_guard():
            return self._infer_amp_program_id
        elif _in_pure_fp16_guard():
            return self._infer_pure_fp16_program_id
        else:
            return self._infer_program_id

    @property
    def train_program(self):
        if False:
            print('Hello World!')
        if _in_amp_guard():
            return self._train_amp_program
        elif _in_pure_fp16_guard():
            return self._train_pure_fp16_program
        else:
            return self._train_program

    @property
    def infer_program(self):
        if False:
            print('Hello World!')
        if _in_amp_guard():
            return self._infer_amp_program
        elif _in_pure_fp16_guard():
            return self._infer_pure_fp16_program
        else:
            return self._infer_program

    @property
    def forward_program(self):
        if False:
            i = 10
            return i + 15
        if self.training:
            if _in_amp_guard():
                progs = self._train_amp_forward_backward_program
            elif _in_pure_fp16_guard():
                progs = self._train_pure_fp16_forward_backward_program
            else:
                progs = self._train_forward_backward_program
            return progs[0]
        else:
            return self.infer_program

    @property
    def backward_program(self):
        if False:
            print('Hello World!')
        if self.training:
            if _in_amp_guard():
                progs = self._train_amp_forward_backward_program
            elif _in_pure_fp16_guard():
                progs = self._train_pure_fp16_forward_backward_program
            else:
                progs = self._train_forward_backward_program
            return progs[1]
        else:
            "\n            Can't just return paddle.static.Program(), because self.backward_program is a property,\n            whenever we call this method, a tmp Program() object is created and is gc immediatly\n            after executed the following line in PartialProgramLayer.__call__.\n\n            >>> self.backward_program.desc.block(0),\n\n            When we access RunProgramAPI, it's possible to get an invalid backward_program address.\n            "
            return self._empty_backward_program_for_eval

    def _verify_program(self, main_program):
        if False:
            print('Hello World!')
        '\n        Verify that the program parameter is initialized, prune some unused params,\n        and remove redundant op callstack.\n        '
        self._check_params_all_inited(main_program)
        self._prune_unused_params(main_program)
        return main_program

    def prepare_gradient_aggregation(self, start_idx, main_program, target_program):
        if False:
            for i in range(10):
                print('nop')
        '\n        Why we need add gradient aggregation operation ?\n        In some cases, if non leaf nodes are used as output, gradient overwriting will occur, such as\n        def forward(self, in):\n            x = 2 * in  # <---- x is a non-leaf node in program.\n            y = x + 3\n            return x, y\n\n        loss = forward(in)[0].sum()\n        loss.backward()  # <----- x@grad will be overwrited by elementwise_add_grad Op\n        '

        def _need_aggregation(var):
            if False:
                i = 10
                return i + 15
            '\n            if exist a op whose inputs is var, then return True\n            '
            if not isinstance(var, framework.Variable) or var.type not in [core.VarDesc.VarType.LOD_TENSOR, core.VarDesc.VarType.SELECTED_ROWS]:
                return False
            if var.dtype not in [paddle.float32, paddle.float64]:
                return False
            for op in main_program.block(0).ops:
                for in_arg in op.input_arg_names:
                    if in_arg == var.name:
                        return True
            return False

        def _insert_aggregation_ops_for_var(target_program, var):
            if False:
                while True:
                    i = 10
            suffix = '@dy2static'
            var_grad_name = var.grad_name
            new_grad_name = var.name + suffix + '@GRAD'
            finded_ops = list(filter(lambda x: x[0] >= start_idx and any((out_arg == var_grad_name for out_arg in x[1].output_arg_names)), enumerate(target_program.block(0).ops)))
            if len(finded_ops) == 0:
                return None
            target_program.block(0).create_var(name=new_grad_name, type=var.type, dtype=var.dtype, shape=var.shape)
            for (idx, op) in finded_ops:
                op._rename_input(var_grad_name, new_grad_name)
                op._rename_output(var_grad_name, new_grad_name)
            target_program.block(0)._insert_op(finded_ops[-1][0] + 1, type='sum', inputs={'X': [var_grad_name, new_grad_name]}, outputs={'Out': var_grad_name})
            return None
        to_processed_vars = list(filter(_need_aggregation, self._outputs.tolist()))
        for _var in to_processed_vars:
            target_program: paddle.static.Program
            target_var = target_program.global_block().var(_var.name)
            _insert_aggregation_ops_for_var(target_program, target_var)

    @switch_to_static_graph
    def _append_backward_desc(self, main_program):
        if False:
            while True:
                i = 10
        program = main_program.clone(for_test=False)
        if self._hooker:
            program = self._hooker.before_append_backward(program)
        targets = []
        for out in self._outputs.tolist():
            if isinstance(out, framework.Variable):
                targets.append(program.global_block().var(out.name))
        start_idx = len(program.block(0).ops) + len(self._outputs.tolist())
        if targets:
            start_idx = len(program.block(0).ops) + len(self._outputs.tolist())
            with backend_guard(self._backend):
                check_type(targets, 'targets', (framework.Variable, list, tuple), 'paddle.static.gradients')
                grad_info_map = backward.calc_gradient_helper(targets=targets, inputs=[])
                x_vars = [program.block(0).var(var.name) for var in self._inputs if isinstance(var, framework.Variable)]
                param_vars = [program.block(0).var(param.name) for param in self._params]
                out_vars = [program.block(0).var(var.name) for var in self._outputs if isinstance(var, framework.Variable)]
                self._grad_var_names = construct_grad_names(grad_info_map, x_vars, param_vars, out_vars)
            if self._hooker:
                (program, start_idx) = self._hooker.after_append_backward(program, start_idx)
            self.prepare_gradient_aggregation(start_idx + 1, main_program, program)
        self._forward_end_index_map[paddle.utils._hash_with_id(program, self)] = start_idx - len(self._outputs.tolist())
        return program

    def _prune_unused_params(self, program):
        if False:
            i = 10
            return i + 15
        '\n        Prune the parameters not used anywhere in the program.\n        The `@to_static` may only decorated a sub function which\n        contains some unused parameters created in `__init__`.\n        So prune these parameters to avoid unnecessary operations in\n        `run_program_op`.\n        '
        required_params = []
        for param in self._params:
            found_param = False
            for block in program.blocks:
                for op in block.ops:
                    if param.name in op.input_arg_names or param.name in op.output_arg_names:
                        required_params.append(param)
                        found_param = True
                        break
                if found_param:
                    break
        self._params = required_params

    def _cast_fp16_if_pure_fp16(self, in_vars):
        if False:
            i = 10
            return i + 15
        if _in_pure_fp16_guard():
            for (i, var) in enumerate(in_vars):
                name = var.name
                if self.program.global_block().has_var(name) and self.program.global_block().var(name).dtype == paddle.float16:
                    in_vars[i] = var.astype('float16')
                    in_vars[i].name = name

    def _prepare_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        attrs = ['forward_global_block', self.forward_program.desc.block(0), 'backward_global_block', self.backward_program.desc.block(0), 'is_test', not self.training, 'program_id', self.program_id]
        if self.training:
            attrs.extend(('param_grad_names', self._grad_var_names.get('param', []), 'out_grad_names', self._grad_var_names.get('out', []), 'x_grad_names', self._grad_var_names.get('x', [])))
        if self._cuda_graph_capture_mode:
            attrs.extend(('cuda_graph_capture_mode', self._cuda_graph_capture_mode, 'cuda_graph_pool_id', self._cuda_graph_pool_id))
        return attrs

    @switch_to_static_graph
    def _build_infer_program(self, infer_program, forward_end_op_index):
        if False:
            for i in range(10):
                print('nop')
        forward_skip_vars = self._parse_skip_gc_vars(infer_program)
        builded_infer_program = add_build_strategy_for(infer_program, 0, forward_end_op_index, self._build_strategy, forward_skip_vars)
        self._apply_inplace_pass(builded_infer_program, None)
        return builded_infer_program

    @switch_to_static_graph
    def _get_forward_backward_program_form(self, whole_program, forward_end_op_index):
        if False:
            for i in range(10):
                print('nop')
        backward_start_op_index = forward_end_op_index + len(self._outputs.var_ids)
        backward_end_op_index = whole_program.desc.block(0).op_size()
        backward_skip_vars = self._parse_skip_gc_vars(whole_program) + self._grad_var_names.get('param', [])
        backward_builded_program = add_build_strategy_for(whole_program, backward_start_op_index, backward_end_op_index, self._build_strategy, backward_skip_vars)
        forward_skip_vars = self._parse_skip_gc_vars(whole_program, backward_builded_program)
        forward_builded_program = add_build_strategy_for(whole_program, 0, forward_end_op_index, self._build_strategy, forward_skip_vars)
        self._apply_inplace_pass(forward_builded_program, backward_builded_program)
        return [forward_builded_program, backward_builded_program]

    def _apply_inplace_pass(self, forward_program, backward_program):
        if False:
            while True:
                i = 10
        attr_types = {'use_cuda': 'bool', 'mem_opt_skip_vars': 'list[str]', 'for_partial_block': 'bool'}
        empty_startup_program = paddle.static.Program()
        use_cuda = True if core.is_compiled_with_cuda() else False
        forward_mem_opt_skip_vars = self._parse_skip_gc_vars(forward_program, backward_program)
        backward_mem_opt_skip_vars = self._parse_skip_gc_vars(forward_program)
        if forward_program:
            attrs = {'use_cuda': use_cuda, 'mem_opt_skip_vars': forward_mem_opt_skip_vars, 'for_partial_block': True}
            if not get_flags('FLAGS_enable_pir_in_executor')['FLAGS_enable_pir_in_executor']:
                _apply_pass(forward_program, empty_startup_program, 'buffer_shared_inplace_pass', attrs, attr_types)
        if backward_program:
            attrs = {'use_cuda': use_cuda, 'mem_opt_skip_vars': backward_mem_opt_skip_vars, 'for_partial_block': True}
            if not get_flags('FLAGS_enable_pir_in_executor')['FLAGS_enable_pir_in_executor']:
                _apply_pass(backward_program, empty_startup_program, 'buffer_shared_inplace_pass', attrs, attr_types)

    @LazyInitialized
    def _inout_var_names(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns Variable Names from self._inputs and self.outputs\n        '
        var_names = []
        for var in self._inputs:
            if isinstance(var, paddle.base.framework.Variable):
                var_names.append(var.desc.name())
        for var in self._outputs:
            if isinstance(var, paddle.base.framework.Variable):
                var_names.append(var.desc.name())
        return var_names

    def _parse_skip_gc_vars(self, program, backward_program=None):
        if False:
            i = 10
            return i + 15
        '\n        Parse variables that need to skip GC after execute it.\n        If specify backward_program, it will keep the variables used in backward.\n        '
        skip_vars = deepcopy(self._inout_var_names)
        for (var_name, var) in program.global_block().vars.items():
            if var.is_data:
                skip_vars.append(var_name)
        if backward_program:
            for var_name in core.parse_safe_eager_deletion_skip_vars(backward_program.desc, True):
                skip_vars.append(var_name)
        return skip_vars

    def _prepare(self, inputs):
        if False:
            i = 10
            return i + 15
        '\n        Prepare inputs, outputs, attrs.\n        '
        assert isinstance(inputs, (tuple, list))
        flatten_inputs = paddle.utils.flatten(inputs)
        input_vars = []
        input_var_names = []
        expected_place = framework._current_expected_place()
        for (i, value) in enumerate(flatten_inputs):
            if isinstance(value, np.ndarray):
                var = None
                var = core.eager.Tensor(value=value, name=self._inputs[i].desc.name(), persistable=False, place=expected_place, zero_copy=True)
            elif isinstance(value, core.eager.Tensor):
                if value.stop_gradient and (not value.place._equals(expected_place)):
                    var = value._copy_to(expected_place, False)
                    var.stop_gradient = True
                else:
                    var = value
            else:
                continue
            input_var_names.append(self._inputs[i].desc.name())
            input_vars.append(var)
        out_tensor_map = {}

        def create_out(var_id):
            if False:
                i = 10
                return i + 15
            var = self._outputs[var_id]
            assert isinstance(var, framework.Variable)
            var_desc = var.desc
            if var_desc.name() in out_tensor_map:
                return out_tensor_map[var_desc.name()]
            out = core.eager.Tensor(var_desc.dtype(), var_desc.shape(), var_desc.name(), var_desc.type(), False)
            out.stop_gradient = var.stop_gradient
            out_tensor_map[var_desc.name()] = out
            return out
        out_vars = list(map(create_out, self._outputs.var_ids))
        return (input_vars, out_vars, input_var_names)

    def _create_scope_vec(self, program_id=None, use_scope_cache=False):
        if False:
            return 10
        inner_scope = self._get_scope(program_id=program_id, use_scope_cache=use_scope_cache)
        return [inner_scope]

    def _create_cuda_graph_vec(self):
        if False:
            return 10
        var = core.eager.Tensor(core.VarDesc.VarType.FP32, [], 'cuda_graph', core.VarDesc.VarType.RAW, True)
        var.stop_gradient = True
        return var

    def _update_stop_gradient(self, out_vars):
        if False:
            i = 10
            return i + 15

        def set_stop_gradient(var_id, eager_tensor):
            if False:
                print('Hello World!')
            var = self._outputs[var_id]
            assert isinstance(var, framework.Variable)
            eager_tensor.stop_gradient = var.stop_gradient
        for (idx, var) in zip(self._outputs.var_ids, out_vars):
            set_stop_gradient(idx, var)

    def _restore_out(self, out_vars):
        if False:
            i = 10
            return i + 15
        '\n        Restores same nested outputs by only replacing the Variable with Tensor.\n        '
        flatten_outputs = self._outputs.tolist()
        for (i, idx) in enumerate(self._outputs.var_ids):
            flatten_outputs[idx] = out_vars[i]
        outs = self._outputs.restore(flatten_outputs)
        if outs is not None and len(outs) == 1:
            outs = outs[0]
        return outs

    @switch_to_static_graph
    def _clone_for_test(self, main_program):
        if False:
            while True:
                i = 10
        return main_program.clone(for_test=True)

    def _is_no_value(self, var):
        if False:
            print('Hello World!')
        if isinstance(var, core.eager.Tensor) and var.shape == [1]:
            if var.numpy()[0] == RETURN_NO_VALUE_MAGIC_NUM:
                return True
        return False

    def _remove_no_value(self, out_vars):
        if False:
            i = 10
            return i + 15
        '\n        Removes invalid value for various-length return statement\n        '
        if isinstance(out_vars, core.eager.Tensor):
            if self._is_no_value(out_vars):
                return None
            return out_vars
        elif isinstance(out_vars, (tuple, list)):
            if isinstance(out_vars, tuple):
                res = tuple((var for var in out_vars if not self._is_no_value(var)))
            else:
                res = [var for var in out_vars if not self._is_no_value(var)]
            has_removed = len(out_vars) > len(res)
            if len(res) == 0 and has_removed:
                return None
            elif len(res) == 1 and has_removed:
                return res[0]
            return res
        return out_vars

    def _set_grad_type(self, params, train_program):
        if False:
            return 10
        for param in params:
            grad_name = param.name + core.grad_var_suffix()
            grad_var = train_program.desc.block(0).find_var(grad_name.encode())
            if grad_var is None:
                continue
            param._set_grad_type(grad_var.type())

    def _check_params_all_inited(self, main_program):
        if False:
            print('Hello World!')
        '\n        Check all params from main program are already initialized, see details as follows:\n            1. all parameters in self._params should be type `framework.EagerParamBase` which are created in dygraph.\n            2. all parameters from transformed program can be found in self._params.\n               Because they share same data with EagerParamBase of original dygraph.\n        '
        if not isinstance(self._params, (list, tuple)):
            raise TypeError('Type of self._params in PartialProgramLayer should be list or tuple, but received %s.' % type(self._params))
        param_and_buffer_names_set = set()
        for (i, var) in enumerate(self._params):
            if not isinstance(var, core.eager.Tensor):
                raise TypeError('Type of self._params[{}] in PartialProgramLayer should be Parameter or Variable, but received {}.'.format(i, type(var)))
            param_and_buffer_names_set.add(var.name)
        for block in main_program.blocks:
            for (name, var) in block.vars.items():
                if isinstance(var, framework.Parameter):
                    if name not in param_and_buffer_names_set:
                        raise ValueError("\n\tWe don't support to define layer with parameters in the function decorated by `@to_static`.\n\tBut we found parameter(%s) was created in the decorated function.\n\n\tRevise suggestion: \n\t\t1. Please ensure all your sublayers are inheritted from nn.Layer.\n\t\t2. Please use nn.ParameterList and nn.LayerList as container instead of using a native Python container such as List" % name)

    def _valid_vars(self, vars):
        if False:
            i = 10
            return i + 15
        return vars if vars else None

def partial_program_from(concrete_program, from_method=False):
    if False:
        print('Hello World!')
    inputs = concrete_program.inputs
    if inputs and from_method:
        inputs = inputs[1:]
    return PartialProgramLayer(concrete_program.main_program, inputs, concrete_program.outputs, concrete_program.name_generator, concrete_program.parameters, **concrete_program.kwargs)

@switch_to_static_graph
def add_build_strategy_for(program, start_op_index, end_op_index, build_strategy=None, skip_vars=None):
    if False:
        for i in range(10):
            print('nop')
    if start_op_index < end_op_index:
        compiled_program = paddle.static.CompiledProgram(core.Graph(program.desc, start_op_index, end_op_index), build_strategy=build_strategy)
        if skip_vars:
            compiled_program._graph.set('skip_gc_vars', set(skip_vars))
        compiled_program._compile(core.Scope(), framework._current_expected_place())
        ir_graph = framework.IrGraph(compiled_program._graph)
        builded_program = ir_graph.to_program()
        if hasattr(compiled_program._program, 'lr_scheduler'):
            builded_program.lr_scheduler = compiled_program._program.lr_scheduler
    else:
        builded_program = paddle.static.Program()
        for var in program.block(0).vars.values():
            builded_program.block(0)._clone_variable(var, False)
    for (origin, current) in zip(program.blocks, builded_program.blocks):
        current.desc.set_parent_idx(origin.desc.parent)
    return builded_program