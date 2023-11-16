import os
import pickle
import numpy as np
import paddle
from paddle import _legacy_C_ops
from paddle.base import backward, core, framework, unique_name
from paddle.base.data_feeder import check_type
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.base.framework import OpProtoHolder
from paddle.framework import in_dynamic_mode
from paddle.jit.dy2static.partial_program import LazyInitialized, add_build_strategy_for
from paddle.jit.dy2static.utils import construct_grad_names
from paddle.nn.layer import layers
__all__ = []
INFER_MODEL_SUFFIX = '.pdmodel'
INFER_PARAMS_SUFFIX = '.pdiparams'
INFER_PARAMS_INFO_SUFFIX = '.pdiparams.info'
INFER_PROPERTY_SUFFIX = '.meta'
LOADED_VAR_SUFFIX = 'load'
PARAMETER_NAME_PREFIX = 'param'
BUFFER_NAME_PREFIX = 'buffer'

def _load_program_desc(model_file_path):
    if False:
        while True:
            i = 10
    with open(model_file_path, 'rb') as f:
        program_desc_str = f.read()
    program_desc = core.ProgramDesc(program_desc_str)
    if not core._is_program_version_supported(program_desc._version()):
        raise ValueError('Unsupported program version: %d\n' % program_desc._version())
    return program_desc

def _is_persistable(var_desc):
    if False:
        return 10
    if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or var_desc.type() == core.VarDesc.VarType.FETCH_LIST or var_desc.type() == core.VarDesc.VarType.READER or (var_desc.type() == core.VarDesc.VarType.RAW):
        return False
    return var_desc.persistable()

def _is_parameter(persistable_var_desc, program_desc):
    if False:
        i = 10
        return i + 15
    input_ops = []
    for block_idx in range(program_desc.num_blocks()):
        block = program_desc.block(block_idx)
        for op_idx in range(block.op_size()):
            op = block.op(op_idx)
            if persistable_var_desc.name() in op.input_arg_names():
                input_ops.append(op)
    for block_idx in range(program_desc.num_blocks()):
        block = program_desc.block(block_idx)
        for op_idx in range(block.op_size()):
            op = block.op(op_idx)
            if persistable_var_desc.name() in op.output_arg_names():
                if op in input_ops:
                    continue
                else:
                    return False
    return True

def _get_persistable_vars(program_desc):
    if False:
        i = 10
        return i + 15
    persistable_vars = []
    for i in range(program_desc.num_blocks()):
        block = program_desc.block(i)
        persistable_vars.extend(list(filter(_is_persistable, block.all_vars())))
    return persistable_vars

def _get_persistable_var_names(program_desc):
    if False:
        i = 10
        return i + 15
    '\n    Get all persistable variable names in ProgramDesc.\n    '
    var_names = []
    persistable_vars = _get_persistable_vars(program_desc)
    for var in persistable_vars:
        var_names.append(var.name())
    return var_names

def _get_all_var_names(program_desc):
    if False:
        for i in range(10):
            print('nop')
    all_var_names = set()
    for i in range(program_desc.num_blocks()):
        block = program_desc.block(i)
        for var in block.all_vars():
            all_var_names.add(var.name())
    return all_var_names

@switch_to_static_graph
def _append_loaded_suffix(name):
    if False:
        i = 10
        return i + 15
    '\n    Append loaded suffix to the given variable name\n    e.g. x ==> x.load_0, x.load_0 ==> x.load_0.load_0\n    '
    suffix = LOADED_VAR_SUFFIX
    new_name = unique_name.generate_with_ignorable_key('.'.join((name, suffix)))
    return new_name

@switch_to_static_graph
def _generate_unique_var_name(prefix):
    if False:
        print('Hello World!')
    return unique_name.generate_with_ignorable_key(prefix)

def _append_loaded_suffix_to_var(program_desc):
    if False:
        while True:
            i = 10
    suffix_varname_dict = {}
    persistable_vars = _get_persistable_vars(program_desc)
    for var_desc in persistable_vars:
        old_name = var_desc.name()
        new_name = _append_loaded_suffix(var_desc.name())
        suffix_varname_dict[new_name] = old_name
        var_desc.set_name(new_name)
        for block_idx in range(program_desc.num_blocks()):
            block = program_desc.block(block_idx)
            block._rename_var(old_name.encode(), new_name.encode())
            for op_idx in range(block.op_size()):
                op = block.op(op_idx)
                op._rename_input(old_name, new_name)
                op._rename_output(old_name, new_name)
    return suffix_varname_dict

@switch_to_static_graph
def _generate_unique_var_name_sync_with_main_program(prefix):
    if False:
        for i in range(10):
            print('nop')
    return unique_name.generate(prefix)

def _get_loaded_var_new_old(program_desc, all_new_old_dict_all):
    if False:
        return 10
    new_old_dict = {}
    persistable_vars = _get_persistable_vars(program_desc)
    for var_desc in persistable_vars:
        name_new = var_desc.name()
        new_old_dict[name_new] = all_new_old_dict_all[name_new]
    return new_old_dict

def _rename_var_program_desc(program_desc, include=None, exclude=None):
    if False:
        i = 10
        return i + 15
    "\n    Change the name of the loaded variables.Use 'unique_name.generate' to avoid duplication.\n    It is used when loading multiple program during inference.\n\n    e.g. linear_0.tmp_3 ==> linear_0.tmp_1, x ==> x_0. For double grad, x@GRAD ==> x_0@GRAD\n    If 'include' is not `None`,variables in include and the corresponding\n      double grad variables (if exist) are renamed.\n    If 'exclude' is not `None`,variables that are in exclude and the\n      corresponding double grad variables (if exist) are not renamed.\n\n    Args:\n        program_desc(ProgramDesc):the variables in it will be modified.\n        include(List):list of names of variables.\n        exclude(List):list of names of variables.\n\n    Returns:\n        tuple of (dict_rename_var_new_old, dict_rename_var_old_new)\n        dict_rename_var_new_old is a dict mapping from new name to old name\n        dict_rename_var_old_new is a dict mapping from old name to new name\n    "
    dict_rename_var_old_new = {}
    dict_rename_var_new_old = {}
    old_names = []
    for b_idx in range(program_desc.num_blocks()):
        cur_block = program_desc.block(b_idx)
        for var in cur_block.all_vars():
            old_names.append(var.name())
    has_double_grad = False
    for b_idx in range(program_desc.num_blocks()):
        cur_block = program_desc.block(b_idx)
        for (var_idx, var) in enumerate(cur_block.all_vars()):
            name_old = var.name()
            is_double_grad_var = '@GRAD' in name_old
            has_double_grad = has_double_grad or is_double_grad_var
            should_rename = (include is None or name_old in include) and (exclude is None or name_old not in exclude) and (not is_double_grad_var)
            if should_rename:
                temp_name = name_old.split('_')
                if len(temp_name) > 1 and temp_name[-1].isnumeric():
                    temp_name = '_'.join(temp_name[:-1])
                else:
                    temp_name = name_old
                while True:
                    name_new = _generate_unique_var_name_sync_with_main_program(temp_name)
                    if name_new not in old_names[:var_idx] + old_names[var_idx + 1:]:
                        break
            else:
                name_new = name_old
            if name_old != name_new:
                cur_block._rename_var(name_old.encode(), name_new.encode())
            if not is_double_grad_var:
                dict_rename_var_old_new[name_old] = name_new
                dict_rename_var_new_old[name_new] = name_old
    if has_double_grad:
        double_grad_rename_dict = {}
        for name_old in dict_rename_var_old_new:
            for b_idx in range(program_desc.num_blocks()):
                cur_block = program_desc.block(b_idx)
                for (var_idx, var) in enumerate(cur_block.all_vars()):
                    var_name = var.name()
                    if '@GRAD' in var_name and name_old in var_name:
                        new_var_name = var_name.replace(name_old, dict_rename_var_old_new[name_old])
                        double_grad_rename_dict[var_name] = new_var_name
        for var_name in double_grad_rename_dict:
            dict_rename_var_old_new[var_name] = double_grad_rename_dict[var_name]
            dict_rename_var_new_old[double_grad_rename_dict[var_name]] = var_name
    for b_idx in range(program_desc.num_blocks()):
        cur_block = program_desc.block(b_idx)
        for op_idx in range(cur_block.op_size()):
            op = cur_block.op(op_idx)
            for input_arg_name in op.input_arg_names():
                if input_arg_name in dict_rename_var_old_new:
                    if input_arg_name != dict_rename_var_old_new[input_arg_name]:
                        op._rename_input(input_arg_name, dict_rename_var_old_new[input_arg_name])
                        if cur_block.has_var(input_arg_name.encode()):
                            cur_block._rename_var(input_arg_name.encode(), dict_rename_var_old_new[input_arg_name].encode())
            for output_arg_name in op.output_arg_names():
                if output_arg_name in dict_rename_var_old_new:
                    if output_arg_name != dict_rename_var_old_new[output_arg_name]:
                        op._rename_output(output_arg_name, dict_rename_var_old_new[output_arg_name])
                        if cur_block.has_var(output_arg_name.encode()):
                            cur_block._rename_var(output_arg_name.encode(), dict_rename_var_old_new[output_arg_name].encode())
    program_desc.flush()
    return (dict_rename_var_new_old, dict_rename_var_old_new)

@switch_to_static_graph
def _build_program_by_desc(program_desc):
    if False:
        print('Hello World!')
    prog = framework.Program()
    prog.desc = program_desc
    prog.blocks = [framework.Block(prog, i) for i in range(prog.desc.num_blocks())]
    prog._sync_with_cpp()
    return prog

def _change_is_test_status(program_desc, is_test):
    if False:
        for i in range(10):
            print('nop')
    for i in range(program_desc.num_blocks()):
        block = program_desc.block(i)
        for j in range(block.op_size()):
            op = block.op(j)
            if op.has_attr('is_test'):
                op._set_attr('is_test', is_test)

class _ProgramHolder:
    """
    Holds the execution information of a Program.

    _ProgramHolder is the execution unit of TranslatedLayer,
    if TranslatedLayer contains multiple _ProgramHolder,
    it can execute multiple methods

    _ProgramHolder is an internal concept.
    """

    def __init__(self, program_desc):
        if False:
            print('Hello World!')
        super().__init__()
        self._input_descs = []
        self._output_descs = []
        self._persistable_names = []
        self._grad_var_names = {}
        self._inner_scope = core.Scope()
        self._suffix_varname_dict = None
        self._infer_program_desc = self._preprocess(program_desc)

    @switch_to_static_graph
    def _create_forward_train_program(self):
        if False:
            while True:
                i = 10
        whole_program = _build_program_by_desc(self.train_program)
        end_op_index = self._infer_program_desc.block(0).op_size()
        if end_op_index > 0:
            return add_build_strategy_for(whole_program, 0, end_op_index)
        else:
            return whole_program

    @LazyInitialized
    def _forward_program_desc(self):
        if False:
            print('Hello World!')
        return self._create_forward_train_program().desc

    @switch_to_static_graph
    def _create_backward_train_program(self):
        if False:
            return 10
        whole_program = _build_program_by_desc(self.train_program)
        start_op_index = self._infer_program_desc.block(0).op_size() + len(self._output_descs)
        end_op_index = whole_program.desc.block(0).op_size()
        if start_op_index < end_op_index:
            return add_build_strategy_for(whole_program, start_op_index, end_op_index)
        else:
            return paddle.static.Program()

    @LazyInitialized
    def _backward_program_desc(self):
        if False:
            for i in range(10):
                print('nop')
        return self._create_backward_train_program().desc

    @property
    def infer_program(self):
        if False:
            print('Hello World!')
        return self._infer_program_desc

    @LazyInitialized
    def train_program(self):
        if False:
            return 10
        return self._append_backward_desc(self._infer_program_desc)

    @property
    def forward_program(self):
        if False:
            return 10
        return self._forward_program_desc

    @property
    def backward_program(self):
        if False:
            for i in range(10):
                print('nop')
        return self._backward_program_desc

    @property
    def input_descs(self):
        if False:
            i = 10
            return i + 15
        return self._input_descs

    @property
    def output_descs(self):
        if False:
            return 10
        return self._output_descs

    @property
    def persistable_names(self):
        if False:
            return 10
        return self._persistable_names

    @property
    def scope(self):
        if False:
            for i in range(10):
                print('nop')
        return self._inner_scope

    @property
    def grad_var_names(self):
        if False:
            while True:
                i = 10
        return self._grad_var_names

    def _preprocess(self, program_desc):
        if False:
            i = 10
            return i + 15
        list_persistable_var = _get_persistable_var_names(program_desc)
        (rename_new_old_dict, _) = _rename_var_program_desc(program_desc, list_persistable_var)
        ops_to_remove = []
        root_block = program_desc.block(0)
        for i in range(root_block.op_size()):
            op = root_block.op(i)
            if op.type() == 'feed':
                ops_to_remove.append(i)
                feed_var_name = op.input('X')[0].encode()
                root_block._remove_var(feed_var_name)
                self._input_descs.append(root_block.find_var(op.output('Out')[0].encode()))
            elif op.type() == 'scale' and op.output('Out')[0].startswith('save_infer_model/scale_'):
                ops_to_remove.append(i)
                out_var_name = op.output('Out')[0].encode()
                root_block._remove_var(out_var_name)
                self._output_descs.append(root_block.find_var(op.input('X')[0].encode()))
            elif op.type() == 'fetch':
                ops_to_remove.append(i)
                fetch_var_name = op.output('Out')[0].encode()
                root_block._remove_var(fetch_var_name)
                if not op.input('X')[0].startswith('save_infer_model/scale_'):
                    self._output_descs.append(root_block.find_var(op.input('X')[0].encode()))
            elif op.has_attr('op_callstack'):
                op.remove_attr('op_callstack')
        for op_idx in reversed(ops_to_remove):
            root_block._remove_op(op_idx, op_idx + 1)
        self._input_descs.reverse()
        tmp_program = _build_program_by_desc(program_desc)
        self._append_scale_to_output(tmp_program)
        self._suffix_varname_dict = _get_loaded_var_new_old(program_desc, rename_new_old_dict)
        self._persistable_names = _get_persistable_var_names(program_desc)
        return program_desc

    @switch_to_static_graph
    def _append_scale_to_output(self, program):
        if False:
            i = 10
            return i + 15
        for out_desc in self._output_descs:
            if out_desc.dtype() == core.VarDesc.VarType.BOOL:
                return
        scale_output_vars = []
        with framework.program_guard(program):
            for (i, out) in enumerate(self._output_descs):
                var = program.global_block().var(out.name())
                var = paddle.scale(var, 1.0, name=f'translated_layer/scale_{i}')
                scale_output_vars.append(var)
        for (i, var) in enumerate(scale_output_vars):
            self._output_descs[i] = var.desc

    @switch_to_static_graph
    def _get_train_forward_program(self, infer_program_desc):
        if False:
            i = 10
            return i + 15
        program_desc_copy = core.ProgramDesc(infer_program_desc)
        _change_is_test_status(program_desc_copy, False)
        program = _build_program_by_desc(program_desc_copy)
        for block_idx in range(program.num_blocks):
            block = program.block(block_idx)
            for op in block.ops:
                if op.type == 'batch_norm':
                    if 'ReserveSpace' not in op.output_names or len(op.output('ReserveSpace')) == 0:
                        reserve_space = block.create_var(name=unique_name.generate_with_ignorable_key('.'.join(['reserve_space', 'tmp'])), dtype=block.var(op.input('X')[0]).dtype, type=core.VarDesc.VarType.LOD_TENSOR, persistable=False, stop_gradient=True)
                        op.desc.set_output('ReserveSpace', [reserve_space.name])
                    continue
                if not OpProtoHolder.instance().has_op_proto(op.type):
                    continue
                proto = OpProtoHolder.instance().get_op_proto(op.type)
                has_create_intermediate_out = False
                for output_proto in proto.outputs:
                    if output_proto.intermediate:
                        intermediate_name = output_proto.name
                        if intermediate_name not in op.output_names:
                            has_create_intermediate_out = True
                            intermediate_var = block.create_var(name=unique_name.generate_with_ignorable_key('.'.join([op.type + '_' + intermediate_name, 'tmp'])), type=core.VarDesc.VarType.LOD_TENSOR, persistable=False, stop_gradient=True)
                            op.desc.set_output(intermediate_name, [intermediate_var.name])
                if has_create_intermediate_out:
                    op.desc.infer_var_type(block.desc)
                    op.desc.infer_shape(block.desc)
        return program

    @switch_to_static_graph
    def _append_backward_desc(self, infer_program_desc):
        if False:
            print('Hello World!')
        program = self._get_train_forward_program(infer_program_desc)
        targets = []
        for out in self._output_descs:
            targets.append(program.global_block().var(out.name()))
        check_type(targets, 'targets', (framework.Variable, list, tuple), 'paddle.static.gradients')
        grad_info_map = backward.calc_gradient_helper(targets=targets, inputs=[])
        x_vars = [program.block(0).var(desc.name()) for desc in self._input_descs]
        param_vars = [program.block(0).var(name) for name in self._persistable_names]
        out_vars = [program.block(0).var(desc.name()) for desc in self._output_descs]
        self._grad_var_names = construct_grad_names(grad_info_map, x_vars, param_vars, out_vars)
        return program.desc

def _load_persistable_vars_by_program(model_path, program_holder, params_filename=None):
    if False:
        print('Hello World!')
    persistable_vars = _get_persistable_vars(program_holder.infer_program)
    load_var_dict = {}
    for each_var in persistable_vars:
        orig_each_name = program_holder._suffix_varname_dict[each_var.name()]
        if _is_parameter(each_var, program_holder.infer_program):
            new_var = framework.EagerParamBase(shape=each_var.shape(), dtype=each_var.dtype(), name=each_var.name(), type=each_var.type(), persistable=True)
        else:
            new_var = framework._create_tensor(type=each_var.type(), name=each_var.name(), shape=each_var.shape(), dtype=each_var.dtype(), persistable=True)
        if params_filename is None:
            framework._dygraph_tracer().trace_op(type='load', inputs={}, outputs={'Out': new_var}, attrs={'file_path': os.path.join(model_path, orig_each_name)})
        new_var.stop_gradient = False
        load_var_dict[each_var.name()] = new_var
    if params_filename is not None:
        load_var_list = []
        dict_name_old_new = {v: k for (k, v) in program_holder._suffix_varname_dict.items()}
        for name in sorted(dict_name_old_new.keys()):
            load_var_list.append(load_var_dict[dict_name_old_new[name]])
        framework._dygraph_tracer().trace_op(type='load_combine', inputs={}, outputs={'Out': load_var_list}, attrs={'file_path': os.path.join(model_path, params_filename)})
        for each_var in persistable_vars:
            if not _is_parameter(each_var, program_holder.infer_program):
                continue
            param = load_var_dict[each_var.name()]
            param.stop_gradient = False
    all_var_names = _get_all_var_names(program_holder.train_program)
    for var_name in load_var_dict:
        grad_var_name = var_name + core.grad_var_suffix()
        if grad_var_name not in all_var_names:
            load_var_dict[var_name].stop_gradient = True
    return load_var_dict

def _load_persistable_vars(model_path, var_info_path, program_holder, params_filename):
    if False:
        for i in range(10):
            print('nop')
    with open(var_info_path, 'rb') as f:
        extra_var_info = pickle.load(f)
    load_var_dict = {}
    load_var_list = []
    inv_suffix_varname_dict = {value: key for (key, value) in program_holder._suffix_varname_dict.items()}
    for name in sorted(inv_suffix_varname_dict):
        if name not in extra_var_info:
            raise RuntimeError('The model to be loaded is not complete.The variable `%s` of program cannot be found in loaded model.', name)
        new_name = inv_suffix_varname_dict[name]
        if extra_var_info[name].get('trainable', None) is not None:
            new_var = framework.EagerParamBase(shape=[1], dtype=core.VarDesc.VarType.FP32, name=new_name, persistable=True)
        else:
            new_var = framework._create_tensor(name=new_name, persistable=True)
        new_var.stop_gradient = extra_var_info[name]['stop_gradient']
        load_var_dict[new_name] = new_var
        load_var_list.append(new_var)
    assert params_filename is not None, 'params_filename should not be None.'
    var_file_path = os.path.join(model_path, params_filename)
    if not os.path.exists(var_file_path):
        if len(extra_var_info) != 0:
            raise ValueError('The model to be loaded is incomplete.')
    else:
        framework._dygraph_tracer().trace_op(type='load_combine', inputs={}, outputs={'Out': load_var_list}, attrs={'file_path': var_file_path})
    return load_var_dict

def _remove_varname_suffix(var_dict, program_holder):
    if False:
        print('Hello World!')
    no_suffix_var_dict = {}
    for var_name in var_dict:
        no_suffix_name = program_holder._suffix_varname_dict[var_name]
        no_suffix_var_dict[no_suffix_name] = var_dict[var_name]
    return no_suffix_var_dict

def _construct_program_holders(model_path, model_filename=None):
    if False:
        for i in range(10):
            print('nop')
    program_holder_dict = {}
    if model_filename is not None:
        model_filename = os.path.basename(model_filename)
        model_file_path = os.path.join(model_path, model_filename)
        model_name = model_filename[:-len(INFER_MODEL_SUFFIX)]
        for filename in os.listdir(model_path):
            if model_filename == filename:
                func_name = 'forward'
                model_file_path = os.path.join(model_path, model_filename)
            elif filename.endswith(INFER_MODEL_SUFFIX) and filename.startswith(model_name):
                parsing_names = filename[len(model_name):-len(INFER_MODEL_SUFFIX) + 1].split('.')
                if len(parsing_names) == 3 and len(parsing_names[1]) > 0:
                    func_name = parsing_names[1]
                    model_file_path = os.path.join(model_path, filename)
                else:
                    continue
            else:
                continue
            program_holder_dict[func_name] = _ProgramHolder(_load_program_desc(model_file_path))
    else:
        for (_, _, file_names) in os.walk(model_path):
            for name in file_names:
                if 'model' in name:
                    model_file_path = os.path.join(model_path, name)
                    method_name = name.strip('_')
                    if method_name == 'model':
                        method_name = 'forward'
                    else:
                        method_name.replace('model', '')
                    program_holder_dict[method_name] = _ProgramHolder(_load_program_desc(model_file_path))
    return program_holder_dict

def _construct_params_and_buffers(model_path, programs, params_filename=None, append_suffix=True):
    if False:
        print('Hello World!')
    var_info_filename = str(params_filename) + '.info'
    var_info_path = os.path.join(model_path, var_info_filename)
    params_path = os.path.join(model_path, str(params_filename))
    if os.path.exists(var_info_path):
        var_dict = _load_persistable_vars(model_path, var_info_path, programs['forward'], params_filename)
        model_name = params_filename[:-len(INFER_PARAMS_SUFFIX)]
        for file_name in os.listdir(model_path):
            if file_name.startswith(model_name) and file_name.endswith(INFER_PARAMS_SUFFIX):
                parsing_names = file_name[len(model_name):-len(INFER_PARAMS_SUFFIX) + 1].split('.')
                if len(parsing_names) == 3 and len(parsing_names[1]) > 0:
                    func_name = parsing_names[1]
                else:
                    continue
            else:
                continue
            var_info_path = os.path.join(model_path, var_info_filename)
            var_dict.update(_load_persistable_vars(model_path, var_info_path, programs[func_name], file_name))
    elif params_filename is not None and (not os.path.exists(params_path)):
        return {}
    else:
        var_dict = _load_persistable_vars_by_program(model_path, programs['forward'], params_filename)
    if not append_suffix:
        var_dict = _remove_varname_suffix(var_dict, programs['forward'])
    return var_dict

def _valid_vars(vars):
    if False:
        return 10
    return vars if vars else None

def _run_dygraph(instance, input, program_holder):
    if False:
        return 10
    input_vars = []
    input_var_names = []
    for (i, value) in enumerate(input):
        if not isinstance(value, (np.ndarray, core.eager.Tensor)):
            raise TypeError('The type of input in TranslatedLayer must be numpy array or Variable(Tensor), but received %s.' % type(value))
        if isinstance(value, np.ndarray):
            var = core.eager.Tensor(value=value, name=program_holder.input_descs[i].name(), persistable=False, place=framework._current_expected_place(), zero_copy=True)
        else:
            var = value
            var.name = program_holder.input_descs[i].name()
        input_var_names.append(var.name)
        input_vars.append(var)
    if instance._input_args_names is None:
        instance._input_args_names = [ins.name() for ins in program_holder.input_descs]
    persistable_vars = []
    for var_name in program_holder.persistable_names:
        dy_var_name = instance._persistable_var_name_dict[var_name]
        if dy_var_name in instance._parameters:
            persistable_vars.append(instance._parameters[dy_var_name])
        elif dy_var_name in instance._buffers:
            persistable_vars.append(instance._buffers[dy_var_name])
        else:
            raise ValueError('The persistable variable %s does not exist in current TranslatedLayer.' % var_name)
    output_vars = []
    for var_desc in program_holder.output_descs:
        var = core.eager.Tensor(dtype=var_desc.dtype(), dims=var_desc.shape(), name=var_desc.name(), type=var_desc.type(), persistable=False)
        output_vars.append(var)
    tmp_scope_vec = [program_holder.scope]
    trace_program = program_holder.infer_program if instance._is_test else program_holder.train_program
    forward_program = program_holder._infer_program_desc if instance._is_test else program_holder.forward_program
    end_op_index = program_holder.infer_program.block(0).op_size()
    attrs = ['global_block', trace_program.block(0), 'start_op_index', 0, 'end_op_index', end_op_index, 'is_test', instance._is_test, 'program_id', paddle.utils._hash_with_id(trace_program, instance), 'x_names', input_var_names]
    if not instance._is_test:
        attrs.extend(('param_grad_names', program_holder.grad_var_names.get('param', []), 'out_grad_names', program_holder.grad_var_names.get('out', []), 'x_grad_names', program_holder.grad_var_names.get('x', [])))
    use_interpretorcore = True
    attrs.extend(('use_interpretorcore', use_interpretorcore))
    if use_interpretorcore:
        attrs.extend(('forward_global_block', forward_program.block(0)))
        if not instance._is_test:
            attrs.extend(('backward_global_block', program_holder.backward_program.block(0)))
    _legacy_C_ops.run_program(_valid_vars(input_vars), _valid_vars(persistable_vars), _valid_vars(output_vars), tmp_scope_vec, None, *attrs)
    for persistable_var in persistable_vars:
        grad_var_name = persistable_var.name + core.grad_var_suffix()
        grad_var = trace_program.block(0).find_var(grad_var_name.encode())
        if grad_var is None:
            continue
        persistable_var._set_grad_type(grad_var.type())
    outs = output_vars
    if len(output_vars) == 1:
        outs = output_vars[0]
    return outs

def _run_static_graph(input, program_holder, trace_program):
    if False:
        return 10
    main_program = framework.default_main_program()
    param_var_names = _get_persistable_var_names(trace_program)
    (_, dict_rename_var_old_new) = _rename_var_program_desc(trace_program, exclude=param_var_names)
    trace_program.flush()
    _append_block(main_program, trace_program, program_holder, input, dict_rename_var_old_new)
    main_program._sync_with_cpp()
    outs = _get_output_from_program(main_program, program_holder, dict_rename_var_old_new)
    if len(outs) == 1:
        outs = outs[0]
    return outs

def _collect_current_and_parent_var(program, block_idx):
    if False:
        while True:
            i = 10
    '\n    Get variables in current block and its parent block.\n\n    Args:\n        program(Program): The program containing the current block.\n        block_idx(int): index of current block.\n\n    Returns:\n        List: list of variables.\n    '
    vars = []
    if block_idx < 0:
        return vars
    for var in program.block(block_idx).vars:
        vars.append(var)
    parent_idx = program.block(block_idx).parent_idx
    if parent_idx > -1:
        vars += _collect_current_and_parent_var(program, parent_idx)
    return vars

def _append_block(dest_program, src_program_desc, program_holder, input_variables, dict_rename_var_old_new=None):
    if False:
        i = 10
        return i + 15
    "\n    Append Variables and Operators in 'src_program_desc' to dest_program.\n\n    Args:\n        dest_program(Program): Variables and Operators are appended to it.\n        src_program_desc(ProgramDesc): Variables in it will be appended to 'dest_program'.\n        program_holder(_ProgramHolder): program_holder of TranslatedLayer\n        input_variables(list): list of input variables\n        dict_rename_var_old_new(None|dict): When using '_rename_var_program_desc',\n        use it to map the name of the variable before it was modified and the new name.\n    "
    origin_block_idx = dest_program.current_block_idx
    param_var_names = _collect_current_and_parent_var(dest_program, origin_block_idx)
    append_var_from_block_desc_static(dest_program.block(origin_block_idx), src_program_desc.block(0), exclude=param_var_names)
    name_inp_desc = [inp.name() for inp in program_holder.input_descs]
    input_names = [inp.name for inp in input_variables]
    if len(name_inp_desc) != len(input_names):
        raise ValueError('The number of input is invalid, expected {}, but received {}.'.format(len(name_inp_desc), len(input_names)))
    for (i, out_name) in enumerate(name_inp_desc):
        if dict_rename_var_old_new:
            out_name = dict_rename_var_old_new[out_name]
        dest_program.block(origin_block_idx).append_op(type='assign', inputs={'X': [input_names[i]]}, outputs={'Out': [out_name]})
    append_ops = append_op_from_block_desc_static(dest_program.block(origin_block_idx), src_program_desc.block(0))
    dest_program._sync_with_cpp()
    offset_block_idx = dest_program.num_blocks - 1
    parent_idx = 0
    if src_program_desc.num_blocks() > 1:
        for src_block_idx in range(1, src_program_desc.num_blocks()):
            src_block = src_program_desc.block(src_block_idx)
            src_parent_idx = src_block.parent
            if src_parent_idx > 0:
                parent_idx = offset_block_idx + parent_idx
            else:
                parent_idx = origin_block_idx
            dest_block = dest_program._create_block(parent_idx=parent_idx)
            append_var_from_block_desc_static(dest_block, src_block, exclude=param_var_names)
            append_ops += append_op_from_block_desc_static(dest_block, src_block)
    dest_program._sync_with_cpp()
    for op in append_ops:
        if op.has_attr('sub_block'):
            sub = op.attr('sub_block')
            if isinstance(sub, framework.core.BlockDesc):
                origin_id = sub.id
            if isinstance(sub, framework.Block):
                origin_id = sub.idx
            op._set_attr('sub_block', dest_program.block(offset_block_idx + origin_id))
    dest_program._sync_with_cpp()
    dest_program.current_block_idx = origin_block_idx

def _get_output_from_program(program, program_holder, dict_rename_var_old_new=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get output name of 'program' according to program_holder\n    "
    outs = []
    for var in program_holder.output_descs:
        for idx in range(program.num_blocks):
            vars = program.block(idx).vars
            var_name = var.name()
            if dict_rename_var_old_new:
                var_name = dict_rename_var_old_new[var_name]
            if var_name in vars:
                out = vars[var_name]
                if out not in outs:
                    outs.append(out)
    return outs

def append_op_from_block_desc_static(block, src_block_desc):
    if False:
        return 10
    "\n    Append Operators of 'src_block_desc' to current block.\n\n    Args:\n        block(Block): append OP of  'src_block_desc' to it.\n        src_block_desc(BlockDesc): append var of  'src_block_desc'\n\n    Returns:\n        List: list of the OP that are append to current block.\n    "
    ops = []
    for i in range(src_block_desc.op_size()):
        ops.append(append_op_from_desc_static(block, src_block_desc.op(i)))
    return ops

def append_op_from_desc_static(block, op_desc):
    if False:
        i = 10
        return i + 15
    "\n    Append Operators to 'block' according to 'op_desc'.\n\n    Args:\n        block(Block): append OP of  'src_block_desc' to it.\n        op_desc(OpDesc): create OP according to it.\n\n    Returns:\n        Operator: OP appended to 'block'.\n    "
    op_type = op_desc.type()
    op_append = block.desc.append_op()
    op_append.copy_from(op_desc)
    op = framework.Operator(block=block, desc=op_append, type=op_type, inputs=None, outputs=None, attrs=None)
    block.ops.append(op)
    return op

def append_var_from_block_desc_static(block, src_block_desc, include=None, exclude=None):
    if False:
        i = 10
        return i + 15
    "\n    Append Variables of 'src_block_desc' to current block.\n    If 'include' is not `None`,variables that are not in include are not append.\n    If 'exclude' is not `None`,variables that are in exclude will are not append.\n\n    Args:\n        block(Block): append Variables of  'src_block_desc' to it.\n        src_block_desc(BlockDesc): append var of  'src_block_desc'\n        include(List):list of names of variables\n        exclude(List):list of names of variables\n\n    Returns:\n        List: list of the variables that are append to current block.\n    "
    vars_append = []
    for var_desc in src_block_desc.all_vars():
        var_desc_name = var_desc.name()
        should_append = (include is None or var_desc_name in include) and (exclude is None or var_desc_name not in exclude)
        if not block.has_var(var_desc_name) and should_append:
            var_type = var_desc.type()
            if var_type in [core.VarDesc.VarType.SELECTED_ROWS, core.VarDesc.VarType.LOD_TENSOR, core.VarDesc.VarType.LOD_TENSOR_ARRAY]:
                data_type = var_desc.dtype()
                var_shape = var_desc.shape()
            else:
                data_type = None
                var_shape = None
            if var_type in [core.VarDesc.VarType.LOD_TENSOR, core.VarDesc.VarType.LOD_TENSOR_ARRAY]:
                lod_level = var_desc.lod_level()
            else:
                lod_level = None
            if var_desc.persistable():
                current_block = block.program.global_block()
            else:
                current_block = block
            vars_append.append(current_block.create_var(name=var_desc.name(), dtype=data_type, type=var_type, shape=var_shape, lod_level=lod_level, persistable=var_desc.persistable(), set_need_check_feed=var_desc.need_check_feed()))
    return vars_append

class TranslatedLayer(layers.Layer):
    """
    TranslatedLayer is a ``paddle.nn.Layer`` for holding the model
    loaded by :ref:`api_paddle_jit_load` . It can be used like a
    general Layer object in eval or train mode.

    .. note:
        The TranslatedLayer objects should not be created by constructor, it only can be loaded and constructed by :ref:`api_paddle_jit_load` .

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
            >>> import numpy as np
            >>> import paddle
            >>> import paddle.nn as nn
            >>> import paddle.optimizer as opt

            >>> BATCH_SIZE = 16
            >>> BATCH_NUM = 4
            >>> EPOCH_NUM = 4

            >>> IMAGE_SIZE = 784
            >>> CLASS_NUM = 10

            >>> # define a random dataset
            >>> class RandomDataset(paddle.io.Dataset):
            ...     def __init__(self, num_samples):
            ...         self.num_samples = num_samples
            ...
            ...     def __getitem__(self, idx):
            ...         image = np.random.random([IMAGE_SIZE]).astype('float32')
            ...         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
            ...         return image, label
            ...
            ...     def __len__(self):
            ...         return self.num_samples
            ...
            >>> class LinearNet(nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
            ...
            ...     @paddle.jit.to_static
            ...     def forward(self, x):
            ...         return self._linear(x)
            ...
            >>> def train(layer, loader, loss_fn, opt):
            ...     for epoch_id in range(EPOCH_NUM):
            ...         for batch_id, (image, label) in enumerate(loader()):
            ...             out = layer(image)
            ...             loss = loss_fn(out, label)
            ...             loss.backward()
            ...             opt.step()
            ...             opt.clear_grad()
            ...             print("Epoch {} batch {}: loss = {}".format(
            ...                 epoch_id, batch_id, np.mean(loss.numpy())))
            ...
            >>> # 1. train & save model.
            >>> # create network
            >>> layer = LinearNet()
            >>> loss_fn = nn.CrossEntropyLoss()
            >>> adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

            >>> # create data loader
            >>> dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            >>> loader = paddle.io.DataLoader(dataset,
            ...     batch_size=BATCH_SIZE,
            ...     shuffle=True,
            ...     drop_last=True,
            ...     num_workers=2
            ... )
            >>> # train
            >>> train(layer, loader, loss_fn, adam)

            >>> # save
            >>> model_path = "linear.example.model"
            >>> paddle.jit.save(layer, model_path)

            >>> # 2. load model as TranslatedLayer
            >>> # load
            >>> translated_layer = paddle.jit.load(model_path)

            >>> # inference
            >>> translated_layer.eval()
            >>> x = paddle.randn([1, IMAGE_SIZE], 'float32')
            >>> pred = translated_layer(x)

            >>> # fine-tune
            >>> translated_layer.train()
            >>> adam = opt.Adam(learning_rate=0.001, parameters=translated_layer.parameters())
            >>> train(translated_layer, loader, loss_fn, adam)

    """

    def __init__(self, programs, persistable_vars):
        if False:
            print('Hello World!')
        super().__init__()
        if not isinstance(programs, dict):
            raise TypeError("TranslatedLayer need to use _ProgramHolder's dict for initialization.")
        if not isinstance(persistable_vars, dict):
            raise TypeError('TranslatedLayer need to use persistable variable dict for initialization.')
        self._program_holder_dict = programs
        self._persistable_var_name_dict = {}
        with unique_name.guard():
            for (name, var) in persistable_vars.items():
                if isinstance(var, framework.EagerParamBase):
                    dy_name = _generate_unique_var_name(PARAMETER_NAME_PREFIX)
                    self._persistable_var_name_dict[name] = dy_name
                    self.add_parameter(dy_name, var)
                elif isinstance(var, core.eager.Tensor):
                    dy_name = _generate_unique_var_name(BUFFER_NAME_PREFIX)
                    self._persistable_var_name_dict[name] = dy_name
                    self.register_buffer(dy_name, var)
                else:
                    raise TypeError('Adding persistent variable which  to layer is not supported now')
        self._is_test = True
        self._input_args_names = None

    @staticmethod
    @framework.dygraph_only
    def _construct(model_path, configs=None):
        if False:
            print('Hello World!')
        model_path = os.path.normpath(model_path)
        if not os.path.isdir(model_path):
            raise ValueError("There is no directory named '%s'" % model_path)
        model_filename = None
        params_filename = None
        if configs is not None:
            model_filename = configs.model_filename
            params_filename = configs.params_filename
        programs = _construct_program_holders(model_path, model_filename)
        persistable_vars = _construct_params_and_buffers(model_path, programs, params_filename)
        translated_layer = TranslatedLayer(programs, persistable_vars)
        for (method_name, program_holder) in programs.items():
            if translated_layer._input_args_names is None:
                translated_layer._input_args_names = [ins.name() for ins in program_holder.input_descs]
            setattr(TranslatedLayer, method_name, TranslatedLayer._execution_method_creator(method_name, program_holder))
        translated_layer.eval()
        return translated_layer

    @staticmethod
    def _execution_method_creator(method_name, program_holder):
        if False:
            for i in range(10):
                print('nop')

        def __i_m_p_l__(self, *input):
            if False:
                while True:
                    i = 10
            program_holder = self._program_holder_dict[__i_m_p_l__.__name__]
            if in_dynamic_mode():
                return _run_dygraph(self, input, program_holder)
            else:
                p = framework.Program._construct_from_desc(core.ProgramDesc(program_holder.infer_program))
                return _run_static_graph(input, program_holder, p.desc)
        __i_m_p_l__.__name__ = method_name
        return __i_m_p_l__

    def train(self):
        if False:
            print('Hello World!')
        self._is_test = False
        self.training = True

    def eval(self):
        if False:
            print('Hello World!')
        self._is_test = True
        self.training = False

    def program(self, method_name='forward'):
        if False:
            return 10
        '\n        Gets translated program of specified method.\n\n        Args:\n            - method_name (string): mehtod name corresponding to the program\n                to be obtained. Default: \'forward\'.\n\n        Returns:\n            Program\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +SKIP(\'`paddle.jit.to_static` can not run in xdoctest\')\n                >>> import numpy as np\n                >>> import paddle\n                >>> from paddle import nn\n                >>> import paddle.optimizer as opt\n\n                >>> BATCH_SIZE = 16\n                >>> BATCH_NUM = 4\n                >>> EPOCH_NUM = 4\n\n                >>> IMAGE_SIZE = 784\n                >>> CLASS_NUM = 10\n\n                >>> # define a random dataset\n                >>> class RandomDataset(paddle.io.Dataset):\n                ...     def __init__(self, num_samples):\n                ...         self.num_samples = num_samples\n                ...\n                ...     def __getitem__(self, idx):\n                ...         image = np.random.random([IMAGE_SIZE]).astype(\'float32\')\n                ...         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype(\'int64\')\n                ...         return image, label\n                ...\n                ...     def __len__(self):\n                ...         return self.num_samples\n                ...\n                >>> class LinearNet(nn.Layer):\n                ...     def __init__(self):\n                ...         super().__init__()\n                ...         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)\n                ...\n                ...     @paddle.jit.to_static\n                ...     def forward(self, x):\n                ...         return self._linear(x)\n                ...\n                >>> def train(layer, loader, loss_fn, opt):\n                ...     for epoch_id in range(EPOCH_NUM):\n                ...         for batch_id, (image, label) in enumerate(loader()):\n                ...             out = layer(image)\n                ...             loss = loss_fn(out, label)\n                ...             loss.backward()\n                ...             opt.step()\n                ...             opt.clear_grad()\n                ...             print("Epoch {} batch {}: loss = {}".format(\n                ...                 epoch_id, batch_id, np.mean(loss.numpy())))\n                ...\n                >>> # create network\n                >>> layer = LinearNet()\n                >>> loss_fn = nn.CrossEntropyLoss()\n                >>> adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())\n                >>> # create data loader\n                >>> dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)\n                >>> loader = paddle.io.DataLoader(dataset,\n                ...     batch_size=BATCH_SIZE,\n                ...     shuffle=True,\n                ...     drop_last=True,\n                ...     num_workers=2\n                ... )\n                >>> # train\n                >>> train(layer, loader, loss_fn, adam)\n\n                >>> # save\n                >>> model_path = "linear.example.model"\n                >>> paddle.jit.save(layer, model_path)\n\n                >>> # load\n                >>> translated_layer = paddle.jit.load(model_path)\n\n                >>> # get program\n                >>> program = translated_layer.program()\n        '
        program_holder = self._get_program_holder(method_name)
        program_desc = program_holder.infer_program
        program = _build_program_by_desc(program_desc)
        return program

    def _get_program_holder(self, method_name='forward'):
        if False:
            while True:
                i = 10
        program_holder = self._program_holder_dict.get(method_name, None)
        if program_holder is None:
            raise ValueError('The method `%s` does not exist in loaded TranslatedLayer.' % method_name)
        return program_holder

    def _input_spec(self, method_name='forward'):
        if False:
            print('Hello World!')
        program_holder = self._get_program_holder(method_name)
        input_spec = []
        for var_desc in program_holder.input_descs:
            spec = paddle.static.InputSpec(shape=var_desc.shape(), dtype=var_desc.dtype(), name=var_desc.name())
            input_spec.append(spec)
        return input_spec

    def _output_spec(self, method_name='forward'):
        if False:
            for i in range(10):
                print('nop')
        program_holder = self._get_program_holder(method_name)
        output_spec = []
        for var_desc in program_holder.output_descs:
            spec = paddle.static.InputSpec(shape=var_desc.shape(), dtype=var_desc.dtype(), name=var_desc.name())
            output_spec.append(spec)
        return output_spec