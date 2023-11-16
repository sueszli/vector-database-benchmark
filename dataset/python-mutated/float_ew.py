"""
Floating point elementwise operations on GPU.
"""
from __future__ import division
from future.utils import native_str
import os.path
import re
import traceback as tb
import numpy as np
from pycuda.tools import context_dependent_memoize
from pytools import memoize
from neon import logger as neon_logger
import neon.backends.nervanagpu as ng
from neon.backends.util.source_module import SourceModule
from neon.backends.cuda_templates import _ew_template, _stage_template, _fin_template, _init_rand_func, _init_rand_round_func, _finish_rand_func, _common_kepler, _common_urand_gen, _common_frand, _common_round, _common_fp16_to_fp32, _ew_types, _ew_strings, _is_finite, _float_ops, _reduction_ops
from neon.backends.cuda_batchnorm import _get_bn_fprop_kernel, _get_bn_bprop_kernel
from neon.backends.kernels.cuda.lookuptable import _get_lut_bprop_kernel, _get_sorting_kernel

def _build_tree(type_args):
    if False:
        return 10
    '\n    rebuild a mutable tree from the stack\n    flag each op node with whether it is scalar or not\n    also include a count of reductions under this node:\n    node: [ arg(op, tensor or const), is_scalar, red_count, left_child, right_child ]\n    '
    stack = list()
    for arg in type_args:
        arg_type = arg[0]
        if arg_type in _float_ops:
            numops = _float_ops[arg_type][0]
            node = [arg, numops > 0, 0]
            for i in range(numops):
                operand = stack.pop()
                if type(operand) is list:
                    node[2] += operand[2]
                    if operand[1] == 0:
                        node[1] = False
                elif operand[0] is ng.GPUTensor and (operand[1] > 0 or not operand[4]):
                    node[1] = False
                node.insert(3, operand)
            stack.append(node)
        elif arg_type in _reduction_ops:
            operand = stack.pop()
            reds = 1
            if type(operand) is list:
                reds += operand[2]
            stack.append([arg, True, reds, operand])
        else:
            stack.append(arg)
    return stack[0]

def _print_tree(node, level=0):
    if False:
        while True:
            i = 10
    '\n    print tree with indentation\n    '
    if type(node) is list:
        neon_logger.display('    ' * level + ', '.join((native_str(s) for s in node[0:3])))
        if len(node) > 3:
            _print_tree(node[3], level + 1)
        if len(node) > 4:
            _print_tree(node[4], level + 1)
    else:
        neon_logger.display('    ' * level + native_str(node))

def _post_order(node, stack=None):
    if False:
        i = 10
        return i + 15
    '\n    generate a stack from a portion of the tree\n    '
    if stack is None:
        stack = list()
    if type(node) is list:
        if len(node) > 3:
            _post_order(node[3], stack)
        if len(node) > 4:
            _post_order(node[4], stack)
        stack.append(node[0])
    else:
        stack.append(node)
    return stack

def _process_node(node, aliases, duplicates):
    if False:
        return 10
    '\n    Takes a node from the tree and searchs for any previously processed\n    duplicates.\n    If not a duplicate, returns a stage based from that node.\n    If a duplicate, the node is replaced with an alias to the dup stage.\n    In both cases the tree is removed below this node (and the alias remains).\n    '
    stack = _post_order(node)
    key = list()
    for item in stack:
        if type(item[0]) is str and item not in aliases:
            key.append(item[0])
        else:
            key.append(item[0:2])
    key = tuple(key)
    dup_node = duplicates.get(key, False)
    if dup_node:
        node[0] = dup_node
        stack = None
    else:
        duplicates[key] = stack[-1]
        aliases.add(stack[-1])
    while len(node) > 3:
        node.pop()
    return stack

def _split_stages(node, duplicates=None, aliases=None, stages=None, parents=None):
    if False:
        print('Hello World!')
    '\n    Split out all reductions and post reduction scalar operations into seperate\n    stacks (stages)\n    This leaves remaining in the tree anything not in these categories.\n    '
    if duplicates is None:
        duplicates = dict()
        aliases = set()
        stages = list()
        parents = list()
    if type(node) is list:
        if node[0][0] != 'assign':
            parents.append(node)
        if len(node) > 3:
            _split_stages(node[3], duplicates, aliases, stages, parents)
        if len(node) > 4:
            _split_stages(node[4], duplicates, aliases, stages, parents)
        if len(parents) > 0:
            parents.pop()
        if node[0][0] in _reduction_ops:
            red_stack = _process_node(node, aliases, duplicates)
            if red_stack:
                stages.append(('reduction', red_stack))
            for parent in parents:
                parent[2] -= 1
            scalar_parent = None
            for parent in parents[::-1]:
                if parent[1] and parent[2] == 0:
                    scalar_parent = parent
                else:
                    break
            if scalar_parent is not None:
                scalar_stack = _process_node(scalar_parent, aliases, duplicates)
                if scalar_stack:
                    stages.append(('scalar', scalar_stack))
    return stages

def _init_rand(template_vals):
    if False:
        return 10
    template_vals['common'].append(_common_urand_gen)
    template_vals['inits'].append(_init_rand_func)
    template_vals['finish'].append(_finish_rand_func)
    return True

@context_dependent_memoize
def _get_compound_kernel(type_args, compute_capability):
    if False:
        return 10
    '\n    generate compound kernel for the optree from type_args\n    '
    tree = _build_tree(type_args)
    stages = _split_stages(tree)
    last_stage = 'red_out' if tree[1] == 1 else 'ew_out'
    stages.append((last_stage, _post_order(tree)))
    stack = list()
    placeholders = list()
    stage_out_reg = dict()
    arg_dict = dict()
    array_ids = set()
    fp16In = False
    rand_init = False
    rand_func = False
    threads = type_args[-1][3]
    template = _ew_template
    template_vals = {'threads': threads, 'name': _get_kernel_name(), 'common': list(), 'inits': list(), 'finish': list()}
    for (stage, stage_data) in enumerate(stages):
        (stage_type, stage_stack) = stage_data
        new_placeholders = list()
        if stage_type == 'reduction':
            new_placeholders.append('loads%d' % stage)
            new_placeholders.append('ops%d' % stage)
            new_placeholders.append('shfl_red%d' % stage)
            template += _stage_template['loop'].format(stage)
            if threads > 32:
                new_placeholders.append('var_red%d' % stage)
                new_placeholders.append('share1_red%d' % stage)
                new_placeholders.append('share2_red%d' % stage)
                template += _stage_template['red'].format(stage)
            else:
                template += _stage_template['red32'].format(stage)
        elif stage_type == 'scalar':
            new_placeholders.append('ops%d' % stage)
            template += _stage_template['red_ops'].format(stage)
        elif stage_type == 'red_out':
            new_placeholders.append('ops%d' % stage)
            template += _stage_template['red_out'].format(stage)
        else:
            new_placeholders.append('loads%d' % stage)
            new_placeholders.append('ops%d' % stage)
            template += _stage_template['loop'].format(stage)
        for key in new_placeholders:
            template_vals[key] = []
        placeholders.extend(new_placeholders)
        for (arg_i, arg) in enumerate(stage_stack):
            (arg_type, arg_id) = arg[0:2]
            if arg_type is ng.GPUTensor:
                (dtype, take_axis) = arg[2:4]
                is_out_tensor = True if stage == len(stages) - 1 and arg_i == 0 else False
                if is_out_tensor:
                    out_dtype = dtype
                    out_take = take_axis
                else:
                    stack.append('a%d' % arg_id)
                ew_dtype = _ew_types[dtype]
                fmt = (arg_id, stage, ew_dtype['type'], ew_dtype['cvt'])
                if arg_id not in array_ids:
                    array_ids.add(arg_id)
                    array_ids.add((arg_id, stage))
                    sig = 'Pii'
                    if take_axis > 0:
                        sig += 'P'
                    if is_out_tensor:
                        ew_out = _ew_strings['out%d' % take_axis]
                        arguments = ew_out['arguments'].format(*fmt)
                        template_vals['inits'].append(ew_out['inits'].format(*fmt))
                    else:
                        ew_in = _ew_strings['in%d' % take_axis]
                        loads = 'loads%d' % stage
                        arguments = ew_in['arguments'].format(*fmt)
                        template_vals['inits'].append(ew_in['inits'].format(*fmt))
                        template_vals[loads].append(ew_in['loads'].format(*fmt))
                    if dtype == 'f2' and (not fp16In):
                        template_vals['common'].append(_common_fp16_to_fp32)
                        fp16In = True
                    arg_dict[arg] = (sig, arguments)
                elif (arg_id, stage) not in array_ids:
                    array_ids.add((arg_id, stage))
                    ew_in = _ew_strings['in%d' % take_axis]
                    loads = 'loads%d' % stage
                    template_vals['inits'].append(ew_in['inits'].format(*fmt))
                    template_vals[loads].append(ew_in['loads'].format(*fmt))
            elif arg_type is float:
                stack.append('c%d' % arg_id)
                if arg not in arg_dict:
                    arg_dict[arg] = ('f', _ew_strings['const']['arguments'].format(arg_id))
            elif arg_type == 'assign':
                ops = 'ops%d' % stage
                sig = 'i'
                arguments = ['const int n%d' % stage]
                if arg[2]:
                    mode = 'random'
                    sig += 'i'
                    arguments.append('const int mantissa_bits')
                    if not rand_init:
                        rand_init = _init_rand(template_vals)
                    template_vals['inits'].append(_init_rand_round_func)
                else:
                    mode = 'nearest'
                arg_dict[arg] = (sig, ', '.join(arguments))
                out_val = stack.pop()
                if out_val[0] == 'i' and out_dtype[0] in 'iu':
                    ew_round = None
                else:
                    ew_round = _ew_strings['round'][mode].get(out_dtype, None)
                    ew_common = _common_round[mode].get(out_dtype, None)
                    if ew_common:
                        template_vals['common'].append(ew_common)
                if ew_round:
                    round_val = 'r%d' % arg_id
                    template_vals[ops].append(ew_round.format(round_val, out_val))
                else:
                    round_val = out_val
                template_vals[ops].append(_ew_strings['out%d' % out_take]['output'].format(round_val))
            elif arg in stage_out_reg:
                stack.append(stage_out_reg[arg])
            elif arg_type in _float_ops:
                if len(template_vals['name']) < 16:
                    template_vals['name'].append(arg_type)
                ops = 'ops%d' % stage
                (num_ops, op_code) = _float_ops[arg_type]
                if arg_type == 'rand':
                    if not rand_init:
                        rand_init = _init_rand(template_vals)
                    if not rand_func:
                        template_vals['common'].append(_common_frand)
                        rand_func = True
                op_list = ['r%d' % arg_id]
                for i in range(num_ops):
                    op_list.append(stack.pop())
                if arg_type == 'onehot':
                    hot_axis = arg[2]
                    test_val = 'i' if hot_axis else 'bid'
                    ew_in = _ew_strings[arg_type + native_str(hot_axis)]
                    loads = 'loads%d' % stage
                    template_vals['inits'].append(ew_in['inits'].format(arg_id))
                    template_vals[loads].append(ew_in['loads'].format(arg_id))
                    op_list.append('onehot%d' % arg_id)
                    op_list.append(test_val)
                    arg_dict[arg] = ('P', ew_in['arguments'].format(arg_id))
                template_vals[ops].append(op_code.format(*op_list))
                if arg_i == len(stage_stack) - 1:
                    stage_out_reg[arg] = op_list[0]
                else:
                    stack.append(op_list[0])
            elif arg_type in _reduction_ops:
                if len(template_vals['name']) < 16:
                    template_vals['name'].append(arg_type)
                arg_dict[arg] = ('i', 'const int n%d' % stage)
                reg = 'i' if 'arg' == arg_type[0:3] else 'r'
                ops = 'ops%d' % stage
                shfl_red = 'shfl_red%d' % stage
                red_arg = '%s%d' % (reg, arg_id)
                red_strings = _reduction_ops[arg_type]
                stack_arg = stack.pop()
                template_vals['inits'].append(red_strings['inits'].format(red_arg))
                template_vals[ops].append(red_strings['ops'].format(red_arg, stack_arg))
                template_vals[shfl_red].append(red_strings['shfl_red'].format(red_arg))
                if threads > 32:
                    var_red = 'var_red%d' % stage
                    shr1_red = 'share1_red%d' % stage
                    shr2_red = 'share2_red%d' % stage
                    template_vals[var_red].append(red_arg)
                    template_vals[shr1_red].append(red_strings['share1_red'].format(red_arg))
                    template_vals[shr2_red].append(red_strings['share2_red'].format(red_arg))
                stage_out_reg[arg] = red_arg
            else:
                raise ValueError('Bad op type.')
    if compute_capability[0] == 3 and compute_capability[1] < 5 or compute_capability[0] < 3:
        template_vals['common'].append(_common_kepler)
    template += _fin_template
    sig = 'P'
    arguments = list()
    unused = 1
    for arg in type_args:
        params = arg_dict.get(arg, False)
        if params:
            sig += params[0]
            arguments.append(params[1])
            del arg_dict[arg]
        elif arg[0] in _reduction_ops:
            sig += 'i'
            arguments.append('const int unused%d' % unused)
            unused += 1
    template_vals['name'] = '_'.join(template_vals['name'])
    template_vals['common'] = '\n'.join(template_vals['common'])
    template_vals['arguments'] = ',\n    '.join(arguments)
    template_vals['inits'] = '\n    '.join(template_vals['inits'])
    template_vals['finish'] = '\n'.join(template_vals['finish'])
    for key in placeholders:
        template_vals[key] = '\n        '.join(template_vals[key])
    code = template % template_vals
    module = SourceModule(code, options=[])
    kernel = module.get_function(template_vals['name'])
    kernel.name = template_vals['name']
    kernel.prepare(sig)
    return kernel

@memoize
def _get_fast_ew_dims(size):
    if False:
        for i in range(10):
            print('nop')
    ew_size = 256
    while ew_size > 0:
        if size % ew_size == 0:
            break
        ew_size -= 32
    if ew_size == 0:
        ew_size = 255
        while ew_size > 0:
            if size % ew_size == 0:
                break
            ew_size -= 1
    shape = (size // ew_size, ew_size)
    return (shape, ng._contiguous_strides(shape))

def call_compound_kernel(rand_state, compute_capability, *args):
    if False:
        i = 10
        return i + 15
    '\n    Pass in a list of GPUTensor objects, constants and operators in postfix notation..\n\n    C +=  2.5 * A * B + 1\n    call_compound_ew_kernel(C, 2.5, A, "mul", B, "mul", 1, "add", C, "add", "assign")\n    '
    out = None
    arg_cnt = 0
    op_cnt = 0
    array_ids = {}
    const_ids = {}
    kernel_args = [rand_state]
    type_args = []
    shape_stack = []
    threads = 32
    red_depth = 0
    contiguous = True
    reduction = False
    broadcast = False
    transpose = False
    argminmax = False
    takeop = False
    axis = 1
    out_shape = args[0].shape
    for arg in args:
        if type(arg) is dict:
            op_name = arg['op']
            if op_name in _reduction_ops:
                if op_name[0:3] == 'arg':
                    argminmax = True
                if arg.get('axis', None) not in (0, 1):
                    raise ValueError('Only reduction along an axis currently supported')
                if reduction is True:
                    if arg['axis'] != axis:
                        raise ValueError('Reduction only allowed along one axis per kernel.')
                else:
                    reduction = True
                    axis = arg['axis']
            elif op_name == 'onehot':
                takeop = True
        elif isinstance(arg, ng.GPUTensor):
            if len(arg.shape) < 2:
                broadcast = True
            elif len(arg.shape) == 2 and min(arg.shape) == 1 and (arg.shape != out_shape):
                broadcast = True
            elif arg.is_trans:
                transpose = True
            elif arg.take_array:
                takeop = True
            elif not arg.is_contiguous:
                contiguous = False
    strides_order = 1 if axis == 1 else -1
    for arg in args:
        if isinstance(arg, ng.GPUTensor):
            if broadcast or reduction or transpose or takeop or (not contiguous):
                if len(arg.shape) == 2:
                    shape = arg.shape
                    strides = list(arg.strides[::strides_order])
                else:
                    raise ValueError('Operations that are not simple elementwise are only currently supported in 2 dimensions.')
            else:
                (shape, strides) = _get_fast_ew_dims(arg.size)
                strides = list(strides[::strides_order])
            if arg in array_ids:
                indx = array_ids[arg]
            else:
                if out is None:
                    out = arg
                    indx = arg_cnt
                else:
                    indx = array_ids[arg] = arg_cnt
                arg_cnt += 1
                if arg.take_array:
                    if arg.base.shape[0] == 1:
                        strides[1 - axis] = 0
                    if arg.base.shape[1] == 1:
                        strides[axis] = 0
                else:
                    if shape[0] == 1:
                        strides[1 - axis] = 0
                    if shape[1] == 1:
                        strides[axis] = 0
                kernel_args.extend((int(arg.gpudata), int(strides[0]), int(strides[1])))
                if arg.take_array:
                    kernel_args.append(arg.take_array[0].gpudata)
            if arg.take_array:
                if axis != 1:
                    take_axis = 2 - arg.take_array[1]
                else:
                    take_axis = arg.take_array[1] + 1
            else:
                take_axis = 0
            type_args.append((ng.GPUTensor, indx, arg.dtype.str[1:], take_axis, shape[axis] == 1))
            shape_stack.append(shape)
        elif type(arg) in (int, float):
            arg = float(arg)
            if arg in const_ids:
                indx = const_ids[arg]
            else:
                indx = const_ids[arg] = arg_cnt
                arg_cnt += 1
                kernel_args.append(arg)
            type_args.append((float, indx))
            shape_stack.append((1, 1))
        elif type(arg) is dict:
            op_name = arg['op']
            if op_name in _float_ops:
                max_shape = [1, 1]
                for op_num in range(_float_ops[op_name][0]):
                    shape = shape_stack.pop()
                    for i in range(2):
                        if shape[i] != max_shape[i]:
                            if shape[i] == 1 or max_shape[i] == 1:
                                max_shape[i] = max(max_shape[i], shape[i])
                            else:
                                raise TypeError('Input shape:%s not compatible' % (shape,))
                if op_name == 'assign':
                    kernel_args.append(int(max_shape[axis]))
                    rounding = out.rounding
                    if rounding:
                        if rounding is True:
                            rounding = 10
                        elif out.dtype.type is np.float32:
                            rounding = min(rounding, 15)
                        elif out.dtype.type is np.float16:
                            rounding = min(rounding, 10)
                        kernel_args.append(max(rounding, 1))
                    if not argminmax:
                        if reduction:
                            if red_depth >= 256:
                                threads = 64
                        elif not (reduction or transpose) and max_shape[1] >= 512:
                            threads = 256
                    type_args.append((op_name, op_cnt, rounding > 0, threads))
                elif op_name == 'onehot':
                    hot_axis = arg['axis'] if axis else 1 - arg['axis']
                    type_args.append((op_name, op_cnt, hot_axis))
                    shape_stack.append(max_shape)
                    kernel_args.append(arg['idx'].gpudata)
                else:
                    type_args.append((op_name, op_cnt))
                    shape_stack.append(max_shape)
            elif op_name in _reduction_ops:
                shape = list(shape_stack.pop())
                red_depth = max(red_depth, shape[axis])
                kernel_args.append(int(shape[axis]))
                type_args.append((op_name, op_cnt))
                shape[axis] = 1
                shape_stack.append(shape)
            else:
                raise TypeError('%s is not a valid operation' % op_name)
            op_cnt += 1
        else:
            raise TypeError('args must be instance of GPUTensor, int, float, or dict (for operators)')
    kernel = _get_compound_kernel(tuple(type_args), compute_capability)
    shared = threads * 4 if reduction and threads > 32 else 0
    if out.backend.bench > 1:
        repeat = 1
        (start, end) = ng._get_events()
        start.record(out.backend.stream)
    else:
        repeat = 1
    for r in range(repeat):
        kernel.prepared_async_call((int(max_shape[1 - axis]), 1, 1), (threads, 1, 1), out.backend.stream, *kernel_args, shared_size=shared)
    if out.backend.bench > 1:
        end.record(out.backend.stream)
        end.synchronize()
        msecs = end.time_since(start) / repeat
        neon_logger.display('%7.3f msecs shape(%d,%d) blk,thd(%d,%d) %s' % (msecs, max_shape[0], max_shape[1], max_shape[1 - axis], threads, kernel.name))
    return out

@context_dependent_memoize
def _get_compensated_sum_kernel(dtype, rounding):
    if False:
        print('Hello World!')
    _compensated_sum = '\n\n%(common)s\n\n__global__ void compensated_sum(unsigned* rand_state,\n          %(type)s* a_sum,\n          %(type)s* a_cmp,\n    const %(type)s* a_add,\n    float cmp_scale, float add_scale,\n    int row_strd, int col_strd, int n, int mantissa_bits)\n{\n    const int tid = threadIdx.x;\n    const int bid = blockIdx.x;\n\n    int offset = bid * row_strd + tid * col_strd;\n    int inc    = 32 * col_strd;\n\n    a_sum += offset;\n    a_cmp += offset;\n    a_add += offset;\n\n    %(inits)s\n\n    for (int i = tid; i < n; i += 32)\n    {\n        float s32 = %(cvt)s(__ldg((const %(type)s*)a_sum));\n        float c32 = %(cvt)s(__ldg((const %(type)s*)a_cmp));\n        float a32 = %(cvt)s(__ldg(a_add));\n\n        // Adjust amount to add by previous compensation\n        float y32 = a32 * add_scale - c32 * cmp_scale;\n\n        // Do the accumulation and truncate to the storage type\n        float rnd_sum = s32 + y32;\n        %(rnd_sum)s\n\n        // Convert accumulation back to fp32 so we can do more math on it\n        float t32 = %(cvt)s(t16);\n\n        // recover the low order bits that were lost in the truncation\n        float rnd_cmp = (t32 - s32) - y32;\n        %(rnd_cmp)s\n\n        *a_sum = t16;\n        *a_cmp = c16;\n\n        a_sum += inc;\n        a_cmp += inc;\n        a_add += inc;\n    }\n    %(finish)s\n}\n'
    template_vals = dict()
    for key in ('common', 'inits', 'finish'):
        template_vals[key] = ''
    if dtype == 'f2':
        template_vals['common'] += _common_fp16_to_fp32
    if rounding:
        template_vals['common'] += _common_urand_gen
        template_vals['common'] += _common_round['nearest'].get(dtype, '')
        template_vals['inits'] += _init_rand_func + _init_rand_round_func
        template_vals['finish'] += _finish_rand_func
        mode = 'random'
    else:
        mode = 'nearest'
    template_vals['common'] += _common_round[mode].get(dtype, '')
    template_vals['type'] = _ew_types[dtype]['type']
    template_vals['cvt'] = _ew_types[dtype]['cvt']
    no_op = 'float {0} = {1};'
    rnd_sum = _ew_strings['round'][mode].get(dtype, no_op)
    rnd_cmp = _ew_strings['round']['nearest'].get(dtype, no_op)
    template_vals['rnd_sum'] = rnd_sum.format('t16', 'rnd_sum')
    template_vals['rnd_cmp'] = rnd_cmp.format('c16', 'rnd_cmp')
    code = _compensated_sum % template_vals
    module = SourceModule(code)
    kernel = module.get_function('compensated_sum')
    kernel.prepare('PPPPffiiii')
    return kernel
nrv_re = re.compile('nervanagpu\\.py$')
name_re = re.compile('\\W')

def _get_kernel_name():
    if False:
        i = 10
        return i + 15
    '\n    Returns the path of the kernel\n    '
    names = ['kernel']
    if 'NVPROF_ID' in os.environ:
        for frame in tb.extract_stack():
            if nrv_re.search(frame[0]):
                break
            caller = frame[0:2]
        (file_path, file_name) = os.path.split(caller[0])
        (path1, path2) = os.path.split(file_path)
        (file_base, ext) = os.path.splitext(file_name)
        for name in (path2, file_base, ext):
            name = name_re.sub('', name)
            if name:
                names.append(name)
        names.append(native_str(caller[1]))
    return names

@context_dependent_memoize
def _get_hist_kernel(dtype_str, nbins, offset):
    if False:
        print('Hello World!')
    '\n    Build a kernel to compute a 64 bin histogram.\n\n    Use templating to generate a customized kernel depending on the input data type.\n\n    Memoized to avoid compiling the same kernel twice.\n    '
    type_str = _ew_types[dtype_str[1:]]
    from string import Template
    code = Template(_common_fp16_to_fp32 + '\n\n#define MAX(a,b) (a > b ? a : b)\n#define MIN(a,b) (a < b ? a : b)\n\n__global__ void kernel_histo (\n    int* d_hist, const $in_type* a1_in,\n    int strides, int size)\n{\n    const int tid = threadIdx.x;\n    const int bid = blockIdx.x;\n\n    __shared__ int s[$nbins];\n    if(tid < $nbins){\n        s[tid] = 0;\n    }\n\n    if(bid == 0 && tid < $nbins){\n        d_hist[tid] = 0;\n    }\n\n    for (int i = tid + blockDim.x*bid; i < size; i += strides)\n    {\n        float a1 = $convert_to_float(__ldg(a1_in + i));\n\n        float absval = fabs(a1);\n\n        float logabs = round(log2f(absval));\n\n        int bin = MIN($nbins-1, MAX(0, logabs-($offset)));\n\n        atomicAdd(&s[bin], 1);\n\n    }\n\n    __syncthreads();\n\n    if(tid < $nbins){\n        atomicAdd(&d_hist[tid], s[tid]);\n    }\n}\n')
    module = SourceModule(code.substitute(in_type=type_str['type'], convert_to_float=type_str['cvt'], nbins=nbins, offset=offset), options=[])
    kernel = module.get_function('kernel_histo')
    kernel.prepare('PPII')
    return kernel

def _compute_hist(tensor, hist, nbins=64, offset=-48):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to compute the histogram of a tensor.\n\n    Arguments:\n        tensor (GPUTensor): the tensor to compute the histogram over\n        hist (gpu pointer): the gpu memory region to store the 64 bin hist in.\n        nbins (int, optional): number of histogram bins, each representing a power of 2\n                               (default 64)\n        offset (int, optional): offset the value of a bin from its idx as a power of two\n                                (default offset=-48 means bin 0 represents 2**-48)\n    '
    threads = 128
    assert nbins < threads and nbins > 0
    size = tensor.size
    strides = np.floor(np.sqrt(size) / threads) * threads
    if strides < threads:
        strides = max(size / threads * threads, threads)
    blocks = max(1, int(strides) // threads)
    kernel_args = [hist, tensor.gpudata, int(strides), size]
    hist_kern = _get_hist_kernel(tensor.dtype.str, nbins, offset)
    hist_kern.prepared_call((blocks, 1, 1), (threads, 1, 1), *kernel_args)