import copy
import functools
import math
import operator
from typing import Any, Tuple
import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import Arg, CallFunction, filter_nodes, KeywordArg, ListOf, Match
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
quantized = torch.ops.quantized
"\nThe quantization.py file primarily incorporates passes related to quantization fusion\nin inductor, includes:\n1. Dequant Promotion;\n2. Conv/GEMM weight prepack with oneDNN Library;\n3. Conv/GEMM quantization fusion with output quant node (if have);\n4. Other pointwise operators' quantization fusion like: qmaxpool2d, qcat and more;\n\nIt also involves int8-mixed-fp32 and int8-mixed-bf16 quantization. The main difference\nof patterns for int8-mixed-bf16, comparing with int8-mixed-fp32, is\n1. There is to(dtype=torch.bfloat16) node at the inputs of activation and weight for Conv/GEMM.\n2. There is to(dtype=torch.float32) node at the outputs of Conv/GEMM before inputs to next quant node.\nRefer to: https://github.com/pytorch/pytorch/issues/111640 for detail design of int8-mixed-bf16\nquantization.\n"

def _may_generate_pattern_with_dtype_convert(pattern, dtype=Arg(), dtype_convert=True):
    if False:
        for i in range(10):
            print('nop')
    if dtype_convert:
        return CallFunction(prims.convert_element_type.default, pattern, dtype)
    else:
        return pattern
'\ndequantize activation:\n    x = x.to(fp32)\n    x = x - zero_point\n    x = x * scale\n'
dequantize_per_tensor_activation_pattern = CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale'))
dequantize_per_channel_weight_pattern = CallFunction(quantized_decomposed.dequantize_per_channel.default, KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype'))
dequantize_per_channel_to_bf16_weight_pattern = _may_generate_pattern_with_dtype_convert(dequantize_per_channel_weight_pattern, KeywordArg('autocast_wgt_dtype'))
dequantize_per_channel_clone_weight_pattern = CallFunction(aten.clone.default, dequantize_per_channel_weight_pattern, memory_format=KeywordArg('memory_format'))
dequantize_per_channel_to_bf16_clone_weight_pattern = CallFunction(aten.clone.default, dequantize_per_channel_to_bf16_weight_pattern, memory_format=KeywordArg('memory_format'))
dequantize_qconv_pt2e_pattern = CallFunction(torch.ops.onednn.qconv2d_pointwise.default, KeywordArg('x'), KeywordArg('x_scale'), KeywordArg('x_zp'), KeywordArg('packed_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('groups'), KeywordArg('inv_output_scale'), KeywordArg('output_zero_point'), KeywordArg('output_dtype'), KeywordArg('attr'), Arg(), Arg())
qlinear_pt2e_pattern = CallFunction(torch.ops.onednn.qlinear_pointwise.default, KeywordArg('x'), KeywordArg('x_scale'), KeywordArg('x_zp'), KeywordArg('packed_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('b'), KeywordArg('output_scale'), KeywordArg('output_zero_point'), KeywordArg('output_dtype'), KeywordArg('postop_name'), KeywordArg('postop_args'), KeywordArg('postop_algorithm'))
dequantize_accum_pattern = CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, KeywordArg('accum'), KeywordArg('accum_dq_dtype')), KeywordArg('accum_zp')), KeywordArg('accum_scale'))

def generate_pattern_with_binary(binary_post_op, computation_call, extra_input_pattern, int8_mixed_bf16_with_inplace_add=False):
    if False:
        print('Hello World!')
    binary_pattern = CallFunction(binary_post_op, computation_call, extra_input_pattern)
    return _may_generate_pattern_with_dtype_convert(binary_pattern, KeywordArg('convert_dtype_after_inplace_add'), int8_mixed_bf16_with_inplace_add)

def generate_pattern_with_unary(computation_call, unary_post_op):
    if False:
        return 10
    if unary_post_op is not None:
        return CallFunction(unary_post_op, computation_call)
    return computation_call

def generate_pattern_with_output_quant(computation_call, dtype=torch.float32):
    if False:
        for i in range(10):
            print('nop')
    '\n    quantize output:\n        output = round(output * o_inv_scale)\n        output = output + zero_point\n        output = clamp_min(output, 0)\n        output = clamp_max(output, 127)\n        output = output.to(uint8)\n    '
    assert dtype in [torch.float32, torch.bfloat16]
    quantized_op_output_pattern_pt2e = CallFunction(prims.convert_element_type.default, CallFunction(aten.clamp_max.default, CallFunction(aten.clamp_min.default, CallFunction(aten.add.Tensor, CallFunction(aten.round.default, CallFunction(aten.mul.Tensor, _may_generate_pattern_with_dtype_convert(computation_call, KeywordArg('autocast_output_quant_dtype'), dtype != torch.float32), KeywordArg('o_inv_scale'))), KeywordArg('o_zp')), KeywordArg('o_qmin')), KeywordArg('o_qmax')), KeywordArg('o_dtype'))
    return quantized_op_output_pattern_pt2e

def _check_node_kwarg_arg_value(check_node, kwarg_name, args_index, expected_value):
    if False:
        return 10
    if kwarg_name in check_node.kwargs:
        actual_value = check_node.kwargs[kwarg_name]
        return actual_value == expected_value
    else:
        assert len(check_node.args) >= args_index + 1
        actual_value = check_node.args[args_index]
        return actual_value == expected_value

def _is_valid_quantized_conv2d_optimization_pattern(output_dtype):
    if False:
        return 10

    def fn(match):
        if False:
            i = 10
            return i + 15
        if output_dtype is not None:
            qconv_node_after_weight_prepack = filter_nodes(match.nodes, torch.ops.onednn.qconv2d_pointwise)[0]
            return _check_node_kwarg_arg_value(qconv_node_after_weight_prepack, 'output_dtype', 13, output_dtype)
        return True
    return fn

def _register_quantized_conv_lowering(pattern, pass_number, computation_op, output_dtype, unary_attr, original_pattern_output_dtype=torch.float32):
    if False:
        while True:
            i = 10

    @register_lowering_pattern(pattern, extra_check=_is_valid_quantized_conv2d_optimization_pattern(output_dtype), pass_number=pass_number)
    def qconv(match: Match, *args, **kwargs):
        if False:
            return 10
        (x, x_scale, x_zp) = (kwargs['x'], kwargs['x_scale'], kwargs['x_zp'])
        (packed_weight, w_scale, w_zp) = (kwargs['packed_weight'], kwargs['w_scale'], kwargs['w_zp'])
        (b, stride, padding, dilation, groups) = (kwargs['b'], kwargs['stride'], kwargs['padding'], kwargs['dilation'], kwargs['groups'])
        assert output_dtype in [None, torch.float32, torch.bfloat16]
        o_inv_scale = kwargs['o_inv_scale'] if output_dtype is None else 1.0
        o_zero_point = kwargs['o_zp'] if output_dtype is None else 0
        assert kwargs['output_dtype'] is original_pattern_output_dtype
        assert kwargs['attr'] == 'none'
        computation_args = (x, x_scale, x_zp, packed_weight, w_scale, w_zp, b, stride, padding, dilation, groups, o_inv_scale, o_zero_point, output_dtype, unary_attr.op_name, unary_attr.scalars_attr, unary_attr.algorithm_attr)
        counters['inductor']['qconv2d_unary_matcher_count'] += 1
        counters['inductor']['qconv2d_unary_matcher_nodes'] += len(match.nodes)
        return L[computation_op](*computation_args)
    return qconv

def _is_valid_quantized_linear_optimization_pattern(output_dtype):
    if False:
        print('Hello World!')

    def fn(match):
        if False:
            i = 10
            return i + 15
        if output_dtype is not None:
            qlinear_node_after_weight_prepack = filter_nodes(match.nodes, torch.ops.onednn.qlinear_pointwise)[0]
            return _check_node_kwarg_arg_value(qlinear_node_after_weight_prepack, 'output_dtype', 9, output_dtype)
        return True
    return fn

def _register_quantized_linear_lowering(pattern, pass_number, computation_op, output_dtype, unary_attr, original_pattern_output_dtype=torch.float32):
    if False:
        while True:
            i = 10

    @register_lowering_pattern(pattern, extra_check=_is_valid_quantized_linear_optimization_pattern(output_dtype), pass_number=pass_number)
    def qlinear(match: Match, *args, **kwargs):
        if False:
            while True:
                i = 10
        (x, x_scale, x_zp) = (kwargs['x'], kwargs['x_scale'], kwargs['x_zp'])
        (packed_weight, w_scale, w_zp) = (kwargs['packed_weight'], kwargs['w_scale'], kwargs['w_zp'])
        b = kwargs['b'] if 'b' in kwargs else None
        o_inv_scale = kwargs['o_inv_scale'] if output_dtype is None else 1.0
        o_zero_point = kwargs['o_zp'] if output_dtype is None else 0
        assert kwargs['output_dtype'] is original_pattern_output_dtype
        assert kwargs['postop_name'] == 'none'
        computation_args = (x, x_scale, x_zp, packed_weight, w_scale, w_zp, b, o_inv_scale, o_zero_point, output_dtype, unary_attr.op_name, unary_attr.scalars_attr, unary_attr.algorithm_attr)
        counters['inductor']['qlinear_unary_matcher_count'] += 1
        counters['inductor']['qlinear_unary_matcher_nodes'] += len(match.nodes)
        return L[computation_op](*computation_args)
    return qlinear

def _register_quantized_conv_binary_lowering(pattern, pass_number, computation_op, output_dtype, binary_unary_attr):
    if False:
        return 10

    @register_lowering_pattern(pattern, pass_number=pass_number)
    def qconv_binary(match: Match, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (x, x_scale, x_zp) = (kwargs['x'], kwargs['x_scale'], kwargs['x_zp'])
        accum = kwargs['accum'] if output_dtype is None else kwargs['accum_after_dequant']
        accum_scale = kwargs['accum_scale'] if output_dtype is None else 1.0
        accum_zp = kwargs['accum_zp'] if output_dtype is None else 0
        (packed_weight, w_scale, w_zp) = (kwargs['packed_weight'], kwargs['w_scale'], kwargs['w_zp'])
        (b, stride, padding, dilation, groups) = (kwargs['b'], kwargs['stride'], kwargs['padding'], kwargs['dilation'], kwargs['groups'])
        o_inv_scale = kwargs['o_inv_scale'] if output_dtype is None else 1.0
        o_zero_point = kwargs['o_zp'] if output_dtype is None else 0
        computation_args = (x, x_scale, x_zp, accum, accum_scale, accum_zp, packed_weight, w_scale, w_zp, b, stride, padding, dilation, groups, o_inv_scale, o_zero_point, output_dtype, binary_unary_attr.binary_op_name, binary_unary_attr.alpha, binary_unary_attr.unary_op_name, binary_unary_attr.scalars_attr, binary_unary_attr.algorithm_attr)
        counters['inductor']['qconv2d_binary_matcher_count'] += 1
        counters['inductor']['qconv2d_binary_matcher_nodes'] += len(match.nodes)
        return L[computation_op](*computation_args)
    return qconv_binary

def _register_quantization_unary_fusion():
    if False:
        i = 10
        return i + 15

    class UnaryAttr:

        def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
            if False:
                i = 10
                return i + 15
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ''
    for original_pattern_output_dtype in [torch.float32, torch.bfloat16]:
        conv_unary_replace_patterns = {UnaryAttr('none', [], ''): generate_pattern_with_output_quant(dequantize_qconv_pt2e_pattern, dtype=original_pattern_output_dtype), UnaryAttr('relu', [], ''): generate_pattern_with_output_quant(generate_pattern_with_unary(dequantize_qconv_pt2e_pattern, aten.relu.default), dtype=original_pattern_output_dtype)}
        for (unary_attr, patterns) in conv_unary_replace_patterns.items():
            _register_quantized_conv_lowering(patterns, 1, torch.ops.onednn.qconv2d_pointwise, None, unary_attr, original_pattern_output_dtype=original_pattern_output_dtype)
        conv_unary_replace_float_out_patterns = {UnaryAttr('relu', [], ''): generate_pattern_with_unary(dequantize_qconv_pt2e_pattern, aten.relu.default)}
        for (unary_attr, patterns) in conv_unary_replace_float_out_patterns.items():
            _register_quantized_conv_lowering(patterns, 2, torch.ops.onednn.qconv2d_pointwise, original_pattern_output_dtype, unary_attr, original_pattern_output_dtype=original_pattern_output_dtype)
        linear_unary_replace_patterns = {UnaryAttr('none', [], ''): generate_pattern_with_output_quant(qlinear_pt2e_pattern, dtype=original_pattern_output_dtype), UnaryAttr('relu', [], ''): generate_pattern_with_output_quant(generate_pattern_with_unary(qlinear_pt2e_pattern, aten.relu.default), dtype=original_pattern_output_dtype)}
        for (unary_attr, patterns) in linear_unary_replace_patterns.items():
            _register_quantized_linear_lowering(patterns, 1, torch.ops.onednn.qlinear_pointwise, None, unary_attr, original_pattern_output_dtype=original_pattern_output_dtype)
        linear_unary_replace_float_out_patterns = {UnaryAttr('relu', [], ''): generate_pattern_with_unary(qlinear_pt2e_pattern, aten.relu.default)}
        for (unary_attr, patterns) in linear_unary_replace_float_out_patterns.items():
            _register_quantized_linear_lowering(patterns, 2, torch.ops.onednn.qlinear_pointwise, original_pattern_output_dtype, unary_attr, original_pattern_output_dtype=original_pattern_output_dtype)

def _register_quantization_binary_fusion():
    if False:
        i = 10
        return i + 15

    class BinaryUnaryAttr:

        def __init__(self, binary_op_name: str, alpha=None, unary_op_name: str='none', scalars_attr=None, algorithm_attr=None):
            if False:
                while True:
                    i = 10
            self.binary_op_name = binary_op_name
            self.alpha = alpha if alpha else 1.0
            self.unary_op_name = unary_op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ''
    for int8_mixed_bf16_with_inplace_add in [False, True]:
        binary_replace_patterns = {BinaryUnaryAttr('add', 1.0, 'none', [], ''): generate_pattern_with_output_quant(generate_pattern_with_binary(aten.add.Tensor, dequantize_qconv_pt2e_pattern, dequantize_accum_pattern, int8_mixed_bf16_with_inplace_add), dtype=torch.bfloat16 if int8_mixed_bf16_with_inplace_add else torch.float32), BinaryUnaryAttr('add', 1.0, 'relu', [], ''): generate_pattern_with_output_quant(generate_pattern_with_unary(generate_pattern_with_binary(aten.add.Tensor, dequantize_qconv_pt2e_pattern, dequantize_accum_pattern, int8_mixed_bf16_with_inplace_add), aten.relu.default), dtype=torch.bfloat16 if int8_mixed_bf16_with_inplace_add else torch.float32)}
        for (binary_unary_attr, patterns) in binary_replace_patterns.items():
            _register_quantized_conv_binary_lowering(patterns, 0, torch.ops.onednn.qconv2d_pointwise.binary, None, binary_unary_attr)
        binary_replace_float_out_patterns = {BinaryUnaryAttr('add', 1.0, 'relu', [], ''): generate_pattern_with_unary(generate_pattern_with_binary(aten.add.Tensor, dequantize_qconv_pt2e_pattern, KeywordArg('accum_after_dequant'), int8_mixed_bf16_with_inplace_add), aten.relu.default)}
        for (binary_unary_attr, patterns) in binary_replace_float_out_patterns.items():
            if int8_mixed_bf16_with_inplace_add:
                _register_quantized_conv_binary_lowering(patterns, 0, torch.ops.onednn.qconv2d_pointwise.binary, torch.bfloat16, binary_unary_attr)
            else:
                _register_quantized_conv_binary_lowering(patterns, 1, torch.ops.onednn.qconv2d_pointwise.binary, torch.float32, binary_unary_attr)
        binary_replace_float_out_patterns = {BinaryUnaryAttr('add', 1.0, 'none', [], ''): generate_pattern_with_binary(aten.add.Tensor, dequantize_qconv_pt2e_pattern, KeywordArg('accum_after_dequant'), int8_mixed_bf16_with_inplace_add)}
        for (binary_unary_attr, patterns) in binary_replace_float_out_patterns.items():
            _register_quantized_conv_binary_lowering(patterns, 1 if int8_mixed_bf16_with_inplace_add else 2, torch.ops.onednn.qconv2d_pointwise.binary, torch.bfloat16 if int8_mixed_bf16_with_inplace_add else torch.float32, binary_unary_attr)

def _is_valid_quantized_maxpool2d_optimization_pattern():
    if False:
        for i in range(10):
            print('nop')

    def fn(match):
        if False:
            i = 10
            return i + 15
        get_item_node = filter_nodes(match.nodes, operator.getitem)[0]
        return get_item_node.args[1] == 0
    return fn

def _register_quantized_maxpool2d_lowering(pattern, computation_op):
    if False:
        print('Hello World!')

    @register_lowering_pattern(pattern, extra_check=_is_valid_quantized_maxpool2d_optimization_pattern())
    def qmaxpool2d(match: Match, *args, **kwargs):
        if False:
            return 10
        x = kwargs['x']
        kernel_size = kwargs['kernel_size']
        stride = kwargs['stride'] if 'stride' in kwargs else None
        padding = kwargs['padding'] if 'padding' in kwargs else 0
        dilation = kwargs['dilation'] if 'dilation' in kwargs else 1
        ceil_mode = kwargs['ceil_mode'] if 'ceil_mode' in kwargs else False
        if padding == 0:
            padding = [0, 0]
        if dilation == 1:
            dilation = [1, 1]
        if not stride:
            stride = kernel_size
        kernel_size = pad_listlike(kernel_size, 2)
        stride = pad_listlike(stride, 2)
        padding = pad_listlike(padding, 2)
        dilation = pad_listlike(dilation, 2)
        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2
        assert len(dilation) == 2
        computation_args = (x, kernel_size, stride, padding, dilation, ceil_mode)
        (computation_args, _) = require_channels_last(computation_op, *computation_args)
        return L[computation_op](*computation_args)
    return qmaxpool2d

def _register_quantization_maxpool2d():
    if False:
        print('Hello World!')
    max_pool2d_args_list = [[KeywordArg('stride')], [KeywordArg('stride'), KeywordArg('padding')], [KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation')], [KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('ceil_mode')]]
    for max_pool2d_args in max_pool2d_args_list:
        dequantize_maxpool2d_pattern = CallFunction(aten.max_pool2d_with_indices.default, dequantize_per_tensor_activation_pattern, KeywordArg('kernel_size'), *max_pool2d_args)
        dequantize_maxpool2d_get_item_pattern = CallFunction(operator.getitem, dequantize_maxpool2d_pattern, Arg())
        _register_quantized_maxpool2d_lowering(generate_pattern_with_output_quant(dequantize_maxpool2d_get_item_pattern), quantized.max_pool2d.default)

def _is_valid_quantized_cat_optimization_pattern():
    if False:
        for i in range(10):
            print('nop')

    def fn(match):
        if False:
            while True:
                i = 10
        sub_nodes = filter_nodes(match.nodes, aten.sub.Tensor)
        zero_points = [node.args[1] for node in sub_nodes]
        add_nodes = filter_nodes(match.nodes, aten.add.Tensor)
        assert len(add_nodes) == 1, 'expect only 1 add node at output quant pattern'
        zero_points.append(add_nodes[0].args[1])
        if not all((zero_point == zero_points[0] for zero_point in zero_points)):
            return False
        mul_nodes = filter_nodes(match.nodes, aten.mul.Tensor)
        scales = [mul_node.args[1] if mul_node.args[0].target is aten.cat.default else 1.0 / mul_node.args[1] for mul_node in mul_nodes]
        if not all((math.isclose(scale, scales[0], rel_tol=1e-05) for scale in scales)):
            return False
        return True
    return fn

def _register_quantized_cat_lowering(pattern, computation_op):
    if False:
        print('Hello World!')

    @register_lowering_pattern(pattern, extra_check=_is_valid_quantized_cat_optimization_pattern())
    def qcat(match: Match, inputs, dim, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        uint8_inputs = [input[0] for input in inputs]
        return L[computation_op](uint8_inputs, dim)
    return qcat
_raw_dequantize_per_tensor_activation_pattern = CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, Arg(), Arg()), Arg()), Arg())

def _register_quantization_cat():
    if False:
        while True:
            i = 10
    dequantize_cat_pattern = CallFunction(aten.cat.default, ListOf(_raw_dequantize_per_tensor_activation_pattern), KeywordArg('dim'))
    _register_quantized_cat_lowering(generate_pattern_with_output_quant(dequantize_cat_pattern), aten.cat)

def _register_quantization_lowerings():
    if False:
        while True:
            i = 10
    _register_quantization_unary_fusion()
    _register_quantization_binary_fusion()
    _register_quantization_maxpool2d()
    _register_quantization_cat()

def _is_valid_dequant_promotion_pattern(dtype=torch.float32):
    if False:
        while True:
            i = 10

    def _inner(match):
        if False:
            for i in range(10):
                print('nop')
        assert dtype in [torch.float32, torch.bfloat16]
        if dtype == torch.float32:
            mul_node = match.output_node()
        else:
            convert_to_bf16_node = match.output_node()
            mul_node = convert_to_bf16_node.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        if mul_node.target is aten.mul.Tensor and sub_node.target is aten.sub.Tensor and (to_fp32_node.target is prims.convert_element_type.default) and (len(list(mul_node.users)) > 1) if dtype == torch.float32 else len(list(convert_to_bf16_node.users)) > 1:
            return True
        return False
    return _inner

def _register_dequant_promotion_pass(pattern, pass_number, dtype=torch.float32):
    if False:
        for i in range(10):
            print('nop')

    @register_freezing_graph_pattern(pattern, extra_check=_is_valid_dequant_promotion_pattern(dtype), pass_number=pass_number)
    def dequant_promotion(match: Match, *args, **kwargs):
        if False:
            print('Hello World!')
        assert dtype in [torch.float32, torch.bfloat16]

        def clone_to_new_node(graph, source_node, user_node):
            if False:
                return 10
            assert source_node.op == 'call_function', 'clone_to_new_node only support node.op call_function'
            with graph.inserting_before(user_node):
                new_node = graph.call_function(source_node.target, args=source_node.args, kwargs=source_node.kwargs)
                new_node.meta = copy.copy(source_node.meta)
                user_node.replace_input_with(source_node, new_node)
            return new_node
        if dtype == torch.float32:
            mul_node = match.output_node()
        else:
            convert_to_bf16_node = match.output_node()
            mul_node = convert_to_bf16_node.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        assert mul_node.target is aten.mul.Tensor
        assert sub_node.target is aten.sub.Tensor
        assert to_fp32_node.target is prims.convert_element_type.default
        graph = match.graph
        user_node_list = list(mul_node.users) if dtype == torch.float32 else list(convert_to_bf16_node.users)
        for user_node in user_node_list:
            if dtype == torch.float32:
                new_mul_node = clone_to_new_node(graph, mul_node, user_node)
            else:
                new_convert_to_bf16_node_node = clone_to_new_node(graph, convert_to_bf16_node, user_node)
                new_mul_node = clone_to_new_node(graph, mul_node, new_convert_to_bf16_node_node)
            new_sub_node = clone_to_new_node(graph, sub_node, new_mul_node)
            _ = clone_to_new_node(graph, to_fp32_node, new_sub_node)
        counters['inductor']['dequant_promotion_matcher_count'] += 1
        counters['inductor']['dequant_promotion_matcher_nodes'] += len(match.nodes)

def _is_valid_dequant_conv2d_pattern(dtype):
    if False:
        print('Hello World!')

    def _inner(match):
        if False:
            i = 10
            return i + 15
        conv_node = match.output_node()
        assert conv_node.target is aten.convolution.default
        input_meta_value = conv_node.args[0].meta.get('val')
        weight_meta_value = conv_node.args[1].meta.get('val')
        for meta_value in [input_meta_value, weight_meta_value]:
            if meta_value is None or meta_value.device.type != 'cpu' or meta_value.dim() != 4:
                return False
        assert dtype in [torch.float32, torch.bfloat16]
        if dtype == torch.float32:
            mul_node = conv_node.args[0]
        else:
            convert_to_bf16 = conv_node.args[0]
            mul_node = convert_to_bf16.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        assert to_fp32_node.target is prims.convert_element_type.default
        assert sub_node.target is aten.sub.Tensor
        assert mul_node.target is aten.mul.Tensor
        if len(list(to_fp32_node.users)) != 1 or len(list(sub_node.users)) != 1 or len(list(mul_node.users)) != 1:
            return False
        return True
    return _inner

def _register_qconv_weight_prepack_pass(pattern, pass_number, dtype=torch.float32):
    if False:
        for i in range(10):
            print('nop')

    @register_freezing_graph_pattern(pattern, extra_check=_is_valid_dequant_conv2d_pattern(dtype), pass_number=pass_number)
    def qconv_weight_prepack(match: Match, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Match the pattern:\n        int8 activation\n          |\n        dequant_per_tensor\n          |\n        Conv2d <- optional(aten.clone.default) <- dequant_per_channel <- int8_weight\n\n        Insert weight prepack node and change the pattern to:\n        int8 activation\n          |\n        onednn.qconv2d_pointwise <- onednn.qconv_prepack <- int8_weight\n        '
        assert dtype in [torch.float32, torch.bfloat16]
        conv_node = match.output_node()
        assert conv_node.target is aten.convolution.default
        if dtype == torch.float32:
            mul_node = conv_node.args[0]
        else:
            convert_to_bf16 = conv_node.args[0]
            mul_node = convert_to_bf16.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        has_clone_to_channel_last_node_in_pattern = conv_node.args[1].target is aten.clone.default
        clone_node = conv_node.args[1] if has_clone_to_channel_last_node_in_pattern else None
        if dtype == torch.float32:
            dequant_per_channel = clone_node.args[0] if has_clone_to_channel_last_node_in_pattern else conv_node.args[1]
        else:
            weight_to_bf16_node = clone_node.args[0] if has_clone_to_channel_last_node_in_pattern else conv_node.args[1]
            dequant_per_channel = weight_to_bf16_node.args[0]
        assert dequant_per_channel.target is quantized_decomposed.dequantize_per_channel.default
        (qx, x_zp, x_scale) = (kwargs['x'], kwargs['x_zp'], kwargs['x_scale'])
        (qw, w_scale, w_zp) = (kwargs['q_weight'], kwargs['w_scale'], kwargs['w_zp'])
        (bias, stride, padding, dilation, groups) = (kwargs['b'], kwargs['stride'], kwargs['padding'], kwargs['dilation'], kwargs['groups'])
        x_shape = qx.meta.get('tensor_meta').shape
        if has_free_symbols(x_shape):
            x_shape = None
        graph = match.graph
        with graph.inserting_before(conv_node):
            packed_weight_inputs = (qw, w_scale, x_scale, x_zp, stride, padding, dilation, groups, x_shape)
            packed_weight_op = torch.ops.onednn.qconv_prepack
            prepack_weight_node = graph.call_function(packed_weight_op, args=packed_weight_inputs)
            new_args: Tuple[Any, ...] = (qx, x_scale, x_zp, prepack_weight_node, w_scale, w_zp, bias, stride, padding, dilation, groups, 1.0, 0, dtype, 'none', [], '')
            new_conv_node = graph.call_function(torch.ops.onednn.qconv2d_pointwise.default, args=new_args)
            conv_node.replace_all_uses_with(new_conv_node)
            new_conv_node.meta.update(conv_node.meta)
            graph.erase_node(conv_node)
            if dtype == torch.bfloat16:
                graph.erase_node(convert_to_bf16)
            graph.erase_node(mul_node)
            graph.erase_node(sub_node)
            graph.erase_node(to_fp32_node)
            if clone_node is not None:
                graph.erase_node(clone_node)
            if dtype == torch.bfloat16:
                graph.erase_node(weight_to_bf16_node)
            graph.erase_node(dequant_per_channel)
            counters['inductor']['qconv2d_weight_prepack_matcher_count'] += 1
            counters['inductor']['qconv2d_weight_prepack_matcher_nodes'] += len(match.nodes)

def _generate_dequant_convolution_node_pattern(_dequant_per_channel_pattern, dtype=torch.float32):
    if False:
        i = 10
        return i + 15
    assert dtype in [torch.float32, torch.bfloat16]
    dequant_convolution_node_pattern = CallFunction(aten.convolution.default, _may_generate_pattern_with_dtype_convert(dequantize_per_tensor_activation_pattern, KeywordArg('autocast_act_dtype'), dtype != torch.float32), _dequant_per_channel_pattern, KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))
    return dequant_convolution_node_pattern

def _generate_qconv_weight_prepack_patterns(dtype=torch.float32):
    if False:
        while True:
            i = 10
    assert dtype in [torch.float32, torch.bfloat16]
    return (_generate_dequant_convolution_node_pattern(dequantize_per_channel_weight_pattern if dtype == torch.float32 else dequantize_per_channel_to_bf16_weight_pattern, dtype), _generate_dequant_convolution_node_pattern(dequantize_per_channel_clone_weight_pattern if dtype == torch.float32 else dequantize_per_channel_to_bf16_clone_weight_pattern, dtype))

def _is_valid_dequant_linear_pattern(dtype):
    if False:
        for i in range(10):
            print('nop')

    def _inner(match):
        if False:
            print('Hello World!')
        linear_node = match.output_node()
        assert linear_node.target in (aten.addmm.default, aten.mm.default)
        input_index = 0 if linear_node.target is aten.mm.default else 1
        assert dtype in [torch.float32, torch.bfloat16]
        if dtype == torch.float32:
            mul_node = linear_node.args[input_index]
        else:
            convert_to_bf16 = linear_node.args[input_index]
            mul_node = convert_to_bf16.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        assert to_fp32_node.target is prims.convert_element_type.default
        assert sub_node.target is aten.sub.Tensor
        assert mul_node.target is aten.mul.Tensor
        if len(list(to_fp32_node.users)) != 1 or len(list(sub_node.users)) != 1 or len(list(mul_node.users)) != 1:
            return False
        return True
    return _inner

def _register_qlinear_weight_prepack_pass(pattern, pass_number, dtype=torch.float32):
    if False:
        i = 10
        return i + 15

    @register_freezing_graph_pattern(pattern, extra_check=_is_valid_dequant_linear_pattern(dtype), pass_number=pass_number)
    def qlinear_weight_prepack(match: Match, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Match the pattern:\n        int8 activation\n          |\n        dequant_per_tensor\n          |\n        mm/addmm <- t <- dequant_per_channel <- int8_weight\n\n        Insert weight prepack node and change the pattern to:\n        int8 activation\n          |\n        onednn.qlinear_pointwise <- onednn.qlinear_prepack <- int8_weight\n        '
        assert dtype in [torch.float32, torch.bfloat16]
        linear_node = match.output_node()
        assert linear_node.target in (aten.addmm.default, aten.mm.default)
        input_index = 0 if linear_node.target is aten.mm.default else 1
        weight_index = input_index + 1
        if dtype == torch.float32:
            mul_node = linear_node.args[input_index]
        else:
            activation_to_bf16_node = linear_node.args[input_index]
            mul_node = activation_to_bf16_node.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        t_node = linear_node.args[weight_index]
        if dtype == torch.float32:
            dequant_per_channel = t_node.args[0]
        else:
            weight_to_bf16_node = t_node.args[0]
            dequant_per_channel = weight_to_bf16_node.args[0]
        assert dequant_per_channel.target is quantized_decomposed.dequantize_per_channel.default
        (qx, x_zp, x_scale) = (kwargs['x'], kwargs['x_zp'], kwargs['x_scale'])
        (qw, w_scale, w_zp) = (kwargs['q_weight'], kwargs['w_scale'], kwargs['w_zp'])
        bias = kwargs['b'] if 'b' in kwargs else None
        x_shape = qx.meta.get('tensor_meta').shape
        if has_free_symbols(x_shape):
            x_shape = None
        graph = match.graph
        with graph.inserting_before(linear_node):
            packed_weight_inputs = (qw, x_shape)
            packed_weight_op = torch.ops.onednn.qlinear_prepack
            prepack_weight_node = graph.call_function(packed_weight_op, args=packed_weight_inputs)
            new_args: Tuple[Any, ...] = (qx, x_scale, x_zp, prepack_weight_node, w_scale, w_zp, bias, 1.0, 0, dtype, 'none', [], '')
            new_linear_node = graph.call_function(torch.ops.onednn.qlinear_pointwise.default, args=new_args)
            linear_node.replace_all_uses_with(new_linear_node)
            new_linear_node.meta.update(linear_node.meta)
            graph.erase_node(linear_node)
            if dtype == torch.bfloat16:
                graph.erase_node(activation_to_bf16_node)
            graph.erase_node(mul_node)
            graph.erase_node(sub_node)
            graph.erase_node(to_fp32_node)
            graph.erase_node(t_node)
            if dtype == torch.bfloat16:
                graph.erase_node(weight_to_bf16_node)
            graph.erase_node(dequant_per_channel)
            counters['inductor']['qlinear_weight_prepack_matcher_count'] += 1
            counters['inductor']['qlinear_weight_prepack_matcher_nodes'] += len(match.nodes)

def _generate_dequant_linear_node_pattern(_dequant_per_channel_pattern, dtype=torch.float32):
    if False:
        for i in range(10):
            print('nop')
    t_pattern = CallFunction(aten.permute.default, _may_generate_pattern_with_dtype_convert(_dequant_per_channel_pattern, KeywordArg('autocast_wgt_dtype'), dtype != torch.float32), KeywordArg('permute_axes'))
    dequant_linear_bias_pattern = CallFunction(aten.addmm.default, KeywordArg('b'), _may_generate_pattern_with_dtype_convert(dequantize_per_tensor_activation_pattern, KeywordArg('autocast_act_dtype'), dtype != torch.float32), t_pattern)
    dequant_linear_no_bias_pattern = CallFunction(aten.mm.default, _may_generate_pattern_with_dtype_convert(dequantize_per_tensor_activation_pattern, KeywordArg('autocast_act_dtype'), dtype != torch.float32), t_pattern)
    return (dequant_linear_bias_pattern, dequant_linear_no_bias_pattern)

def _generate_qlinear_weight_prepack_patterns(dtype=torch.float32):
    if False:
        return 10
    return _generate_dequant_linear_node_pattern(dequantize_per_channel_weight_pattern, dtype)

@functools.lru_cache(None)
def _register_quantization_weight_pack_pass():
    if False:
        i = 10
        return i + 15
    for dtype in [torch.float32, torch.bfloat16]:
        _register_dequant_promotion_pass(_may_generate_pattern_with_dtype_convert(dequantize_per_tensor_activation_pattern, KeywordArg('autocast_act_dtype'), dtype != torch.float32), pass_number=0, dtype=dtype)
        weight_prepack_patterns = _generate_qconv_weight_prepack_patterns(dtype)
        for weight_prepack_pattern in weight_prepack_patterns:
            _register_qconv_weight_prepack_pass(weight_prepack_pattern, pass_number=1, dtype=dtype)
        weight_prepack_patterns = _generate_qlinear_weight_prepack_patterns(dtype)
        for weight_prepack_pattern in weight_prepack_patterns:
            _register_qlinear_weight_prepack_pass(weight_prepack_pattern, pass_number=1, dtype=dtype)