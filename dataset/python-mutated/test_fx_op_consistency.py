"""Test consistency between the output values of torch.onnx FX exported operators
and torch operators given the same inputs.

Usage:

    pytest test/onnx/test_op_consistency.py

    To run tests on a specific operator (e.g. torch.ceil):

    pytest test/onnx/test_op_consistency.py -k ceil
    pytest test/onnx/test_op_consistency.py -k nn_functional_scaled_dot_product_attention

    Read more on Running and writing tests:
        https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests

Note:

    When new ops are supported, please scroll down to modify the EXPECTED_SKIPS_OR_FAILS and
    TESTED_OPS lists. See "Modify this section"

"""
from __future__ import annotations
import copy
from typing import Any, Callable, Collection, Optional, Tuple, Union
import onnx_test_common
import parameterized
import torch
from onnx_test_common import skip, xfail
from torch.testing._internal import common_device_type, common_methods_invocations, common_utils
from torch.testing._internal.opinfo import core as opinfo_core
TESTED_OPS: frozenset[str] = frozenset(['abs', 'acos', 'acosh', 'add', 'addmm', 'all', 'allclose', 'amax', 'amin', 'any', 'arange', 'argmax', 'argmin', 'as_strided', 'asin', 'asinh', 'atan', 'atanh', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'baddbmm', 'bmm', 'broadcast_to', 'cat', 'ceil', 'chunk', 'clamp', 'clamp_max', 'clamp_min', 'clone', 'constant_pad_nd', 'contiguous', 'cos', 'cosh', 'cross', 'cumsum', 'div', 'dot', 'eq', 'equal', 'erf', 'exp', 'exp2', 'expand', 'expand_as', 'fill', 'flip', 'floor', 'fmod', 'full', 'full_like', 'gather', 'hstack', 'index_put', 'linalg.vector_norm', 'logit', 'mean', 'native_batch_norm', 'new_full', 'new_ones', 'new_zeros', 'nn.functional.adaptive_avg_pool1d', 'nn.functional.adaptive_avg_pool2d', 'nn.functional.adaptive_avg_pool3d', 'nn.functional.avg_pool1d', 'nn.functional.avg_pool2d', 'nn.functional.avg_pool3d', 'nn.functional.batch_norm', 'nn.functional.conv1d', 'nn.functional.cross_entropy', 'nn.functional.celu', 'nn.functional.dropout', 'nn.functional.elu', 'nn.functional.embedding', 'nn.functional.embedding_bag', 'nn.functional.max_pool1d', 'nn.functional.max_pool2d', 'nn.functional.max_pool3d', 'nn.functional.nll_loss', 'nn.functional.normalize', 'nonzero', 'rsub', 'scatter_add', 'scatter_reduce', 'square', 'stft', 'sub', 'sum', 'unflatten', 'var_mean', 'vstack'])
COMPLEX_TESTED_OPS = frozenset(['abs', 'stft'])

def xfail_torchlib_forward_compatibility(op_name: str, variant_name: str='', *, reason: str, github_issue: str, opsets: Optional[Collection[Union[int, Callable[[int], bool]]]]=None, dtypes: Optional[Collection[torch.dtype]]=None, matcher: Optional[Callable[[Any], bool]]=None, enabled_if: bool=True):
    if False:
        for i in range(10):
            print('nop')
    'Prefer using this (xfail) over skip when possible.\n\n    Only skip when the test is not failing consistently.\n    '
    return xfail(op_name, variant_name=variant_name, reason=f'{reason}. GitHub Issue: {github_issue}', opsets=opsets, dtypes=dtypes, matcher=matcher, enabled_if=enabled_if)

def skip_torchlib_forward_compatibility(op_name: str, variant_name: str='', *, reason: str, github_issue: str, opsets: Optional[Collection[Union[int, Callable[[int], bool]]]]=None, dtypes: Optional[Collection[torch.dtype]]=None, matcher: Optional[Callable[[Any], Any]]=None, enabled_if: bool=True):
    if False:
        i = 10
        return i + 15
    'Prefer using xfail_torchlib_forward_compatibility over this (skip) when possible.\n\n    Only skip when the test is not failing consistently.\n    '
    return skip(op_name, variant_name=variant_name, reason=f'{reason}. GitHub Issue: {github_issue}', opsets=opsets, dtypes=dtypes, matcher=matcher, enabled_if=enabled_if)
EXPECTED_SKIPS_OR_FAILS: Tuple[onnx_test_common.DecorateMeta, ...] = (xfail('add', dtypes=onnx_test_common.BOOL_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('Add')), xfail('add', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_script_does_not_support('Add', 'int8, int16, uint8 have type issue.')), xfail('addmm', dtypes=onnx_test_common.BOOL_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('Addmm')), xfail_torchlib_forward_compatibility('all', reason=onnx_test_common.reason_onnx_script_does_not_support('aten.all.dims'), github_issue='https://github.com/microsoft/onnxscript/pull/1084'), xfail('allclose', dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES + onnx_test_common.FLOAT_TYPES, reason=onnx_test_common.reason_dynamo_does_not_support('Allclose')), xfail('amax', dtypes=(torch.int16, *onnx_test_common.BOOL_TYPES), reason=onnx_test_common.reason_onnx_does_not_support('ReduceMin', 'bool, int16')), xfail('amin', dtypes=(torch.int16, *onnx_test_common.BOOL_TYPES), reason=onnx_test_common.reason_dynamo_does_not_support('ReduceMin', 'bool, int16')), xfail_torchlib_forward_compatibility('any', reason=onnx_test_common.reason_onnx_script_does_not_support('aten.any.dims'), github_issue='https://github.com/microsoft/onnxscript/pull/1084'), xfail('arange', dtypes=(torch.uint8,), reason=onnx_test_common.reason_onnx_script_does_not_support('Arange', 'uint8, int8')), xfail('arange', dtypes=(torch.int16, torch.int32), reason="AssertionError: The values for attribute 'shape' do not match"), xfail('argmax', dtypes=(torch.int16, torch.int64), reason=onnx_test_common.reason_onnx_runtime_does_not_support('ArgMax', 'int16, int64')), xfail('argmin', dtypes=(torch.uint8, torch.int8, torch.int16, torch.int64), reason=onnx_test_common.reason_onnx_runtime_does_not_support('ArgMin', 'uint8, int8, int16, int64')), skip('as_strided', variant_name='partial_views', reason="ONNX doesn't have partial view for tensor; [PostInline][ORT] segfaults"), xfail('baddbmm', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Matmul', 'uint8, int8, int16')), xfail('bmm', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Matmul', 'uint8, int8, int16')), skip('ceil', dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('Ceil', 'bool and int')), xfail('chunk', dtypes=onnx_test_common.BOOL_TYPES, reason=onnx_test_common.reason_onnx_runtime_does_not_support('Chunk', 'bool')), xfail('chunk', dtypes=(torch.uint8, torch.int8, torch.int16, torch.float16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Chunk', 'uint8, int8, int16, float16')), xfail('clamp', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Max', 'uint8, int8, int16')), xfail('clamp_max', dtypes=onnx_test_common.BOOL_TYPES, reason=onnx_test_common.reason_onnx_script_does_not_support('Clamp_max', 'bool')), xfail('clamp_max', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Max', 'uint8, int8, int16')), xfail('clamp_min', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Max', 'uint8, int8, int16')), xfail('clamp_min', dtypes=onnx_test_common.BOOL_TYPES, reason=onnx_test_common.reason_onnx_script_does_not_support('Clamp_min', 'bool')), xfail('constant_pad_nd', dtypes=(torch.int16,), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Constant_pad_nd', 'int16')), xfail('cumsum', dtypes=onnx_test_common.BOOL_TYPES + (torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_does_not_support('Cumsum', 'bool, uint8, int8, int16')), xfail('cumsum', dtypes=(torch.float16,), reason=onnx_test_common.reason_onnx_runtime_does_not_support('RUNTIME_EXCEPTION :             Exception during initialization: /onnxruntime_src/onnxruntime/core/framework/            allocation_planner.cc:230 int& onnxruntime::PlannerImpl::            UseCount(onnxruntime::OrtValueIndex) n >= 0 && static_cast<size_t>(n)             < ort_value_info_.size() was false.')), xfail('cross', reason=onnx_test_common.reason_onnx_script_does_not_support('linalg_cross')), xfail('dot', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_does_not_support('MatMul', 'uint8, int8, int16')), xfail('eq', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Equal', 'uint8, int8, int16')), xfail('equal', reason=onnx_test_common.reason_dynamo_does_not_support('aten.equal.default')), xfail('floor', dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('Floor', 'bool, int')), xfail('index_put', dtypes=onnx_test_common.BOOL_TYPES, reason=onnx_test_common.reason_onnx_script_does_not_support('index_put', 'bool')), xfail('index_put', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_script_does_not_support('Add', 'int8, int16')), xfail('nn.functional.adaptive_avg_pool2d', reason=onnx_test_common.reason_onnx_script_does_not_support('RecursionError:             maximum recursion depth exceeded while calling a Python object')), xfail('nn.functional.adaptive_avg_pool3d', reason=onnx_test_common.reason_onnx_script_does_not_support('aten._adaptive_avg_pool3d.default')), xfail('nn.functional.avg_pool1d', dtypes=onnx_test_common.INT_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('AveragePool', 'int')), xfail('nn.functional.avg_pool2d', dtypes=onnx_test_common.INT_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('AveragePool', 'int')), xfail('nn.functional.avg_pool3d', dtypes=onnx_test_common.INT_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('AveragePool', 'int')), xfail('nn.functional.conv1d', dtypes=(torch.int64,), reason=onnx_test_common.reason_onnx_does_not_support('Conv1d', 'int64')), xfail('nn.functional.conv2d', dtypes=(torch.int64,), reason=onnx_test_common.reason_onnx_does_not_support('Conv2d', 'int64')), xfail('nn.functional.dropout', reason=onnx_test_common.reason_dynamo_does_not_support('Dropout')), xfail('nn.functional.max_pool2d', dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('Max_pool2d')), xfail('nn.functional.max_pool3d', dtypes=onnx_test_common.BOOL_TYPES + onnx_test_common.INT_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('Max_pool3d')), xfail('nonzero', dtypes=(torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('NonZero', 'int8, int16')), xfail('rsub', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Mul', 'uint8, int8, int16')), xfail('scatter_add', dtypes=(torch.float16,), reason=onnx_test_common.reason_onnx_runtime_does_not_support('ScatterElements reduction=sum', 'float16')), xfail('scatter_reduce', variant_name='sum', dtypes=(torch.float16,), reason=onnx_test_common.reason_onnx_runtime_does_not_support('ScatterElements reduction=sum', 'float16')), xfail('scatter_reduce', variant_name='prod', dtypes=(torch.float16,), reason=onnx_test_common.reason_onnx_runtime_does_not_support('ScatterElements reduction=prod', 'float16')), xfail('scatter_reduce', variant_name='amin', dtypes=onnx_test_common.BOOL_TYPES + (torch.float16,), reason=onnx_test_common.reason_onnx_runtime_does_not_support('ScatterElements reduction=amin', 'float16')), xfail('scatter_reduce', variant_name='amax', dtypes=onnx_test_common.BOOL_TYPES + (torch.float16,), reason=onnx_test_common.reason_onnx_runtime_does_not_support('ScatterElements reduction=amax', 'float16')), xfail('scatter_reduce', variant_name='mean', reason="ONNX doesn't support reduce='mean' option"), xfail('square', dtypes=(torch.int8, torch.uint8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Pow', 'int8, uint8, int16')), xfail('stft', reason=onnx_test_common.reason_dynamo_does_not_support('aten._fft_r2c.default')), xfail('sub', dtypes=(torch.uint8, torch.int8, torch.int16), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Mul', 'uint8, int8, int16')), xfail('unflatten', dtypes=onnx_test_common.BOOL_TYPES, reason=onnx_test_common.reason_onnx_does_not_support('Unflatten')))
SKIP_XFAIL_SUBTESTS: tuple[onnx_test_common.DecorateMeta, ...] = (xfail('addmm', matcher=lambda sample: sample.input.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), reason=onnx_test_common.reason_onnx_runtime_does_not_support('Gemm', 'uint8, int8, int16, int32, int64')), skip('amax', matcher=lambda sample: len(sample.input.shape) == 0, reason='Op (ReduceMax) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0'), skip('amin', matcher=lambda sample: len(sample.input.shape) == 0, reason='Op (ReduceMax) [ShapeInferenceError] axis must be in [-rank, rank-1]. input rank was 0'), skip('cat', matcher=lambda sample: sample.input[0].equal(torch.tensor([])), reason='core dump - cat does not support zero-dim tensors yet'), xfail('index_put', matcher=lambda sample: sample.args[0][0].dtype == torch.bool and sample.kwargs.get('accumulate') is False, reason=onnx_test_common.reason_dynamo_does_not_support('https://github.com/pytorch/pytorch/issues/101150')), xfail('nn.functional.avg_pool1d', matcher=lambda sample: sample.kwargs.get('ceil_mode') is True and (sample.kwargs.get('count_include_pad') is True or sample.input.shape[2] % (sample.args[0][0] if isinstance(sample.args[0], tuple) else sample.args[0]) != 0), reason="fixme: ORT doesn't match PyTorch when ceil_mode=True until opset 19"), xfail('nn.functional.avg_pool2d', matcher=lambda sample: len(sample.args) > 5 and sample.args[5] is not None or sample.kwargs.get('divisor_override') is not None, reason="ONNX doesn't support divisor_override argument"), xfail('nn.functional.avg_pool3d', matcher=lambda sample: sample.kwargs.get('ceil_mode') is True, reason="fixme: ORT doesn't match PyTorch when ceil_mode=True until opset 19"), xfail('nn.functional.avg_pool3d', matcher=lambda sample: len(sample.args) > 5 and sample.args[5] is not None or sample.kwargs.get('divisor_override') is not None, reason="ONNX doesn't support divisor_override argument"), skip('nn.functional.conv1d', matcher=lambda sample: isinstance(sample.kwargs.get('padding'), str), reason='String padding is not accepted by aten::conv1d'), skip('nn.functional.conv2d', matcher=lambda sample: isinstance(sample.kwargs.get('padding'), str), reason='String padding is not accepted by aten::conv2d'), skip('nn.functional.cross_entropy', matcher=lambda sample: not isinstance(sample.kwargs.get('weight'), int), reason='ONNX SoftmaxCrossEntropyLoss op only accept argument[weight] is int type'), skip_torchlib_forward_compatibility('nn.functional.embedding_bag', matcher=lambda sample: sample.kwargs.get('padding_idx') is not None or True, reason=onnx_test_common.reason_onnx_script_does_not_support("'padding_idx' overload for _embedding_bag and _embedding_bag_forward_only. 'padding_idx=-1' is emitted for aten op when 'padding_idx' is not provided"), github_issue='https://github.com/microsoft/onnxscript/issues/1056'), skip('nn.functional.max_pool3d', matcher=lambda sample: sample.kwargs.get('ceil_mode') is True and sample.kwargs.get('padding') == 1, reason='FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed'), xfail('nonzero', matcher=lambda sample: len(sample.input.shape) == 0 and sample.kwargs.get('as_tuple', False) is False, reason="Output 'shape' do not match: torch.Size([0, 1]) != torch.Size([0, 0])."), xfail('scatter_add', matcher=lambda sample: len(sample.input.shape) == 0, reason='fixme: Rank(0) input will lead ORT failed due to different rank(result) in if-else branch'), skip('scatter_reduce', matcher=lambda sample: sample.kwargs.get('include_self') is False, reason="ONNX does't support include_self=False option"), xfail('unflatten', reason='Logic not implemented for size 0 inputs in op.Reshape', matcher=lambda sample: any((dim == 0 for dim in sample.input.shape))))
OPS_DB = copy.deepcopy(common_methods_invocations.op_db)
OP_WITH_SKIPPED_XFAIL_SUBTESTS = frozenset((meta.op_name for meta in SKIP_XFAIL_SUBTESTS))
ALL_OPS_IN_DB = frozenset((op_info.name for op_info in OPS_DB))
assert TESTED_OPS.issubset(ALL_OPS_IN_DB), f'{TESTED_OPS - ALL_OPS_IN_DB} not in OPS_DB'

class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        if False:
            while True:
                i = 10
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        if False:
            return 10
        return self.operator(*args, **self.kwargs)

def _should_skip_xfail_test_sample(op_name: str, sample) -> Tuple[Optional[str], Optional[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a reason if a test sample should be skipped.'
    if op_name not in OP_WITH_SKIPPED_XFAIL_SUBTESTS:
        return (None, None)
    for decorator_meta in SKIP_XFAIL_SUBTESTS:
        if decorator_meta.op_name == op_name:
            assert decorator_meta.matcher is not None, 'Matcher must be defined'
            if decorator_meta.matcher(sample):
                return (decorator_meta.test_behavior, decorator_meta.reason)
    return (None, None)

def _run_test_output_match(test_suite: onnx_test_common._TestONNXRuntime, device: str, dtype: torch.dtype, op: opinfo_core.OpInfo):
    if False:
        return 10
    assert device == 'cpu'
    samples = op.sample_inputs(device, dtype, requires_grad=False)
    for (i, cpu_sample) in enumerate(samples):
        inputs = (cpu_sample.input, *cpu_sample.args)
        with test_suite.subTest(opset=test_suite.opset_version, sample_num=i, inputs=repr(inputs), kwargs=repr(cpu_sample.kwargs)):
            (test_behavior, reason) = _should_skip_xfail_test_sample(op.name, cpu_sample)
            with onnx_test_common.normal_xfail_skip_test_behaviors(test_behavior, reason):
                model = SingleOpModel(op.op, cpu_sample.kwargs)
                model.eval()
                if dtype == torch.float32:
                    rtol = 1e-05
                    atol = 2e-05
                elif dtype == torch.float16 and op.name in test_suite.fp16_low_precision_list:
                    rtol = 0.01
                    atol = 0.001
                else:
                    rtol = None
                    atol = None
                test_suite.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(model, inputs, rtol=rtol, atol=atol)

def _get_test_class_name(cls, num, params_dict) -> str:
    if False:
        for i in range(10):
            print('nop')
    del cls
    del num
    return params_dict['name']

@parameterized.parameterized_class([{'name': f'TestOnnxModelOutputConsistency_opset{opset}', 'opset_version': opset} for opset in onnx_test_common.FX_TESTED_OPSETS], class_name_func=_get_test_class_name)
class TestOnnxModelOutputConsistency(onnx_test_common._TestONNXRuntime):
    """Test output consistency between exported ONNX models and PyTorch eager mode.

    This is a parameterized test suite.
    """
    opset_version = -1
    op_level_debug: bool = False
    dynamic_shapes: bool = False
    fp16_low_precision_list = ['nn.functional.batch_norm', 'native_batch_norm', 'dot', 'logit']

    @common_device_type.ops([op for op in OPS_DB if op.name in TESTED_OPS], allowed_dtypes=onnx_test_common.TESTED_DTYPES)
    def test_output_match(self, device: str, dtype: torch.dtype, op):
        if False:
            print('Hello World!')
        'Test the ONNX exporter.'
        _run_test_output_match(self, device, dtype, op)

    @common_device_type.ops([op for op in OPS_DB if op.name in COMPLEX_TESTED_OPS], allowed_dtypes=onnx_test_common.COMPLEX_TYPES)
    def test_output_match_complex(self, device: str, dtype: torch.dtype, op):
        if False:
            i = 10
            return i + 15
        'Test the ONNX exporter with complex dtype.'
        _run_test_output_match(self, device, dtype, op)
for opset in onnx_test_common.FX_TESTED_OPSETS:
    test_class_name = f'TestOnnxModelOutputConsistency_opset{opset}'
    onnx_test_common.add_decorate_info(OPS_DB, test_class_name, 'test_output_match', opset=opset, skip_or_xfails=EXPECTED_SKIPS_OR_FAILS)
    onnx_test_common.add_decorate_info(OPS_DB, test_class_name, 'test_output_match_complex', opset=opset, skip_or_xfails=EXPECTED_SKIPS_OR_FAILS)
    common_device_type.instantiate_device_type_tests(globals()[test_class_name], globals(), only_for='cpu')
if __name__ == '__main__':
    common_utils.run_tests()