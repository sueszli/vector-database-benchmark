"""Tests for tensorflow.python.framework.python_api_info."""
from absl.testing import parameterized
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_api_info
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

def Const(x):
    if False:
        i = 10
        return i + 15
    return constant_op.constant(x)

@test_util.run_all_in_graph_and_eager_modes
class PythonAPIInfoTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            return 10
        context.ensure_initialized()
        super(PythonAPIInfoTest, self).setUp()

    def makeConverterForGenOp(self, op_name):
        if False:
            i = 10
            return i + 15
        'Returns a PythonAPIInfo for the given gen_op.'
        api_info = _pywrap_python_api_info.PythonAPIInfo(op_name)
        api_info.InitializeFromRegisteredOp(op_name)
        return api_info

    def makeConverterFromParamSpecs(self, api_name, param_names, input_specs, attr_specs, defaults=()):
        if False:
            return 10
        'Returns a PythonAPIInfo built from the given specs.'
        api_info = _pywrap_python_api_info.PythonAPIInfo(api_name)
        api_info.InitializeFromParamSpecs(input_specs, attr_specs, param_names, defaults)
        return api_info

    @parameterized.named_parameters([('RegexFullMatch', 'RegexFullMatch', "DebugInfo for RegexFullMatch:\n  param_names=[input, pattern, name]\n  defaults_tuple=('RegexFullMatch',)\n  inputs=[\n    {index=0, name=input, is_list=0},\n    {index=1, name=pattern, is_list=0},]\n  inputs_with_fixed_dtype=[\n    {index=0, dtype=DT_STRING, is_list=0},\n    {index=1, dtype=DT_STRING, is_list=0},]\n"), ('Abs', 'Abs', "DebugInfo for Abs:\n  param_names=[x, name]\n  defaults_tuple=('Abs',)\n  attributes=[\n    {inferred_index=0, name=T, type=type},]\n  inputs=[\n    {index=0, name=x, is_list=0},]\n  inputs_with_type_attr=[\n    {type_attr=T, tensor_params=[0], ok_dtypes=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64]},]\n  inferred_type_attrs=[T]\n"), ('AddV2', 'AddV2', "DebugInfo for AddV2:\n  param_names=[x, y, name]\n  defaults_tuple=('AddV2',)\n  attributes=[\n    {inferred_index=0, name=T, type=type},]\n  inputs=[\n    {index=0, name=x, is_list=0},\n    {index=1, name=y, is_list=0},]\n  inputs_with_type_attr=[\n    {type_attr=T, tensor_params=[0, 1], ok_dtypes=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128]},]\n  inferred_type_attrs=[T]\n"), ('GatherV2', 'GatherV2', "DebugInfo for GatherV2:\n  param_names=[params, indices, axis, batch_dims, name]\n  defaults_tuple=(0, 'GatherV2')\n  attributes=[\n    {index=3, name=batch_dims, type=int},\n    {inferred_index=0, name=Tparams, type=type},\n    {inferred_index=1, name=Tindices, type=type},\n    {inferred_index=2, name=Taxis, type=type},]\n  inputs=[\n    {index=0, name=params, is_list=0},\n    {index=1, name=indices, is_list=0},\n    {index=2, name=axis, is_list=0},]\n  inputs_with_type_attr=[\n    {type_attr=Tparams, tensor_params=[0]},\n    {type_attr=Tindices, tensor_params=[1], ok_dtypes=[DT_INT16, DT_INT32, DT_INT64]},\n    {type_attr=Taxis, tensor_params=[2], ok_dtypes=[DT_INT32, DT_INT64]},]\n  inferred_type_attrs=[Tparams, Tindices, Taxis]\n"), ('ReduceJoin', 'ReduceJoin', "DebugInfo for ReduceJoin:\n  param_names=[inputs, reduction_indices, keep_dims, separator, name]\n  defaults_tuple=(False, '', 'ReduceJoin')\n  attributes=[\n    {index=2, name=keep_dims, type=bool},\n    {index=3, name=separator, type=string},]\n  inputs=[\n    {index=0, name=inputs, is_list=0},\n    {index=1, name=reduction_indices, is_list=0},]\n  inputs_with_fixed_dtype=[\n    {index=0, dtype=DT_STRING, is_list=0},\n    {index=1, dtype=DT_INT32, is_list=0},]\n"), ('ParseExampleV2', 'ParseExampleV2', "DebugInfo for ParseExampleV2:\n  param_names=[serialized, names, sparse_keys, dense_keys, ragged_keys, dense_defaults, num_sparse, sparse_types, ragged_value_types, ragged_split_types, dense_shapes, name]\n  defaults_tuple=('ParseExampleV2',)\n  attributes=[\n    {inferred_index=0, name=Tdense, type=list(type)},\n    {index=6, name=num_sparse, type=int},\n    {index=7, name=sparse_types, type=list(type)},\n    {index=8, name=ragged_value_types, type=list(type)},\n    {index=9, name=ragged_split_types, type=list(type)},\n    {index=10, name=dense_shapes, type=list(shape)},]\n  inputs=[\n    {index=0, name=serialized, is_list=0},\n    {index=1, name=names, is_list=0},\n    {index=2, name=sparse_keys, is_list=0},\n    {index=3, name=dense_keys, is_list=0},\n    {index=4, name=ragged_keys, is_list=0},\n    {index=5, name=dense_defaults, is_list=1},]\n  inputs_with_fixed_dtype=[\n    {index=0, dtype=DT_STRING, is_list=0},\n    {index=1, dtype=DT_STRING, is_list=0},\n    {index=2, dtype=DT_STRING, is_list=0},\n    {index=3, dtype=DT_STRING, is_list=0},\n    {index=4, dtype=DT_STRING, is_list=0},]\n  inputs_with_type_list_attrs=[\n    {type_list_attr=Tdense, tensor_list_params=[5], ok_dtypes=[DT_FLOAT, DT_INT64, DT_STRING]},]\n  inferred_type_list_attrs=[Tdense]\n"), ('BroadcastArgs', 'BroadcastArgs', "DebugInfo for BroadcastArgs:\n  param_names=[s0, s1, name]\n  defaults_tuple=('BroadcastArgs',)\n  attributes=[\n    {inferred_index=0, name=T, type=type},]\n  inputs=[\n    {index=0, name=s0, is_list=0},\n    {index=1, name=s1, is_list=0},]\n  inputs_with_type_attr=[\n    {type_attr=T, default_dtype=DT_INT32, tensor_params=[0, 1], ok_dtypes=[DT_INT32, DT_INT64]},]\n  inferred_type_attrs=[T]\n")])
    def testInitializeFromRegisteredOp(self, op_name, debug_info):
        if False:
            print('Hello World!')
        api_info = self.makeConverterForGenOp(op_name)
        self.assertEqual(api_info.DebugInfo().strip(), debug_info.strip())

    @parameterized.named_parameters([('NoParams', 'NoParams', [], {}, {}, 'DebugInfo for NoParams:\n  param_names=[]\n  defaults_tuple=()\n'), ('OnlyNameParam', 'OnlyNameParam', ['name'], {}, {}, 'DebugInfo for OnlyNameParam:\n  param_names=[name]\n  defaults_tuple=()\n'), ('SomeBinaryOp', 'SomeBinaryOp', ['x', 'y'], dict(x='T', y='T'), dict(T='type'), 'DebugInfo for SomeBinaryOp:\n  param_names=[x, y]\n  defaults_tuple=()\n  attributes=[\n    {inferred_index=0, name=T, type=type},]\n  inputs=[\n    {index=0, name=x, is_list=0},\n    {index=1, name=y, is_list=0},]\n  inputs_with_type_attr=[\n    {type_attr=T, tensor_params=[0, 1]},]\n  inferred_type_attrs=[T]\n'), ('AllAttributeTypes', 'AllAttributeTypes', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], {}, dict(a='any', b='float', c='int', d='string', e='bool', f='type', g='shape', h='tensor', i='list(any)', j='list(float)', k='list(int)', l='list(string)', m='list(bool)', n='list(type)', o='list(shape)', p='list(tensor)'), 'DebugInfo for AllAttributeTypes:\n  param_names=[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]\n  defaults_tuple=()\n  attributes=[\n    {index=0, name=a, type=any},\n    {index=1, name=b, type=float},\n    {index=2, name=c, type=int},\n    {index=3, name=d, type=string},\n    {index=4, name=e, type=bool},\n    {index=5, name=f, type=type},\n    {index=6, name=g, type=shape},\n    {index=7, name=h, type=tensor},\n    {index=8, name=i, type=list(any)},\n    {index=9, name=j, type=list(float)},\n    {index=10, name=k, type=list(int)},\n    {index=11, name=l, type=list(string)},\n    {index=12, name=m, type=list(bool)},\n    {index=13, name=n, type=list(type)},\n    {index=14, name=o, type=list(shape)},\n    {index=15, name=p, type=list(tensor)},]\n')])
    def testInitializeFromParamSpecs(self, api_name, param_names, input_specs, attr_specs, debug_info):
        if False:
            i = 10
            return i + 15
        api_info = self.makeConverterFromParamSpecs(api_name, param_names, input_specs, attr_specs)
        self.assertEqual(api_info.DebugInfo().strip(), debug_info.strip())
if __name__ == '__main__':
    googletest.main()