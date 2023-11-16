import argparse
import os
from api_gen import NAMESPACE_TEMPLATE, CodeGen
CPP_FILE_TEMPLATE = '\n#include <pybind11/pybind11.h>\n\n#include "paddle/fluid/pybind/static_op_function.h"\n#include "paddle/fluid/pybind/eager_op_function.h"\n#include "paddle/fluid/pybind/manual_static_op_function.h"\n#include "paddle/phi/core/enforce.h"\n#include "paddle/fluid/eager/api/utils/global_utils.h"\n\n{body}\n\n'
NAMESPACE_INNER_TEMPLATE = '\n{function_impl}\n\nstatic PyMethodDef OpsAPI[] = {{\n{ops_api}\n{{nullptr, nullptr, 0, nullptr}}\n}};\n\nvoid BindOpsAPI(pybind11::module *module) {{\n  if (PyModule_AddFunctions(module->ptr(), OpsAPI) < 0) {{\n    PADDLE_THROW(phi::errors::Fatal("Add C++ api to core.ops failed!"));\n  }}\n  if (PyModule_AddFunctions(module->ptr(), ManualOpsAPI) < 0) {{\n    PADDLE_THROW(phi::errors::Fatal("Add C++ api to core.ops failed!"));\n  }}\n}}\n'
FUNCTION_IMPL_TEMPLATE = '\nstatic PyObject *{name}(PyObject *self, PyObject *args, PyObject *kwargs) {{\n  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {{\n    VLOG(6) << "Call static_api_{name}";\n    return static_api_{name}(self, args, kwargs);\n  }} else {{\n    VLOG(6) << "Call eager_api_{name}";\n    return eager_api_{name}(self, args, kwargs);\n  }}\n}}'
STATIC_ONLY_FUNCTION_IMPL_TEMPLATE = '\nstatic PyObject *{name}(PyObject *self, PyObject *args, PyObject *kwargs) {{\n  VLOG(6) << "Call static_api_{name}";\n  return static_api_{name}(self, args, kwargs);\n}}'
OPS_API_TEMPLATE = '\n{{"{name}", (PyCFunction)(void (*)(void)){name}, METH_VARARGS | METH_KEYWORDS, "C++ interface function for {name}."}},'
NEED_GEN_STATIC_ONLY_APIS = ['fetch', 'fused_embedding_eltwise_layernorm', 'fused_fc_elementwise_layernorm', 'fused_multi_transformer_xpu', 'fused_scale_bias_relu_conv_bn', 'fused_scale_bias_add_relu', 'fusion_transpose_flatten_concat', 'generate_sequence_xpu', 'layer_norm_act_xpu', 'multi_encoder_xpu', 'multihead_matmul', 'squeeze_excitation_block', 'yolo_box_xpu', 'fusion_gru', 'fusion_seqconv_eltadd_relu', 'fusion_seqexpand_concat_fc', 'fusion_repeated_fc_relu', 'fusion_squared_mat_sub', 'fused_attention', 'fused_feedforward', 'fc', 'self_dp_attention', 'get_tensor_from_selected_rows']
NO_NEED_GEN_STATIC_ONLY_APIS = ['add_n_', 'add_n_with_kernel', 'assign_value', 'batch_norm_', 'c_allgather', 'c_allreduce_max', 'c_allreduce_sum', 'c_embedding', 'c_identity', 'c_reduce_sum', 'dpsgd', 'embedding_grad_sparse', 'fused_batch_norm_act_', 'fused_bn_add_activation_', 'fused_elemwise_add_activation', 'fused_scale_bias_relu_conv_bn', 'fused_scale_bias_add_relu', 'memcpy', 'print', 'recv_v2', 'rnn_', 'seed', 'send_v2', 'set_value', 'set_value_', 'set_value_with_tensor', 'set_value_with_tensor_', 'shadow_feed', 'sparse_momentum']

class OpsAPIGen(CodeGen):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()

    def _need_skip(self, op_info, op_name):
        if False:
            i = 10
            return i + 15
        return super()._need_skip(op_info, op_name) or op_name.endswith(('_grad', '_grad_', 'xpu')) or op_name in NO_NEED_GEN_STATIC_ONLY_APIS

    def _gen_one_function_impl(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name in NEED_GEN_STATIC_ONLY_APIS:
            return STATIC_ONLY_FUNCTION_IMPL_TEMPLATE.format(name=name)
        else:
            return FUNCTION_IMPL_TEMPLATE.format(name=name)

    def _gen_one_ops_api(self, name):
        if False:
            while True:
                i = 10
        return OPS_API_TEMPLATE.format(name=name)

    def gen_cpp_file(self, op_yaml_files, op_compat_yaml_file, namespaces, cpp_file_path):
        if False:
            return 10
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)
        op_info_items = self._parse_yaml(op_yaml_files, op_compat_yaml_file)
        function_impl_str = ''
        ops_api_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                if self._need_skip(op_info, op_name):
                    continue
                function_impl_str += self._gen_one_function_impl(op_name)
                ops_api_str += self._gen_one_ops_api(op_name)
        inner_body = NAMESPACE_INNER_TEMPLATE.format(function_impl=function_impl_str, ops_api=ops_api_str)
        body = inner_body
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(cpp_file_path, 'w') as f:
            f.write(CPP_FILE_TEMPLATE.format(body=body))

def ParseArguments():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Generate Dialect Python C Files By Yaml')
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--namespaces', type=str)
    parser.add_argument('--ops_api_file', type=str)
    return parser.parse_args()
if __name__ == '__main__':
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(',')
    op_compat_yaml_file = args.op_compat_yaml_file
    if args.namespaces is not None:
        namespaces = args.namespaces.split(',')
    ops_api_file = args.ops_api_file
    code_gen = OpsAPIGen()
    code_gen.gen_cpp_file(op_yaml_files, op_compat_yaml_file, namespaces, ops_api_file)