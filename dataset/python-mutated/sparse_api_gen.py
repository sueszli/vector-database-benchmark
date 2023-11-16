import argparse
import yaml
from api_base import PREFIX_TENSOR_NAME
from api_gen import ForwardAPI

class SparseAPI(ForwardAPI):

    def __init__(self, api_item_yaml):
        if False:
            return 10
        super().__init__(api_item_yaml)

    def gene_api_declaration(self):
        if False:
            print('Hello World!')
        return f"\n// {', '.join(self.outputs['names'])}\n{super().gene_api_declaration()}\n"

    def gene_output(self, out_dtype_list, out_tensor_type_list=None, code_indent='', inplace_flag=False):
        if False:
            while True:
                i = 10
        kernel_output = []
        output_names = []
        output_create = ''
        return_type = self.get_return_type_with_intermediate(inplace_flag)
        output_type_map = {'dense': 'TensorType::DENSE_TENSOR', 'sparse_coo': 'TensorType::SPARSE_COO', 'sparse_csr': 'TensorType::SPARSE_CSR'}
        if len(out_dtype_list) == 1:
            kernel_output.append('kernel_out')
            output_names.append('kernel_out')
            inplace_assign = ' = ' + self.inplace_map[self.outputs['names'][0]] if inplace_flag and self.inplace_map is not None and (self.outputs['names'][0] in self.inplace_map) else ''
            output_create = f'\n    {return_type} api_output{inplace_assign};\n    auto* kernel_out = SetSparseKernelOutput(&api_output, {output_type_map[out_dtype_list[0]]});'
        elif len(out_dtype_list) > 1:
            output_create = f'\n    {return_type} api_output;'
            if inplace_flag:
                output_create = f'\n    {return_type} api_output{{'
                for out_name in self.outputs['names']:
                    if out_name in self.inplace_map:
                        output_create = output_create + self.inplace_map[out_name] + ', '
                    else:
                        output_create += 'Tensor(), '
                output_create = output_create[:-2] + '};'
            for i in range(len(out_dtype_list)):
                kernel_output.append(f'kernel_out_{i}')
                output_names.append(f'kernel_out_{i}')
                output_create = output_create + f'\n    auto* kernel_out_{i} = SetSparseKernelOutput(&std::get<{i}>(api_output), {output_type_map[out_dtype_list[i]]});'
        else:
            raise ValueError(f'{self.api} : Output error: the output should not be empty.')
        return (kernel_output, output_names, output_create)

    def gen_sparse_kernel_context(self, kernel_output_names):
        if False:
            for i in range(10):
                print('nop')
        input_trans_map = {'const Tensor&': 'const phi::TenseBase&', 'const std::vector<Tensor>&': 'const std::vector<phi::TenseBase>&', 'const paddle::optional<Tensor>&': 'paddle::optional<const phi::TenseBase&>'}
        out_trans_map = {'Tensor': 'phi::TenseBase*', 'std::vector<Tensor>': 'std::vector<phi::TenseBase*>'}
        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']
        input_types = self.inputs['tensor_type']
        tensor_type_map = {'dense': 'phi::DenseTensor', 'sparse_coo': 'phi::SparseCooTensor', 'sparse_csr': 'phi::SparseCsrTensor'}
        inputsname2tensortype = {}
        for i in range(len(input_names)):
            inputsname2tensortype[input_names[i]] = input_types[i]
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names
        infer_meta = self.infer_meta
        infer_meta_params = infer_meta['param'] if infer_meta['param'] is not None else input_names + attr_names
        kernel_context_code = ''
        for param in kernel_param:
            if param in input_names and param not in infer_meta_params:
                var_name = '    auto ' + PREFIX_TENSOR_NAME + param + ' = '
                if self.inputs['input_info'][param] == 'const Tensor&':
                    if inputsname2tensortype[param] == 'sparse_coo':
                        kernel_context_code = kernel_context_code + var_name + 'PrepareDataForSparseCooTensor(' + param + ');\n'
                    elif inputsname2tensortype[param] == 'sparse_csr':
                        kernel_context_code = kernel_context_code + var_name + 'PrepareDataForSparseCsrTensor(' + param + ');\n'
                    else:
                        kernel_context_code = kernel_context_code + var_name + 'PrepareDataForDenseTensorInSparse(' + param + ');\n'
                elif param in self.optional_vars:
                    tensor_type = 'phi::DenseTensor'
                    for (name, input_type) in zip(input_names, input_types):
                        if param == name:
                            tensor_type = tensor_type_map[input_type]
                            break
                    optional_var = 'paddle::optional<' + tensor_type + '>('
                    if inputsname2tensortype[param] == 'sparse_coo':
                        kernel_context_code = kernel_context_code + var_name + 'PrepareDataForSparseCooTensor(' + param + ');\n'
                    elif inputsname2tensortype[param] == 'sparse_csr':
                        kernel_context_code = kernel_context_code + var_name + 'PrepareDataForSparseCsrTensor(' + param + ');\n'
                    else:
                        kernel_context_code = kernel_context_code + var_name + 'PrepareDataForDenseTensorInSparse(' + param + ');\n'
        for param in kernel_param:
            if param in input_names:
                if param in self.optional_vars:
                    kernel_context_code = kernel_context_code + f'\n    kernel_context.EmplaceBackInput({param} ? &(*{PREFIX_TENSOR_NAME}{param}) : nullptr);'
                else:
                    kernel_context_code = kernel_context_code + f'\n    kernel_context.EmplaceBackInput({PREFIX_TENSOR_NAME}{param}.get());'
                continue
            if param in attr_names:
                if 'IntArray' in self.attrs['attr_info'][param][0]:
                    param = 'phi::IntArray(' + param + ')'
                elif 'Scalar' in self.attrs['attr_info'][param][0]:
                    param = 'phi::Scalar(' + param + ')'
            elif isinstance(param, bool):
                param = str(param).lower()
            else:
                param + str(param) + ', '
            kernel_context_code = kernel_context_code + f'\n    kernel_context.EmplaceBackAttr({param});'
        for out_name in kernel_output_names:
            kernel_context_code = kernel_context_code + f'\n    kernel_context.EmplaceBackOutput({out_name});'
        return kernel_context_code

    def prepare_input(self):
        if False:
            return 10
        input_names = self.inputs['names']
        input_types = self.inputs['tensor_type']
        attr_names = self.attrs['names']
        infer_meta = self.infer_meta
        infer_meta_params = infer_meta['param'] if infer_meta['param'] is not None else input_names + attr_names
        inputsname2tensortype = {}
        for i in range(len(input_names)):
            inputsname2tensortype[input_names[i]] = input_types[i]
        create_input_var_code = ''
        tensor_type_map = {'dense': 'phi::DenseTensor', 'sparse_coo': 'phi::SparseCooTensor', 'sparse_csr': 'phi::SparseCsrTensor'}
        for param in infer_meta_params:
            if param in input_names:
                var_name = '    auto ' + PREFIX_TENSOR_NAME + param + ' = '
                if self.inputs['input_info'][param] == 'const Tensor&':
                    if inputsname2tensortype[param] == 'sparse_coo':
                        create_input_var_code = create_input_var_code + var_name + 'PrepareDataForSparseCooTensor(' + param + ');\n'
                    elif inputsname2tensortype[param] == 'sparse_csr':
                        create_input_var_code = create_input_var_code + var_name + 'PrepareDataForSparseCsrTensor(' + param + ');\n'
                    else:
                        create_input_var_code = create_input_var_code + var_name + 'PrepareDataForDenseTensorInSparse(' + param + ');\n'
                elif param in self.optional_vars:
                    tensor_type = 'phi::DenseTensor'
                    for (name, input_type) in zip(input_names, input_types):
                        if param == name:
                            tensor_type = tensor_type_map[input_type]
                            break
                    optional_var = 'paddle::optional<' + tensor_type + '>('
                    if inputsname2tensortype[param] == 'sparse_coo':
                        create_input_var_code = create_input_var_code + var_name + 'PrepareDataForSparseCooTensor(' + param + ');\n'
                    elif inputsname2tensortype[param] == 'sparse_csr':
                        create_input_var_code = create_input_var_code + var_name + 'PrepareDataForSparseCsrTensor(' + param + ');\n'
                    else:
                        create_input_var_code = create_input_var_code + var_name + 'PrepareDataForDenseTensorInSparse(' + param + ');\n'
        return f'{create_input_var_code}'

    def gen_sparse_kernel_code(self, kernel_name, inplace_flag=False):
        if False:
            for i in range(10):
                print('nop')
        (_, kernel_output_names, output_create) = self.gene_output(self.kernel['dispatch'][kernel_name][1], None, '', inplace_flag)
        kernel_context_code = self.gen_sparse_kernel_context(kernel_output_names)
        return_code = '' if len(self.gene_return_code()) == 0 else '  ' + self.gene_return_code()
        return f'''\n    VLOG(6) << "{self.api} api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";\n    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(\n        "{kernel_name}", {{kernel_backend, kernel_layout, kernel_data_type}});\n    const auto& phi_kernel = kernel_result.kernel;\n    if (FLAGS_low_precision_op_list) {{\n      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("{self.api}", kernel_data_type);\n    }}\n    VLOG(6) << "{self.api} api sparse kernel: " << phi_kernel;\n\n    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);\n    auto kernel_context = phi::KernelContext(dev_ctx);\n{output_create}\n{self.prepare_input()}\n{self.gene_infer_meta(kernel_output_names, '')}\n{kernel_context_code}\n    phi_kernel(&kernel_context);\n  {return_code}'''

    def get_condition_code(self, kernel_name):
        if False:
            i = 10
            return i + 15
        assert self.kernel['dispatch'][kernel_name], f"{self.api} api: the tensor type of inputs and outputs for kernel isn't set, see also 'kernel:func' of 'conv3d' in sparse_ops.yaml."
        input_types = self.kernel['dispatch'][kernel_name][0]
        sparse_type_map = {'sparse_coo': 'DataLayout::SPARSE_COO', 'sparse_csr': 'DataLayout::SPARSE_CSR'}
        condition_list = []
        tensor_type_list = []
        for (i, in_type) in enumerate(input_types):
            if in_type == 'dense':
                if self.inputs['names'][i] in self.optional_vars:
                    condition_list.append(f"(!{self.inputs['names'][i]} || phi::DenseTensor::classof({self.inputs['names'][i]}->impl().get()))")
                else:
                    condition_list.append(f"phi::DenseTensor::classof({self.inputs['names'][i]}.impl().get())")
            elif in_type == 'sparse_coo':
                condition_list.append(f"{self.inputs['names'][i]}.is_sparse_coo_tensor()")
            else:
                condition_list.append(f"{self.inputs['names'][i]}.is_sparse_csr_tensor()")
            tensor_type_list.append(in_type)
        self.inputs['tensor_type'] = tensor_type_list
        return ' && '.join(condition_list)

    def gene_dispatch_code(self, kernel_name, inplace_flag=False):
        if False:
            i = 10
            return i + 15
        return f'\n  if ({self.get_condition_code(kernel_name)}) {{\n{self.gen_sparse_kernel_code(kernel_name, inplace_flag)}\n  }}\n'

    def gene_base_api_code(self, inplace_flag=False):
        if False:
            print('Hello World!')
        api_func_name = self.get_api_func_name()
        if inplace_flag and api_func_name[-1] != '_':
            api_func_name += '_'
        kernel_dispatch_code = f'{self.gene_kernel_select()}\n'
        for kernel_name in self.kernel['func']:
            kernel_dispatch_code += self.gene_dispatch_code(kernel_name, inplace_flag)
        return f'\nPADDLE_API {self.get_return_type(inplace_flag)} {api_func_name}({self.get_define_args(inplace_flag)}) {{\n{kernel_dispatch_code}\n  PADDLE_THROW(phi::errors::Unimplemented(\n          "The kernel of ({self.api}) for input tensors is unimplemented, please check the type of input tensors."));\n}}\n'

def header_include():
    if False:
        print('Hello World!')
    return '\n#include <tuple>\n\n#include "paddle/phi/api/include/tensor.h"\n#include "paddle/phi/common/scalar.h"\n#include "paddle/phi/common/int_array.h"\n#include "paddle/utils/optional.h"\n'

def source_include(header_file_path):
    if False:
        return 10
    return f'\n#include "{header_file_path}"\n#include <memory>\n\n#include "glog/logging.h"\n#include "paddle/utils/flags.h"\n\n#include "paddle/phi/api/lib/api_gen_utils.h"\n#include "paddle/phi/api/lib/data_transform.h"\n#include "paddle/phi/api/lib/kernel_dispatch.h"\n#include "paddle/phi/core/kernel_registry.h"\n#include "paddle/phi/infermeta/unary.h"\n#include "paddle/phi/infermeta/binary.h"\n#include "paddle/phi/infermeta/ternary.h"\n#include "paddle/phi/infermeta/multiary.h"\n#include "paddle/utils/none.h"\n\n#include "paddle/phi/infermeta/sparse/unary.h"\n#include "paddle/phi/infermeta/sparse/binary.h"\n#include "paddle/phi/infermeta/sparse/multiary.h"\n\nPD_DECLARE_int32(low_precision_op_list);\n'

def api_namespace():
    if False:
        return 10
    return ('\nnamespace paddle {\nnamespace experimental {\nnamespace sparse {\n\n', '\n\n}  // namespace sparse\n}  // namespace experimental\n}  // namespace paddle\n')

def generate_api(api_yaml_path, header_file_path, source_file_path):
    if False:
        while True:
            i = 10
    with open(api_yaml_path, 'r') as f:
        apis = yaml.load(f, Loader=yaml.FullLoader)
    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')
    namespace = api_namespace()
    header_file.write('#pragma once\n')
    header_file.write(header_include())
    header_file.write(namespace[0])
    include_header_file = 'paddle/phi/api/include/sparse_api.h'
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])
    for api in apis:
        sparse_api = SparseAPI(api)
        if sparse_api.is_dygraph_api:
            sparse_api.is_dygraph_api = False
        header_file.write(sparse_api.gene_api_declaration())
        source_file.write(sparse_api.gene_api_code())
    header_file.write(namespace[1])
    source_file.write(namespace[1])
    header_file.close()
    source_file.close()

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Generate PaddlePaddle C++ Sparse API files')
    parser.add_argument('--api_yaml_path', help='path to sparse api yaml file', default='paddle/phi/api/yaml/sparse_ops.yaml')
    parser.add_argument('--api_header_path', help='output of generated api header code file', default='paddle/phi/api/include/sparse_api.h')
    parser.add_argument('--api_source_path', help='output of generated api source code file', default='paddle/phi/api/lib/sparse_api.cc')
    options = parser.parse_args()
    api_yaml_path = options.api_yaml_path
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path
    generate_api(api_yaml_path, header_file_path, source_file_path)
if __name__ == '__main__':
    main()