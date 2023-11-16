import argparse
import re
import yaml
from api_base import PREFIX_TENSOR_NAME
from api_gen import ForwardAPI, api_namespace, declare_extension_api, header_include, source_include
API_IMPL_TEMPLATE = '\nPADDLE_API {} {}({}) {{\n  // Kernel Key Construction{}\n  // Kernel Dispatch Body{}\n}}\n'
DIPATCH_END_GUARD_TEMPLATE = '\nPADDLE_THROW(phi::errors::Unimplemented(\n          "The kernel of ({}) for input tensors is unimplemented, please check the type of input tensors."));\n'
MAIN_DIST_BRANCH_TEMPLATE = "\n  // Auto Parallel condition\n  if (run_auto_parallel) {{\n    // 1. InferSpmd (Infer DistAttr of Inputs&Outputs){}\n    // 2. Create API Output & Prepare Dist and Dense Output{}\n    // 3. Infer DistTensor's Global Shape{}\n\n    if (rank_is_in_current_mesh) {{\n      // 4. Select Kernel{}\n      // 5. Reshard Input{}\n\n      // 6. PrepareData (DataTransform & Prepare Dense Input){}\n      // 7. Infer Local DenseTensor Meta{}\n      // 8. DenseTensor Kernel Call{}\n    }}\n\n    // 9. Set Output Dist Attr For Default Impl{}\n\n    // 10. Return\n    {}\n  }}\n"
GET_MESH_TEMPLATE = '\n    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>({}impl())->dist_attr().process_mesh();\n    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);'
AUTO_PARALLEL_COND_TEMPLATE = '\n  bool run_auto_parallel = AllInputsAreDistTensor({input_args});\n  bool rank_is_in_current_mesh = true;\n  if (run_auto_parallel) {{{mesh}\n  }}\n  if (rank_is_in_current_mesh) {{{kernel_code}\n  }}\n'
SINGLE_DIST_META_IN_TEMPLATE = '\n    auto meta_dist_input_{name} = MakeDistMetaTensor(*{name}.impl());'
VECTOR_DIST_META_IN_TEMPLATE = '\n    std::vector<phi::distributed::DistMetaTensor> meta_dist_input_{name};\n    for(auto& e : {name}) {{\n        meta_dist_input_{name}.push_back(MakeDistMetaTensor(*e.impl()));\n    }}'
OPTIONAL_SINGLE_DIST_META_IN_TEMPLATE = '\n    auto meta_dist_input_{name} = {name} ? MakeDistMetaTensor(*(*{name}).impl()) : phi::distributed::DistMetaTensor();'
OPTIONAL_VECTOR_DIST_META_IN_TEMPLATE = '\n    std::vector<phi::distributed::DistMetaTensor> meta_dist_input_{name};\n    if ({name}) {{\n        for(auto& e : *{name}) {{\n            meta_dist_input_{name}.push_back(MakeDistMetaTensor(*e.impl()));\n        }}\n    }}'
INFER_SPMD_TEMPLATE = '\n    auto spmd_info = phi::distributed::{}({});\n'
GENERAL_INFER_SPMD_TEMPLATE = '\n    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic({});\n'
UNSUPPORTED_INFER_SPMD_COMMENT_TEMPLATE = '\n    // API `{}` does not support InferSpmd now\n'
API_OUT_CREATION_TEMPLATE = '\n    {} api_output{};\n'
INPLACE_API_OUT_CREATION_TEMPLATE = '\n    {} api_output{{{}}};\n'
SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD = '\n    auto dist_out = SetKernelDistOutput(&api_output);\n    auto dense_out = dist_out->unsafe_mutable_value();\n    if (!rank_is_in_current_mesh) {{\n      *dense_out = phi::DenseTensor(\n            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),\n            phi::DenseTensorMeta());\n    }}\n'
MULTI_SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD = '\n    auto dist_out_{idx} = SetKernelDistOutput({out});\n    auto dense_out_{idx} = dist_out_{idx} ? dist_out_{idx}->unsafe_mutable_value() : nullptr;\n    if (!rank_is_in_current_mesh) {{\n      *dense_out_{idx} = phi::DenseTensor(\n            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),\n            phi::DenseTensorMeta());\n    }}\n'
SINGLE_OUT_CREATION_TEMPLATE = '\n    auto dist_out = SetKernelDistOutput(&api_output, spmd_info.second[0]);\n    auto dense_out = dist_out->unsafe_mutable_value();\n    if (!rank_is_in_current_mesh) {{\n      *dense_out = phi::DenseTensor(\n            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),\n            phi::DenseTensorMeta());\n    }}\n'
MULTI_SINGLE_OUT_CREATION_TEMPLATE = '\n    auto dist_out_{idx} = SetKernelDistOutput({out}, spmd_info.second[{idx}]);\n    auto dense_out_{idx} = dist_out_{idx}->unsafe_mutable_value();\n    if (!rank_is_in_current_mesh) {{\n      *dense_out_{idx} = phi::DenseTensor(\n            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),\n            phi::DenseTensorMeta());\n    }}\n'
VECTOR_OUT_CREATION_TEMPLATE = '\n    auto dist_out = SetKernelDistOutput({}, &api_output);\n    std::vector<phi::DenseTensor*> dense_out(dist_out.size());\n    for (size_t i = 0; i < dist_out.size(); ++i) {{\n      dense_out[i] = const_cast<phi::DenseTensor*>(&dist_out[i]->value());\n      if (!rank_is_in_current_mesh) {{\n        *dense_out[i] = phi::DenseTensor(\n                std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),\n                phi::DenseTensorMeta());\n      }}\n    }}\n'
MULTI_VECTOR_OUT_CREATION_TEMPLATE = '\n    auto dist_out_{out_name} = SetKernelDistOutput({dist_output_arg}, {in_name});\n    std::vector<phi::DenseTensor*> dense_out_{out_name}(dist_out_{out_name}.size());\n    for (size_t i = 0; i < dist_out_{out_name}.size(); ++i) {{\n        dense_out_{out_name}[i] = const_cast<phi::DenseTensor*>(&dist_out_{out_name}[i]->value());\n        if (!rank_is_in_current_mesh) {{\n          *dense_out_{out_name}[i] = phi::DenseTensor(\n                  std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),\n                  phi::DenseTensorMeta());\n        }}\n    }}\n'
MULTI_VECTOR_INPLACE_AND_OPTIONAL_OUT_CREATION_TEMPLATE = '\n    auto dist_out_{out_name} = {out_func}({size}, {in_name});\n    std::vector<phi::DenseTensor*> dense_out_{out_name}(dist_out_{out_name}.size());\n    for (size_t i = 0; i < dist_out_{out_name}.size(); ++i) {{\n        dense_out_{out_name}[i] = dist_out_{out_name}[i] ? const_cast<phi::DenseTensor*>(&dist_out_{out_name}[i]->value()) : nullptr;\n    }}\n'
SINGLE_GLOBAL_META_IN_TEMPLATE = 'MakeMetaTensor(*{}.impl()), '
VECTOR_GLOBAL_META_IN_TEMPLATE = '{}_meta_ptr_vec, '
VECTOR_GLOBAL_META_IN_DECL_TEMPLATE = '\n    std::vector<phi::MetaTensor> {name}_meta_vec;\n    for (auto tmp : {name}) {{\n      {name}_meta_vec.emplace_back(MakeMetaTensor(*tmp.impl()));\n    }}\n    std::vector<const phi::MetaTensor*> {name}_meta_ptr_vec({name}_meta_vec.size());\n    for (size_t i=0; i < {name}_meta_ptr_vec.size(); ++i) {{\n      {name}_meta_ptr_vec[i] = &{name}_meta_vec[i];\n    }}\n'
OPTIONAL_GLOBAL_SINGLE_META_IN_TEMPLATE = 'meta_dist_{}, '
OPTIONAL_GLOBAL_SINGLE_META_IN_DECL_TEMPLATE = '\n    phi::MetaTensor meta_dist_{name} = {name} ? MakeMetaTensor(*(*{name}).impl()) : phi::MetaTensor();\n'
OPTIONAL_GLOBAL_VECTOR_META_IN_TEMPLATE = '{}_meta_ptr_vec, '
OPTIONAL_GLOBAL_VECTOR_META_IN_DECL_TEMPLATE = '\n    std::vector<phi::MetaTensor> {name}_meta_vec_tmp;\n    if ({name}) {{\n      for (auto tmp : *{name}) {{\n        {name}_meta_vec_tmp.emplace_back(MakeMetaTensor(*tmp.impl()));\n      }}\n    }}\n    std::vector<const phi::MetaTensor*> {name}_meta_ptr_vec_tmp({name}_meta_vec_tmp.size());\n    for (size_t i = 0; i < {name}_meta_ptr_vec_tmp.size(); ++i) {{\n      {name}_meta_ptr_vec_tmp[i] = &{name}_meta_vec_tmp[i];\n    }}\n    paddle::optional<std::vector<const phi::MetaTensor*>> {name}_meta_ptr_vec =\n        {name} ? paddle::make_optional<std::vector<const phi::MetaTensor*>>({name}_meta_ptr_vec_tmp) : paddle::none;\n'
SINGLE_GLOBAL_META_OUT_DECL_TEMPLATE = '\n    phi::MetaTensor meta_{}({});'
VECTOR_GLOBAL_META_OUT_DECL_TEMPLATE = '\n    std::vector<phi::MetaTensor> {name}_meta_vec;\n    for (auto tmp : {name}) {{\n      {name}_meta_vec.emplace_back(phi::MetaTensor(tmp));\n    }}\n    std::vector<phi::MetaTensor*> {name}_meta_ptr_vec({name}.size());\n    for (size_t i = 0; i < {name}_meta_vec.size(); ++i) {{\n      {name}_meta_ptr_vec[i] = &{name}_meta_vec[i];\n    }}\n'
INFER_GLOBAL_SHAPE_TEMPLATE = '\n    phi::{}({}{});\n'
KERNEL_SELECTION_TEMPLATE = '\n      VLOG(6) << "{} API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";\n      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(\n          "{}", {{kernel_backend, kernel_layout, kernel_data_type}});\n      const auto& kernel = kernel_result.kernel;\n      VLOG(6) << "{} kernel: " << kernel;\n      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);\n'
INPUT_RESHARD_TEMPLATE = '\n      auto dist_input_{name} = ReshardApiInputToKernelInput(dev_ctx, {name}, spmd_info.first[{idx}]);'
GENERAL_INPUT_RESHARD_TEMPLATE = '\n      auto dist_input_{name} = ReshardApiInputToReplicatedKernelInput(dev_ctx, {name}, spmd_info.first[{idx}]);'
UNSUPPORTED_RESHARD_INPUT_COMMENT_TEMPLATE = '\n      // API `{}` does not need to support ReshardInput at this time\n'
SINGLE_PREPARE_DATA_TEMPLATE = '\n      dist_input_{name} = PrepareDataForDistTensor(dist_input_{name}, GetKernelInputArgDef(kernel.InputAt({idx}), kernel_backend), {trans_flag}, kernel_result.is_stride_kernel);\n      auto input_{name} = &dist_input_{name}->value();\n'
SINGLE_PREPARE_DATA_TEMPLATE_NO_RESHARD = '\n      auto dist_input_{name} = PrepareDataForDistTensor({name}, GetKernelInputArgDef(kernel.InputAt({idx}), kernel_backend), {trans_flag}, kernel_result.is_stride_kernel);\n      auto input_{name} = &dist_input_{name}->value();\n'
VECTOR_PREPARE_DATA_TEMPLATE = '\n      auto dist_input_{name}_vec = PrepareDataForDistTensor(dist_input_{name}, GetKernelInputArgDef(kernel.InputAt({idx}), kernel_backend), {trans_flag}, kernel_result.is_stride_kernel);\n      std::vector<const phi::DenseTensor*> dense_input_{name}_vec;\n      for (auto tmp : dist_input_{name}_vec) {{\n        dense_input_{name}_vec.emplace_back(&tmp->value());\n      }}\n      std::vector<phi::MetaTensor> dense_input_{name}_meta_vec = MakeMetaTensor(dense_input_{name}_vec);\n      std::vector<const phi::MetaTensor*> dense_input_{name}_meta_ptr_vec(dense_input_{name}_meta_vec.size());\n      for (size_t i = 0; i < dense_input_{name}_meta_ptr_vec.size(); ++i) {{\n        dense_input_{name}_meta_ptr_vec[i] = &dense_input_{name}_meta_vec[i];\n      }}\n'
OPTIONAL_SINGLE_PREPARE_DATA_TEMPLATE = '\n      dist_input_{name} = PrepareDataForDistTensor(dist_input_{name}, GetKernelInputArgDef(kernel.InputAt({idx}), kernel_backend), {trans_flag}, kernel_result.is_stride_kernel);\n      paddle::optional<phi::DenseTensor> input_{name} = dist_input_{name} ? paddle::make_optional<phi::DenseTensor>((*dist_input_{name})->value()) : paddle::none;\n'
OPTIONAL_SINGLE_PREPARE_DATA_TEMPLATE_NO_RESHARD = '\n      auto dist_input_{name} = PrepareDataForDistTensor(dist_input_{name}, GetKernelInputArgDef(kernel.InputAt({idx}), kernel_backend), {trans_flag}, kernel_result.is_stride_kernel);\n      paddle::optional<phi::DenseTensor> input_{name} = dist_input_{name} ? paddle::make_optional<phi::DenseTensor>(dist_input_{name}->value()) : paddle::none;\n'
OPTIONAL_VECTOR_PREPARE_DATA_TEMPLATE = '\n      auto dist_input_{name}_vec = PrepareDataForDistTensor(dist_input_{name}, GetKernelInputArgDef(kernel.InputAt({idx}), kernel_backend), {trans_flag}, kernel_result.is_stride_kernel);\n      std::vector<const phi::DenseTensor*> dense_input_{name}_vec;\n      if ({name}) {{\n        for (auto tmp : *dist_input_{name}_vec) {{\n          dense_input_{name}_vec.emplace_back(&tmp->value());\n      }}\n    }}\n    paddle::optional<std::vector<const phi::DenseTensor*>> input_{name}(dense_input_{name}_vec);\n    std::vector<phi::MetaTensor> dense_input_{name}_meta_vec = MakeMetaTensor(dense_input_{name}_vec);\n    std::vector<const phi::MetaTensor*> dense_input_{name}_meta_ptr_vec_tmp(dense_input_{name}_meta_vec.size());\n    for (size_t i = 0; i < dense_input_{name}_meta_ptr_vec_tmp.size(); ++i) {{\n      dense_input_{name}_meta_ptr_vec_tmp[i] = &dense_input_{name}_meta_vec[i];\n    }}\n    paddle::optional<std::vector<const phi::MetaTensor*>> dense_input_{name}_meta_ptr_vec =\n            {name} ? paddle::make_optional<std::vector<const phi::MetaTensor*>>(dense_input_{name}_meta_ptr_vec_tmp) : paddle::none;\n'
INFER_META_SINGLE_INPUT_TEMPLATE = '\n    auto dist_input_{} = {}.impl();\n    auto input_{} = &(static_cast<phi::distributed::DistTensor*>(dist_input_{}.get())->value());\n'
INFER_META_OPTIONAL_INPUT_TEMPLATE = '\n    paddle::optional<phi::TensorBase> input_{} = {} ? paddle::optional<phi::TensorBase>(*{}->impl()) : paddle::none;\n'
INFER_META_VECTOR_INPUT_TEMPLATE = '\n    auto input_{}_uq_ptr = TensorToDenseTensor({});\n    const auto& input_{} = *input_{}_uq_ptr;\n'
SINGLE_META_IN_TEMPLATE = 'MakeMetaTensor(*input_{}), '
VECTOR_META_IN_TEMPLATE = 'dense_input_{}_meta_ptr_vec, '
OPTIONAL_SINGLE_META_IN_TEMPLATE = 'MakeMetaTensor(input_{}), '
OPTIONAL_VECTOR_META_IN_TEMPLATE = 'dense_input_{}_meta_ptr_vec, '
SINGLE_META_OUT_DECL_TEMPLATE = '\n      phi::MetaTensor meta_{}({});'
VECTOR_META_OUT_DECL_TEMPLATE = '\n      std::vector<phi::MetaTensor> {name}_meta_vec = MakeMetaTensor({name});\n      std::vector<phi::MetaTensor*> {name}_meta_ptr_vec({name}_meta_vec.size());\n      for (size_t i = 0; i < {name}_meta_vec.size(); ++i) {{\n        {name}_meta_ptr_vec[i] = &{name}_meta_vec[i];\n      }}\n'
INFER_META_TEMPLATE = '\n      phi::{}({}{});\n'
SINGLE_OUTPUT_NAME = 'dense_out'
VECTOR_OUTPUT_NAME_TEMPLATE = '\n'
TUPLE_OUTPUT_NAME_TEMPLATE = '\n'
KERNEL_CALL_TEMPLATE = '\n      using kernel_signature = {};\n      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();\n      (*kernel_fn)({}, {});\n'
SINGLE_SET_DIST_OUT_DIMS = '\n    dist_out->unsafe_set_dims(dense_out->dims());\n'
MULTI_SINGLE_SET_DIST_OUT_DIMS = '\n    dist_out_{}->unsafe_set_dims(dense_out_{}->dims());\n'
VECTOR_SET_DIST_OUT_DIMS = '\n    for (size_t i = 0; i < dist_out.size(); ++i) {{\n        dist_out[i]->unsafe_set_dims(dense_out[i]->dims());\n    }}\n'
PREFIX_VECTOR_TENSOR_NAME = 'dense_input_'
SUFFIX_VECTOR_TENSOR_NAME = '_vec'
CURRENT_PROCESS_MESH_TEMPLATE = '\n    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?\n               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();'
SET_SINGLE_OUT_REPLICATED_DIST_ATTR_TEMPLATE = '\n    SetReplicatedDistAttrForOutput({}, current_process_mesh);'
SET_VECTOR_OUT_REPLICATED_DIST_ATTR_TEMPLATE = '\n    for (size_t i = 0; i < {name}.size(); ++i) {{\n        SetReplicatedDistAttrForOutput({name}[i], current_process_mesh);\n    }}\n'
NONEED_TO_SET_DIST_ATTR_COMMENT_TEMPLATE = '\n    // API `{}` does not need to set DistAttr for output.'
ops_infer_shape_in_runtime = ['bincount', 'bicubic_interp', 'bilinear_interp', 'linear_interp', 'nearest_interp', 'trilinear_interp']

class DistForwardAPI(ForwardAPI):

    def __init__(self, api_item_yaml):
        if False:
            return 10
        super().__init__(api_item_yaml)
        self.init_dist_api_members()

    def init_dist_api_members(self):
        if False:
            i = 10
            return i + 15
        self.gene_dist_input_func = {'const Tensor&': {'dense': self.generate_single_dense_input}, 'const std::vector<Tensor>&': {'dense': self.generate_vector_dense_input}, 'const paddle::optional<Tensor>&': {'dense': self.generate_optional_single_dense_input}, 'const paddle::optional<std::vector<Tensor>>&': {'dense': self.generate_optional_vector_dense_input}}
        self.inplace_flag = False
        self.dist_output_args = []
        self.dense_output_args = []
        self.generate_infer_spmd = False
        self.generate_general_infer_spmd = False

    def parse_infer_meta(self, infer_meta_config):
        if False:
            for i in range(10):
                print('nop')
        infer_meta = infer_meta_config
        if 'param' not in infer_meta_config:
            infer_meta['param'] = None
        if 'spmd_rule' not in infer_meta_config:
            infer_meta['spmd_rule'] = None
        return infer_meta

    def need_to_generate_code_for_inplace_impl(self, i):
        if False:
            while True:
                i = 10
        return self.inplace_flag and self.inplace_map is not None and (self.outputs['names'][i] in self.inplace_map)

    def need_to_generate_code_for_view_impl(self, i):
        if False:
            for i in range(10):
                print('nop')
        return not self.inplace_flag and self.view_map is not None and (self.outputs['names'][i] in self.view_map)

    def is_inplace_output(self, i):
        if False:
            while True:
                i = 10
        return self.outputs['names'][i] in self.inplace_map

    def is_inplace_and_optional_output(self, i):
        if False:
            i = 10
            return i + 15
        return self.outputs['names'][i] in self.inplace_map and self.inplace_map[self.outputs['names'][i]] in self.optional_vars

    def vector_output_size_assertion_check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.outputs['out_size_expr'] is not None, f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."

    def generate_non_computation_rank_clip_code(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if len(self.inputs['names']) > 0:
            mesh = ''
            if self.inputs['input_info'][self.inputs['names'][0]] == 'const Tensor&':
                mesh = GET_MESH_TEMPLATE.format('{}.'.format(self.inputs['names'][0]))
            elif self.inputs['input_info'][self.inputs['names'][0]] == 'const paddle::optional<Tensor>&':
                mesh = GET_MESH_TEMPLATE.format('{}->'.format(self.inputs['names'][0]))
            elif self.inputs['input_info'][self.inputs['names'][0]] == 'const std::vector<Tensor>&':
                mesh = GET_MESH_TEMPLATE.format('{}[0].'.format(self.inputs['names'][0]))
            elif self.inputs['input_info'][self.inputs['names'][0]] == 'const paddle::optional<std::vector<Tensor>>&':
                mesh = GET_MESH_TEMPLATE.format('{}->at(0).'.format(self.inputs['names'][0]))
            return mesh
        else:
            return ''

    def gene_kernel_backend_select(self):
        if False:
            while True:
                i = 10
        backend_select_code = ''
        if self.kernel['backend'] is not None:
            if '>' in self.kernel['backend']:
                vars_list = self.kernel['backend'].split('>')
                assert len(vars_list) == 2, f"{self.api} api: The number of params to set backend with '>' only allows 2, but received {len(vars_list)}."
                assert vars_list[0].strip() in self.attrs['names'] and self.attrs['attr_info'][vars_list[0].strip()][0] == 'const Place&', f"{self.api} api: When use '>' to set kernel backend, the first param should be a attribute with Place type."
                backend_select_code = f'\n    kernel_backend = ParseBackendWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});\n'
            else:
                backend_args = [ele.strip() for ele in self.kernel['backend'].split(',')]
                backend_select_code = f"\n    kernel_backend = ParseBackend({', '.join(backend_args)});\n"
        return backend_select_code

    def gene_kernel_select(self) -> str:
        if False:
            print('Hello World!')
        api = self.api
        input_names = self.inputs['names']
        attrs = self.attrs
        kernel = self.kernel
        kernel_key_item_init = '\n  Backend kernel_backend = Backend::UNDEFINED;\n  DataLayout kernel_layout = DataLayout::UNDEFINED;\n  DataType kernel_data_type = DataType::UNDEFINED;\n'
        attr_backend_count = 0
        attr_layout_count = 0
        attr_data_type_count = 0
        for attr_name in attrs['names']:
            if attrs['attr_info'][attr_name][0] == 'const Place&':
                assert kernel['backend'] is not None, f"{api} api: When there is a parameter with 'Place' type in attributes, you must set backend of kernel manually."
                attr_backend_count = attr_backend_count + 1
            if attrs['attr_info'][attr_name][0] == 'DataLayout':
                assert kernel['layout'] is not None, f"{api} api: When there is a parameter with 'DataLayout' type in attributes, you must set layout of kernel manually."
                attr_layout_count = attr_layout_count + 1
            if attrs['attr_info'][attr_name][0] == 'DataType':
                assert kernel['data_type'] is not None, f"{api} api: When there is a parameter with 'DataType' type in attributes, you must set data_type of kernel manually."
                attr_data_type_count = attr_data_type_count + 1
        kernel_select_code = self.gene_kernel_backend_select()
        if kernel['layout'] is not None:
            if '>' in kernel['layout']:
                vars_list = kernel['layout'].split('>')
                assert len(vars_list) == 2, f"{api} api: The number of params to set layout with '>' only allows 2, but received {len(vars_list)}."
                assert vars_list[0].strip() in attrs['names'] and attrs['attr_info'][vars_list[0].strip()][0] == 'DataLayout', f"{api} api: When use '>' to set kernel layout, the first param should be a attribute with DataLayout type."
                kernel_select_code = kernel_select_code + f'\n    kernel_layout = ParseLayoutWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});\n'
            else:
                vars_list = kernel['layout'].split(',')
                assert len(vars_list) == 1, f'{api} api: The number of params to set layout must be 1, but received {len(vars_list)}.'
                kernel_select_code = kernel_select_code + f'\n    kernel_layout = ParseLayout({vars_list[0].strip()});\n'
        if kernel['data_type'] is not None:

            def process_data_type_args(args_item):
                if False:
                    print('Hello World!')
                args_item = args_item.strip()
                complex_match_result = re.match('complex\\((?P<param_name>\\w+)\\)', args_item)
                if complex_match_result:
                    return f"phi::dtype::ToComplex(ParseDataType({complex_match_result.group('param_name')}))"
                else:
                    return f'ParseDataType({args_item})'
            if '>' in kernel['data_type']:
                vars_list = kernel['data_type'].split('>')
                assert len(vars_list) == 2, f"{api} api: The number of params to set data_type with '>' only allows 2, but received {len(vars_list)}."
                assert vars_list[0].strip() in attrs['names'] and attrs['attr_info'][vars_list[0].strip()][0] == 'DataType', f"{api} api: When use '>' to set kernel data_type, the first param should be a attribute with DataType type."
                kernel_select_code = kernel_select_code + f'\n    kernel_data_type = ParseDataTypeWithInputOrder({vars_list[0].strip()}, {vars_list[1].strip()});\n'
            else:
                vars_list = kernel['data_type'].split(',')
                assert len(vars_list) == 1, f'{api} api: The number of params to set data_type only allows 1, but received {len(vars_list)}.'
                kernel_select_code = kernel_select_code + f'\n    kernel_data_type = {process_data_type_args(vars_list[0])};\n'
        if len(input_names) == 0:
            assert attr_backend_count > 0 and attr_data_type_count > 0, f"{api} api: When there is no input tensor, the args must have 'Place' and 'DataType'."
        kernel_select_args = ''
        for input_name in input_names:
            kernel_select_args = kernel_select_args + input_name + ', '
        if len(kernel_select_args) > 2:
            kernel_select_args = kernel_select_args[:-2]
        if len(input_names) > 0:
            kernel_select_code = kernel_select_code + f'\n    if (kernel_backend == Backend::UNDEFINED\n          || kernel_layout == DataLayout::UNDEFINED\n          || kernel_data_type == DataType::UNDEFINED ) {{\n      auto kernel_key_set = ParseKernelKeyByInputArgs({kernel_select_args});\n      auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();\n      if (kernel_backend == Backend::UNDEFINED) {{\n        kernel_backend = kernel_key.backend();\n      }}\n      if (kernel_layout == DataLayout::UNDEFINED) {{\n        kernel_layout = kernel_key.layout();\n      }}\n      if (kernel_data_type == DataType::UNDEFINED) {{\n        kernel_data_type = kernel_key.dtype();\n      }}\n    }}'
        input_args = ''
        for input_name in self.inputs['names']:
            input_args = input_args + input_name + ', '
        if len(input_args) > 2:
            input_args = input_args[:-2]
        mesh = self.generate_non_computation_rank_clip_code()
        if_condition_code = AUTO_PARALLEL_COND_TEMPLATE.format(input_args=input_args, mesh=mesh, kernel_code=kernel_select_code)
        return kernel_key_item_init + if_condition_code

    def generate_specialized_infer_spmd_code(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_params = self.kernel['param']
        if kernel_params is None:
            kernel_params = input_names + attr_names
        input_decl_code = ''
        input_args_code = ''
        for param in kernel_params:
            if param in input_names:
                if self.inputs['input_info'][param] == 'const Tensor&':
                    input_decl_code += SINGLE_DIST_META_IN_TEMPLATE.format(name=param)
                    input_args_code += 'meta_dist_input_' + param + ', '
                elif self.inputs['input_info'][param] == 'const std::vector<Tensor>&':
                    input_decl_code += VECTOR_DIST_META_IN_TEMPLATE.format(name=param)
                    input_args_code += 'meta_dist_input_' + param + ', '
                else:
                    raise ValueError(f"{self.api} : Param of infer_spmd error : {self.inputs['input_info'][param]} type is not supported.")
            elif param in attr_names:
                input_args_code = input_args_code + param + ', '
            elif isinstance(param, str):
                input_args_code = input_args_code + '"' + param + '", '
            elif isinstance(param, bool):
                input_args_code = input_args_code + str(param).lower() + ', '
            else:
                input_args_code = input_args_code + str(param) + ', '
        infer_spmd_code = ''
        infer_spmd_func_code = self.infer_meta['spmd_rule']
        infer_spmd_code = INFER_SPMD_TEMPLATE.format(infer_spmd_func_code, input_args_code[:-2])
        self.generate_infer_spmd = True
        return input_decl_code + infer_spmd_code

    def generate_general_infer_spmd_code(self) -> str:
        if False:
            print('Hello World!')
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_params = self.kernel['param']
        if kernel_params is None:
            kernel_params = input_names + attr_names
        input_decl_code = ''
        input_args_code = ''
        for param in kernel_params:
            if param in input_names:
                if self.inputs['input_info'][param] == 'const Tensor&':
                    input_decl_code += SINGLE_DIST_META_IN_TEMPLATE.format(name=param)
                    input_args_code += 'meta_dist_input_' + param + ', '
                elif self.inputs['input_info'][param] == 'const paddle::optional<Tensor>&':
                    input_decl_code += OPTIONAL_SINGLE_DIST_META_IN_TEMPLATE.format(name=param)
                    input_args_code += 'meta_dist_input_' + param + ', '
                elif self.inputs['input_info'][param] == 'const std::vector<Tensor>&':
                    input_decl_code += VECTOR_DIST_META_IN_TEMPLATE.format(name=param)
                    input_args_code += 'meta_dist_input_' + param + ', '
                elif self.inputs['input_info'][param] == 'const paddle::optional<std::vector<Tensor>>&':
                    input_decl_code += OPTIONAL_VECTOR_DIST_META_IN_TEMPLATE.format(name=param)
                    input_args_code += 'meta_dist_input_' + param + ', '
                else:
                    raise ValueError(f"{self.api} : Param of infer_spmd error : {self.inputs['input_info'][param]} type is not supported.")
            else:
                pass
        if input_decl_code == '':
            return UNSUPPORTED_INFER_SPMD_COMMENT_TEMPLATE.format(self.api)
        infer_spmd_code = GENERAL_INFER_SPMD_TEMPLATE.format(input_args_code[:-2])
        self.generate_infer_spmd = True
        self.generate_general_infer_spmd = True
        return input_decl_code + infer_spmd_code

    def generate_infer_spmd_code(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.infer_meta['spmd_rule'] is not None:
            return self.generate_specialized_infer_spmd_code()
        else:
            return self.generate_general_infer_spmd_code()

    def generate_output_creation_code(self) -> str:
        if False:
            while True:
                i = 10
        output_num = len(self.outputs['types'])
        return_type = self.get_return_type_with_intermediate(self.inplace_flag)
        output_creation_code = ''
        output_creation_code += '\n    phi::DeviceContext* dev_ctx = nullptr;'
        if output_num == 1:
            if self.need_to_generate_code_for_inplace_impl(0):
                inplace_assign_code = ' = ' + self.inplace_map[self.outputs['names'][0]]
                output_creation_code += API_OUT_CREATION_TEMPLATE.format(return_type, inplace_assign_code)
            else:
                output_creation_code += API_OUT_CREATION_TEMPLATE.format(return_type, '')
            self.dist_output_args.append('dist_out')
            self.dense_output_args.append('dense_out')
            if self.outputs['types'][0] == 'Tensor':
                if self.infer_meta['spmd_rule'] is not None:
                    output_creation_code += SINGLE_OUT_CREATION_TEMPLATE
                else:
                    output_creation_code += SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD
            elif self.outputs['types'][0] == 'std::vector<Tensor>':
                dist_output_arg = 'spmd_info.second[0]' if self.infer_meta['spmd_rule'] is not None else self.outputs['out_size_expr'][0]
                output_creation_code += VECTOR_OUT_CREATION_TEMPLATE.format(dist_output_arg)
            else:
                self.vector_output_size_assertion_check()
        elif output_num > 1:
            if self.inplace_flag:
                inplace_assign_code = ''
                for (i, out_name) in enumerate(self.outputs['names']):
                    if self.need_to_generate_code_for_inplace_impl(i):
                        inplace_assign_code += self.inplace_map[out_name] + ', '
                    else:
                        inplace_assign_code += 'Tensor(), '
                inplace_assign_code = inplace_assign_code[:-2]
                output_creation_code += INPLACE_API_OUT_CREATION_TEMPLATE.format(return_type, inplace_assign_code)
            else:
                output_creation_code += API_OUT_CREATION_TEMPLATE.format(return_type, '')
            for (i, out_type) in enumerate(self.outputs['types']):
                self.dist_output_args.append(f'dist_out_{i}')
                self.dense_output_args.append(f'dense_out_{i}')
                set_out_func = 'SetKernelDistOutput'
                get_out_code = f'&std::get<{i}>(api_output)'
                if self.is_inplace_and_optional_output(i):
                    get_out_code = f'std::get<{i}>(api_output).get_ptr()'
                if out_type == 'std::vector<Tensor>':
                    self.vector_output_size_assertion_check()
                    if self.is_inplace_output(i):
                        set_out_func = 'SetKernelDistInplaceOutput'
                        if self.is_inplace_and_optional_output(i):
                            set_out_func = 'SetKernelDistInplaceOptionalOutput'
                            get_out_code = f'std::get<{i}>(api_output)'
                        output_creation_code += MULTI_VECTOR_INPLACE_AND_OPTIONAL_OUT_CREATION_TEMPLATE.format(out_func=set_out_func, out_name=i, size=self.outputs['out_size_expr'][i], in_name=get_out_code)
                    else:
                        dist_output_arg = f'spmd_info.second[{i}]' if self.infer_meta['spmd_rule'] is not None else self.outputs['out_size_expr'][i]
                        output_creation_code += MULTI_VECTOR_OUT_CREATION_TEMPLATE.format(out_name=i, dist_output_arg=dist_output_arg, in_name=get_out_code)
                elif self.infer_meta['spmd_rule'] is not None:
                    output_creation_code += MULTI_SINGLE_OUT_CREATION_TEMPLATE.format(idx=i, out=get_out_code)
                else:
                    output_creation_code += MULTI_SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD.format(idx=i, out=get_out_code)
        else:
            raise ValueError(f'{self.api} : Output error: the output should not be empty.')
        return output_creation_code

    def generate_infer_global_shape_code(self) -> str:
        if False:
            return 10
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        infer_meta = self.infer_meta
        infer_meta_func_code = infer_meta['func']
        infer_meta_params = infer_meta['param'] if infer_meta['param'] is not None else input_names + attr_names
        input_meta_code = ''
        input_args_code = ''
        for param in infer_meta_params:
            if param in input_names:
                if self.inputs['input_info'][param] == 'const Tensor&':
                    input_args_code += SINGLE_GLOBAL_META_IN_TEMPLATE.format(param)
                elif self.inputs['input_info'][param] == 'const std::vector<Tensor>&':
                    input_args_code += VECTOR_GLOBAL_META_IN_TEMPLATE.format(param)
                    input_meta_code += VECTOR_GLOBAL_META_IN_DECL_TEMPLATE.format(name=param)
                elif self.inputs['input_info'][param] == 'const paddle::optional<Tensor>&':
                    input_args_code += OPTIONAL_GLOBAL_SINGLE_META_IN_TEMPLATE.format(param)
                    input_meta_code += OPTIONAL_GLOBAL_SINGLE_META_IN_DECL_TEMPLATE.format(name=param)
                elif self.inputs['input_info'][param] == 'const paddle::optional<std::vector<Tensor>>&':
                    input_args_code += OPTIONAL_GLOBAL_VECTOR_META_IN_TEMPLATE.format(param)
                    input_meta_code += OPTIONAL_GLOBAL_VECTOR_META_IN_DECL_TEMPLATE.format(name=param)
                else:
                    raise ValueError(f"{self.api} : Param of infer_spmd error : {self.inputs['input_info'][param]} type is not supported.")
            elif param in attr_names:
                input_args_code = input_args_code + param + ', '
            elif isinstance(param, str):
                input_args_code = input_args_code + '"' + param + '", '
            elif isinstance(param, bool):
                input_args_code = input_args_code + str(param).lower() + ', '
            else:
                input_args_code = input_args_code + str(param) + ', '
        output_decl_code = ''
        output_args_code = ''
        for (i, out_name) in enumerate(self.dist_output_args):
            if self.outputs['types'][i] == 'std::vector<Tensor>':
                output_decl_code += VECTOR_GLOBAL_META_OUT_DECL_TEMPLATE.format(name=out_name)
                output_args_code += f'{out_name}_meta_ptr_vec, '
            else:
                output_decl_code += SINGLE_GLOBAL_META_OUT_DECL_TEMPLATE.format(out_name, out_name)
                if len(self.dense_output_args) == 1:
                    output_args_code += f'&meta_{out_name}, '
                else:
                    output_args_code += f'{out_name} ? &meta_{out_name} : nullptr, '
        output_args_code = output_args_code[:-2]
        return output_decl_code + input_meta_code + INFER_GLOBAL_SHAPE_TEMPLATE.format(infer_meta_func_code, input_args_code, output_args_code)

    def generate_kernel_selection_code(self) -> str:
        if False:
            return 10
        return KERNEL_SELECTION_TEMPLATE.format(self.api, self.kernel['func'][0], self.kernel['func'][0])

    def generate_reshard_input_code(self) -> str:
        if False:
            print('Hello World!')
        input_reshard_code = ''
        if self.generate_infer_spmd is True:
            input_names = self.inputs['names']
            kernel_params = self.kernel['param'] if self.kernel['param'] is not None else input_names
            for (i, param) in enumerate(kernel_params):
                if param in input_names:
                    if self.inputs['input_info'][param] in ['const Tensor&', 'const std::vector<Tensor>&', 'const paddle::optional<Tensor>&', 'const paddle::optional<std::vector<Tensor>>&']:
                        input_reshard_code += INPUT_RESHARD_TEMPLATE.format(name=param, idx=i)
                    else:
                        raise ValueError(f"{self.api} : Param of reshard input error : {self.inputs['input_info'][param]} type is not supported.")
                else:
                    pass
        else:
            input_reshard_code = UNSUPPORTED_RESHARD_INPUT_COMMENT_TEMPLATE.format(self.api)
        return input_reshard_code

    def generate_single_dense_input(self, input_name):
        if False:
            while True:
                i = 10
        input_tensor_code = ''
        trans_flag = self.gene_trans_flag(input_name)
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names
        if self.generate_infer_spmd is True:
            input_tensor_code += SINGLE_PREPARE_DATA_TEMPLATE.format(name=input_name, idx=kernel_param.index(input_name), trans_flag=trans_flag)
        else:
            input_tensor_code += SINGLE_PREPARE_DATA_TEMPLATE_NO_RESHARD.format(arg=input_name, idx=kernel_param.index(input_name), trans_flag=trans_flag)
        return input_tensor_code

    def generate_vector_dense_input(self, input_name):
        if False:
            for i in range(10):
                print('nop')
        input_tensor_code = ''
        trans_flag = self.gene_trans_flag(input_name)
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names
        input_tensor_code += VECTOR_PREPARE_DATA_TEMPLATE.format(name=input_name, idx=kernel_param.index(input_name), trans_flag=trans_flag)
        return input_tensor_code

    def generate_optional_single_dense_input(self, input_name):
        if False:
            for i in range(10):
                print('nop')
        input_tensor_code = ''
        trans_flag = self.gene_trans_flag(input_name)
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names
        if self.generate_infer_spmd is True:
            input_tensor_code += OPTIONAL_SINGLE_PREPARE_DATA_TEMPLATE.format(name=input_name, idx=kernel_param.index(input_name), trans_flag=trans_flag)
        else:
            input_tensor_code += OPTIONAL_SINGLE_PREPARE_DATA_TEMPLATE_NO_RESHARD.format(name=input_name, idx=kernel_param.index(input_name), trans_flag=trans_flag)
        return input_tensor_code

    def generate_optional_vector_dense_input(self, input_name):
        if False:
            for i in range(10):
                print('nop')
        input_tensor_code = ''
        trans_flag = self.gene_trans_flag(input_name)
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names
        input_tensor_code += OPTIONAL_VECTOR_PREPARE_DATA_TEMPLATE.format(name=input_name, idx=kernel_param.index(input_name), trans_flag=trans_flag)
        return input_tensor_code

    def generate_prepare_data_code(self) -> str:
        if False:
            print('Hello World!')
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        kernel_param = self.kernel['param']
        if kernel_param is None:
            kernel_param = input_names + attr_names
        input_tensor_code = ''
        for (i, input_name) in enumerate(input_names):
            if input_name in kernel_param:
                api_tensor_type = self.inputs['input_info'][input_name]
                phi_tensor_type = 'dense'
                if api_tensor_type in self.gene_dist_input_func.keys():
                    input_tensor_code += self.gene_dist_input_func[api_tensor_type][phi_tensor_type](input_name)
                else:
                    pass
            elif input_name in self.infer_meta['param']:
                if input_name in self.optional_vars:
                    input_tensor_code += INFER_META_OPTIONAL_INPUT_TEMPLATE.format(input_name, input_name, input_name, input_name)
                elif self.inputs['input_info'][input_name] == 'const std::vector<Tensor>&':
                    input_tensor_code += INFER_META_VECTOR_INPUT_TEMPLATE.format(input_name, input_name, input_name)
                else:
                    input_tensor_code += INFER_META_SINGLE_INPUT_TEMPLATE.format(input_name, input_name, input_name, input_name)
        return input_tensor_code

    def generate_infer_meta_code(self) -> str:
        if False:
            return 10
        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        infer_meta = self.infer_meta
        infer_meta_func_code = infer_meta['func']
        infer_meta_params = infer_meta['param'] if infer_meta['param'] is not None else input_names + attr_names
        input_args_code = ''
        for param in infer_meta_params:
            if param in input_names:
                if self.inputs['input_info'][param] == 'const Tensor&':
                    input_args_code += SINGLE_META_IN_TEMPLATE.format(param)
                elif self.inputs['input_info'][param] == 'const std::vector<Tensor>&':
                    input_args_code += VECTOR_META_IN_TEMPLATE.format(param)
                elif self.inputs['input_info'][param] == 'const paddle::optional<Tensor>&':
                    input_args_code += OPTIONAL_SINGLE_META_IN_TEMPLATE.format(param)
                elif self.inputs['input_info'][param] == 'const paddle::optional<std::vector<Tensor>>&':
                    input_args_code += OPTIONAL_VECTOR_META_IN_TEMPLATE.format(param)
                else:
                    raise ValueError(f"{self.api} : Param of infer_meta error : {self.inputs['input_info'][param]} type is not supported.")
            elif param in attr_names:
                input_args_code = input_args_code + param + ', '
            elif isinstance(param, str):
                input_args_code = input_args_code + '"' + param + '", '
            elif isinstance(param, bool):
                input_args_code = input_args_code + str(param).lower() + ', '
            else:
                input_args_code = input_args_code + str(param) + ', '
        output_decl_code = ''
        output_args_code = ''
        for (i, out_name) in enumerate(self.dense_output_args):
            if self.outputs['types'][i] == 'std::vector<Tensor>':
                output_decl_code += VECTOR_META_OUT_DECL_TEMPLATE.format(name=out_name)
                output_args_code += f'{out_name}_meta_ptr_vec, '
            else:
                output_decl_code += SINGLE_META_OUT_DECL_TEMPLATE.format(out_name, out_name)
                if len(self.dense_output_args) == 1:
                    output_args_code += f'&meta_{out_name}, '
                else:
                    output_args_code += f'{out_name} ? &meta_{out_name} : nullptr, '
        output_args_code = output_args_code[:-2]
        return output_decl_code + INFER_META_TEMPLATE.format(infer_meta_func_code, input_args_code, output_args_code)

    def generate_kernel_call_code(self) -> str:
        if False:
            i = 10
            return i + 15
        dense_input_trans_map = {'const Tensor&': 'const phi::DenseTensor&', 'const std::vector<Tensor>&': 'const std::vector<const phi::DenseTensor*>&', 'const paddle::optional<Tensor&>': 'paddle::optional<const phi::DenseTensor&>', 'const paddle::optional<Tensor>&': 'const paddle::optional<phi::DenseTensor>&', 'const paddle::optional<std::vector<Tensor>>&': 'const paddle::optional<std::vector<const phi::DenseTensor*>>&'}
        dense_output_trans_map = {'Tensor': 'phi::DenseTensor*', 'std::vector<Tensor>': 'std::vector<phi::DenseTensor*>'}
        input_names = self.inputs['names']
        input_infos = self.inputs['input_info']
        kernel_args_type_list = ['const phi::DeviceContext&']
        attr_names = self.attrs['names']
        kernel_args = self.kernel['param']
        if kernel_args is None:
            kernel_args = input_names + attr_names
        input_args = ['*dev_ctx']
        for arg in kernel_args:
            if arg in input_names:
                if arg in self.optional_vars:
                    input_args.append(PREFIX_TENSOR_NAME + arg)
                elif input_infos[arg] == 'const Tensor&':
                    input_args.append('*' + PREFIX_TENSOR_NAME + arg)
                elif input_infos[arg] == 'const std::vector<Tensor>&':
                    input_args.append(PREFIX_VECTOR_TENSOR_NAME + arg + SUFFIX_VECTOR_TENSOR_NAME)
                else:
                    pass
                kernel_args_type_list.append(dense_input_trans_map[input_infos[arg]])
            elif arg in attr_names:
                if 'IntArray' in self.attrs['attr_info'][arg][0]:
                    kernel_args_type_list.append('const phi::IntArray&')
                    arg = 'phi::IntArray(' + arg + ')'
                elif 'vector<phi::Scalar>' in self.attrs['attr_info'][arg][0]:
                    kernel_args_type_list.append('const std::vector<phi::Scalar>&')
                elif 'Scalar' in self.attrs['attr_info'][arg][0]:
                    kernel_args_type_list.append('const phi::Scalar&')
                    arg = 'phi::Scalar(' + arg + ')'
                else:
                    kernel_args_type_list.append(self.attrs['attr_info'][arg][0])
                input_args.append(arg)
            elif isinstance(arg, bool):
                input_args.append(str(arg).lower())
            else:
                input_args.append(str(arg))
        for (i, out_type) in enumerate(self.outputs['types']):
            kernel_args_type_list.append(dense_output_trans_map[out_type])
        kernel_signature = 'void(*)(' + ', '.join(kernel_args_type_list) + ')'
        result = KERNEL_CALL_TEMPLATE.format(kernel_signature, ', '.join(input_args), ', '.join(self.dense_output_args))
        global ops_infer_shape_in_runtime
        if self.kernel['func'][0] in ops_infer_shape_in_runtime:
            if len(self.outputs['types']) == 1:
                if self.outputs['types'][0] == 'Tensor':
                    result += SINGLE_SET_DIST_OUT_DIMS
                elif self.outputs['types'][0] == 'std::vector<Tensor>':
                    result += VECTOR_SET_DIST_OUT_DIMS
            else:
                for i in range(len(self.outputs['types'])):
                    result += MULTI_SINGLE_SET_DIST_OUT_DIMS.format(i, i)
        return result

    def generate_output_dist_attr_setting(self) -> str:
        if False:
            while True:
                i = 10
        set_out_dist_attr_code = ''
        if self.generate_general_infer_spmd is True:
            set_out_dist_attr_code += CURRENT_PROCESS_MESH_TEMPLATE
            for (i, out_name) in enumerate(self.dist_output_args):
                if self.outputs['types'][i] == 'std::vector<Tensor>':
                    set_out_dist_attr_code += SET_VECTOR_OUT_REPLICATED_DIST_ATTR_TEMPLATE.format(name=out_name)
                else:
                    set_out_dist_attr_code += SET_SINGLE_OUT_REPLICATED_DIST_ATTR_TEMPLATE.format(out_name)
        else:
            set_out_dist_attr_code = NONEED_TO_SET_DIST_ATTR_COMMENT_TEMPLATE.format(self.api)
        return set_out_dist_attr_code

    def generate_return_code(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.gene_return_code()

    def generate_auto_paralel_branch(self) -> str:
        if False:
            print('Hello World!')
        if len(self.inputs['names']) == 0:
            return ''
        return MAIN_DIST_BRANCH_TEMPLATE.format(self.generate_infer_spmd_code(), self.generate_output_creation_code(), self.generate_infer_global_shape_code(), self.generate_kernel_selection_code(), self.generate_reshard_input_code(), self.generate_prepare_data_code(), self.generate_infer_meta_code(), self.generate_kernel_call_code(), self.generate_output_dist_attr_setting(), self.generate_return_code())

    def check_argument_whether_support_auto_parallel(self):
        if False:
            print('Hello World!')
        for name in self.inputs['names']:
            if self.inputs['input_info'][name] not in ['const Tensor&', 'const std::vector<Tensor>&', 'const paddle::optional<Tensor>&', 'const paddle::optional<std::vector<Tensor>>&']:
                return False
        for out_type in self.outputs['types']:
            if out_type not in ['Tensor', 'std::vector<Tensor>']:
                return False
        return True

    def gene_base_api_code(self, inplace_flag=False):
        if False:
            while True:
                i = 10
        self.inplace_flag = inplace_flag
        self.dist_output_args = []
        self.dense_output_args = []
        api_func_name = self.get_api_func_name()
        if inplace_flag and api_func_name[-1] != '_':
            api_func_name += '_'
        if len(self.kernel['func']) > 1:
            kernel_dispatch_code = ''
            dist_branch_code = ''
            for kernel_name in self.kernel['func']:
                if 'sparse' not in kernel_name and '_sr' not in kernel_name and (len(self.inputs['names']) > 0) and (len(self.view_map) == 0) and self.check_argument_whether_support_auto_parallel() and (not self.api.endswith('_double_grad')) and (not self.api.endswith('_triple_grad')):
                    dist_branch_code += self.generate_auto_paralel_branch()
            kernel_dispatch_code += dist_branch_code
            for kernel_name in self.kernel['func']:
                kernel_dispatch_code += self.gene_dispatch_code(kernel_name, inplace_flag)
            return API_IMPL_TEMPLATE.format(self.get_return_type(inplace_flag), api_func_name, self.get_define_args(inplace_flag), self.gene_kernel_select(), kernel_dispatch_code + DIPATCH_END_GUARD_TEMPLATE.format(self.api))
        else:
            dist_branch_code = ''
            if len(self.inputs['names']) > 0 and len(self.view_map) == 0 and self.check_argument_whether_support_auto_parallel() and (not self.api.endswith('_double_grad')) and (not self.api.endswith('_triple_grad')):
                dist_branch_code = self.generate_auto_paralel_branch()
            return API_IMPL_TEMPLATE.format(self.get_return_type(inplace_flag), api_func_name, self.get_define_args(inplace_flag), self.gene_kernel_select(), dist_branch_code + self.gen_kernel_code(self.kernel['func'][0], '', inplace_flag))

def generate_api(api_yaml_path, is_fused_ops_yaml, header_file_path, source_file_path):
    if False:
        i = 10
        return i + 15
    apis = []
    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)
    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')
    namespace = api_namespace()
    header_file.write('#pragma once\n')
    header_file.write(header_include())
    header_file.write(namespace[0])
    include_header_file = 'paddle/phi/api/include/fused_api.h' if is_fused_ops_yaml is True else 'paddle/phi/api/include/api.h'
    if is_fused_ops_yaml is True:
        new_apis = [api for api in apis if 'support_dygraph_mode' in api and api['support_dygraph_mode'] is True]
        apis = new_apis
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])
    for api in apis:
        dist_foward_api = DistForwardAPI(api)
        if dist_foward_api.is_dygraph_api:
            dist_foward_api.is_dygraph_api = False
        header_file.write(dist_foward_api.gene_api_declaration())
        if is_fused_ops_yaml is True:
            source_file.write(dist_foward_api.gene_api_code())
        else:
            source_file.write(dist_foward_api.gene_api_code())
    header_file.write(namespace[1])
    source_file.write(namespace[1])
    source_file.write(declare_extension_api())
    header_file.close()
    source_file.close()

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Generate PaddlePaddle C++ API files')
    parser.add_argument('--api_yaml_path', help='path to api yaml file', nargs='+', default=['paddle/phi/api/yaml/ops.yaml'])
    parser.add_argument('--is_fused_ops_yaml', help='flag of fused ops yaml', action='store_true')
    parser.add_argument('--api_header_path', help='output of generated api header code file', default='paddle/phi/api/include/api.h')
    parser.add_argument('--api_source_path', help='output of generated api source code file', default='paddle/phi/api/lib/api.cc')
    options = parser.parse_args()
    api_yaml_path = options.api_yaml_path
    is_fused_ops_yaml = options.is_fused_ops_yaml
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path
    generate_api(api_yaml_path, is_fused_ops_yaml, header_file_path, source_file_path)
if __name__ == '__main__':
    main()