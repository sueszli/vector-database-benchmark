_INFERMETA_NEED_META_CONFIG = {'SplitInferMeta', 'SumInferMeta', 'SplitWithNumInferMeta', 'ConcatInferMeta', 'ReduceIntArrayAxisInferMeta', 'ReshapeWithXShapeInferMeta', 'SliceRawInferMeta', 'StackInferMeta', 'Conv2dTransposeInferMeta'}
_PREPARE_DATA_WITH_VECTOR_INT64_MTTABLE_ATTRIBUTE = {'FrobeniusNormOp'}
OP_BUILD_TEMPLATE = '\nvoid {op_name}::Build({build_args}) {{\n{build_info}\n{get_attributes}\n{build_mutable_attributes}\n{build_inputs}\n{build_attributes}\n{build_outputs}\n}}\n'
OP_INFO_TEMPLATE = '  VLOG(4) << "Start build {op_name}";\n'

def GenBuildInputArgsStr(op_input_name_list, op_attribute_name_list, op_attribute_build_arg_type_list, op_attribute_default_value_list, op_mutable_attribute_name_list, op_non_mutable_attribute_name_list, op_non_mutable_attribute_build_arg_type_list, op_non_mutable_attribute_default_value_list, for_func_define=True, mutable_attr_is_input=False, attr_args_is_map=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Example: pir::Builder &builder, pir::OperationArgument &argument, pir::Value x_, phi::DataType dtype=phi::DataType::UNDEFINED, phi::Place place={}\n    '
    build_args_str = 'pir::Builder &builder, pir::OperationArgument &argument'
    if len(op_input_name_list) > 0:
        for input_name in op_input_name_list:
            build_args_str += ', pir::Value ' + input_name + '_'
    if mutable_attr_is_input:
        if len(op_mutable_attribute_name_list) > 0:
            for mutable_attr in op_mutable_attribute_name_list:
                build_args_str += ', pir::Value ' + mutable_attr + '_'
        if attr_args_is_map:
            build_args_str += ', pir::AttributeMap attributes'
        else:
            for attr_idx in range(len(op_non_mutable_attribute_name_list)):
                build_args_str += ', ' + op_non_mutable_attribute_build_arg_type_list[attr_idx] + ' ' + op_non_mutable_attribute_name_list[attr_idx]
                if for_func_define:
                    if op_non_mutable_attribute_default_value_list[attr_idx] is not None:
                        default_value = op_non_mutable_attribute_default_value_list[attr_idx]
                        if op_non_mutable_attribute_build_arg_type_list[attr_idx] != 'const std::string&':
                            if default_value[0] == "'" or default_value[0] == '"':
                                default_value = default_value[1:]
                            if default_value[-1] == "'" or default_value[-1] == '"':
                                default_value = default_value[0:-1]
                        build_args_str += '=' + default_value
    elif attr_args_is_map:
        build_args_str += ', pir::AttributeMap attributes'
    else:
        for attr_idx in range(len(op_attribute_name_list)):
            build_args_str += ', ' + op_attribute_build_arg_type_list[attr_idx] + ' ' + op_attribute_name_list[attr_idx]
            if for_func_define:
                if op_attribute_default_value_list[attr_idx] is not None:
                    default_value = op_attribute_default_value_list[attr_idx]
                    if op_attribute_build_arg_type_list[attr_idx] != 'const std::string&':
                        if default_value[0] == "'" or default_value[0] == '"':
                            default_value = default_value[1:]
                        if default_value[-1] == "'" or default_value[-1] == '"':
                            default_value = default_value[0:-1]
                    build_args_str += '=' + default_value
    return build_args_str
mutable_attribute_phi_type_maps = {'int': 'phi::DataType::INT32', 'int64_t': 'phi::DataType::INT64', 'float': 'phi::DataType::FLOAT32', 'double': 'phi::DataType::FLOAT64', 'std::vector<int64_t>': 'phi::DataType::INT64', 'const std::vector<int64_t>&': 'phi::DataType::INT64', 'bool': 'phi::DataType::BOOL'}

def GenBuildInserFullForMutableAttribute(op_class_name, op_attribute_name_list, op_attribute_build_arg_type_list, op_mutable_attribute_name_list, op_mutable_attribute_type_list):
    if False:
        print('Hello World!')
    build_mutable_attribute = ''
    BUILD_INTARRAY_ATTRIBUTE_TEMPLATE = '  // Generate int_array mutable attribute: {attr_name}\n  paddle::dialect::FullIntArrayOp full_{attr_name}_op = builder.Build<paddle::dialect::FullIntArrayOp>({attr_name}, {phi_dtype}, phi::CPUPlace());\n  pir::OpResult {attr_name}_ = full_{attr_name}_op->result(0);\n    '
    BUILD_SCALAR_ATTRIBUTE_TEMPLATE = '  // Generate scalar mutable attribute: {attr_name}\n  paddle::dialect::FullOp full_{attr_name}_op = builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{{1}}, {attr_name}, {phi_dtype}, phi::CPUPlace());\n  pir::OpResult {attr_name}_ = full_{attr_name}_op->result(0);\n    '
    for idx in range(len(op_mutable_attribute_name_list)):
        attr_name = op_mutable_attribute_name_list[idx]
        attr_type = op_mutable_attribute_type_list[idx][0]
        if attr_name in op_attribute_name_list:
            phi_dtype = mutable_attribute_phi_type_maps[op_attribute_build_arg_type_list[op_attribute_name_list.index(attr_name)]]
        else:
            phi_dtype = mutable_attribute_phi_type_maps[op_mutable_attribute_type_list[idx][1]]
        if attr_type == 'paddle::dialect::IntArrayAttribute':
            build_mutable_attribute += BUILD_INTARRAY_ATTRIBUTE_TEMPLATE.format(attr_name=attr_name, phi_dtype=phi_dtype)
        else:
            build_mutable_attribute += BUILD_SCALAR_ATTRIBUTE_TEMPLATE.format(attr_name=attr_name, phi_dtype=phi_dtype)
    return build_mutable_attribute

def GenBuildInputs(op_input_name_list, op_mutable_attribute_name_list):
    if False:
        while True:
            i = 10
    BUILD_INPUT_TEMPLATE = '  std::vector<pir::Value> argument_inputs = {{{inputs_args}}};\n  argument.AddInputs(argument_inputs);\n'
    build_input_str = '  VLOG(4) << "Builder construction inputs";\n'
    input_name_list = op_input_name_list + op_mutable_attribute_name_list
    if len(input_name_list) > 0:
        inputs_args_str = ''
        inputs_args_str += '_, '.join(input_name_list) + '_'
        build_input_str += BUILD_INPUT_TEMPLATE.format(inputs_args=inputs_args_str)
    return build_input_str

def GenBuildAttributes(op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list):
    if False:
        for i in range(10):
            print('nop')
    INTARRAY_STR_TEMPLATE = '  pir::Attribute attr_{attr_name} = {op_attribute_type}::get(pir::IrContext::Instance(), phi::IntArray({attr}));\n'
    SCALAR_STR_TEMPLATE = '  pir::Attribute attr_{attr_name} = paddle::dialect::TransToIrAttribute({attr}, pir::IrContext::Instance());\n'
    STR_TEMPLATE = '  pir::Attribute attr_{attr_name} = {op_attribute_type}::get(pir::IrContext::Instance(), {attr});\n'
    ARRAY_ATTRIBUTE_TEMPLATE = '  std::vector<pir::Attribute> vec_{attr_name};\n  for (size_t i = 0; i < static_cast<size_t>({attr_size}); i++) {{\n    {create_attribute}\n    vec_{attr_name}.push_back(attr_{attr_name});\n  }}\n  pir::Attribute attr_{attr_name} = pir::ArrayAttribute::get(pir::IrContext::Instance(), vec_{attr_name});\n'
    attr_str = '  VLOG(4) << "Builder construction attributes";\n'
    array_attr_type = 'pir::ArrayAttribute<'
    for idx in range(len(op_non_mutable_attribute_name_list)):
        if array_attr_type in op_non_mutable_attribute_type_list[idx]:
            inner_attribute_type = op_non_mutable_attribute_type_list[idx][len(array_attr_type):-1]
            if inner_attribute_type == 'paddle::dialect::IntArrayAttribute':
                attr_str += ARRAY_ATTRIBUTE_TEMPLATE.format(attr_name=op_non_mutable_attribute_name_list[idx], attr_size=op_non_mutable_attribute_name_list[idx] + '.size()', create_attribute=INTARRAY_STR_TEMPLATE.format(attr_name=op_non_mutable_attribute_name_list[idx], op_attribute_type=inner_attribute_type, attr=op_non_mutable_attribute_name_list[idx] + '[i]'))
            elif inner_attribute_type == 'paddle::dialect::ScalarAttribute':
                attr_str += ARRAY_ATTRIBUTE_TEMPLATE.format(attr_name=op_non_mutable_attribute_name_list[idx], attr_size=op_non_mutable_attribute_name_list[idx] + '.size()', create_attribute=SCALAR_STR_TEMPLATE.format(attr_name=op_non_mutable_attribute_name_list[idx], attr=op_non_mutable_attribute_name_list[idx] + '[i]'))
            else:
                attr_str += ARRAY_ATTRIBUTE_TEMPLATE.format(attr_name=op_non_mutable_attribute_name_list[idx], attr_size=op_non_mutable_attribute_name_list[idx] + '.size()', create_attribute=STR_TEMPLATE.format(attr_name=op_non_mutable_attribute_name_list[idx], op_attribute_type=inner_attribute_type, attr=op_non_mutable_attribute_name_list[idx] + '[i]'))
        elif op_non_mutable_attribute_type_list[idx] == 'paddle::dialect::IntArrayAttribute':
            attr_str += INTARRAY_STR_TEMPLATE.format(attr_name=op_non_mutable_attribute_name_list[idx], op_attribute_type=op_non_mutable_attribute_type_list[idx], attr=op_non_mutable_attribute_name_list[idx])
        elif op_non_mutable_attribute_type_list[idx] == 'paddle::dialect::ScalarAttribute':
            attr_str += SCALAR_STR_TEMPLATE.format(attr_name=op_non_mutable_attribute_name_list[idx], attr=op_non_mutable_attribute_name_list[idx])
        else:
            attr_str += STR_TEMPLATE.format(attr_name=op_non_mutable_attribute_name_list[idx], op_attribute_type=op_non_mutable_attribute_type_list[idx], attr=op_non_mutable_attribute_name_list[idx])
        attr_str += '  argument.AddAttribute("{attr_name}", attr_{attr_name});\n'.format(attr_name=op_non_mutable_attribute_name_list[idx])
    return attr_str

def GenBuildOutputs(op_class_name, op_input_name_list, op_input_type_list, op_input_optional_list, op_mutable_attribute_name_list, op_mutable_attribute_type_list, op_output_name_list, op_output_type_list, op_output_size_list, op_output_optional_list, op_infer_meta_map, op_inplace_map, mutable_attr_is_input=False):
    if False:
        while True:
            i = 10
    build_output_str = '  VLOG(4) << "Builder construction outputs";\n'
    CREATE_INPUT_METATENSOR_TEMPLATE = '\n  VLOG(4) << "Builder construction  dense_{name}";\n  paddle::dialect::IrTensor ir_tensor_{name}(paddle::dialect::TransToPhiDataType({name}.dtype()),\n                                                      {name}.dims(),\n                                                      {name}.data_layout(),\n                                                      {name}.lod(),\n                                                      {name}.offset());\n  VLOG(4) << "Builder construction  meta_{name}";\n  paddle::dialect::IrMetaTensor meta_{name}(&ir_tensor_{name});\n'
    CREATE_OPTIONAL_INPUT_METATENSOR_TEMPLATE = '\n  paddle::dialect::IrMetaTensor meta_{name};\n  paddle::dialect::IrTensor ir_tensor_{name};\n  if ({name}_.impl() != nullptr) {{\n    paddle::dialect::DenseTensorType {name} = {name}_.type().dyn_cast<paddle::dialect::DenseTensorType>();\n    VLOG(4) << "Builder construction  dense_{name}";\n    ir_tensor_{name} = paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}.dtype()),\n                                                        {name}.dims(),\n                                                        {name}.data_layout(),\n                                                        {name}.lod(),\n                                                        {name}.offset());\n    VLOG(4) << "Builder construction  meta_{name}";\n    meta_{name} = paddle::dialect::IrMetaTensor(&ir_tensor_{name});\n  }}\n\n'
    CREATE_INPUT_VEC_METATENSOR_TEMPLATE = '  std::vector<paddle::dialect::IrTensor> vec_ir_tensor_{name};\n  for (size_t i=0; i < static_cast<size_t>({name}.size()); i++) {{\n    vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),\n                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),\n                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),\n                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),\n                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));\n  }}\n  std::vector<paddle::dialect::IrMetaTensor> vec_meta_{name};\n  for (size_t i=0; i < vec_ir_tensor_{name}.size(); i++) {{\n    vec_meta_{name}.push_back(paddle::dialect::IrMetaTensor(&vec_ir_tensor_{name}[i]));\n  }}\n\n  std::vector<const phi::MetaTensor*> meta_{name};\n  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{\n    meta_{name}.push_back(&vec_meta_{name}[i]);\n  }}\n '
    CREATE_OPTIONAL_INPUT_VEC_METATENSOR_TEMPLATE = '  std::vector<paddle::dialect::IrTensor> vec_ir_tensor_{name};\n  if ({name}_.impl() != nullptr) {{\n    pir::VectorType {name} = {name}_.type().dyn_cast<pir::VectorType>();\n    for (size_t i=0; i < static_cast<size_t>({name}.size()); i++) {{\n        vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),\n                                                                        {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),\n                                                                        {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),\n                                                                        {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),\n                                                                        {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));\n    }}\n  }}\n\n  std::vector<paddle::dialect::IrMetaTensor> vec_meta_{name};\n  for (size_t i=0; i < vec_ir_tensor_{name}.size(); i++) {{\n    vec_meta_{name}.push_back(paddle::dialect::IrMetaTensor(&vec_ir_tensor_{name}[i]));\n  }}\n\n  std::vector<const phi::MetaTensor*> meta_{name};\n  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{\n    meta_{name}.push_back(&vec_meta_{name}[i]);\n  }}\n\n'
    CREATE_INTARRAY_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE = '  phi::IntArray {name};\n  if ({name}_.dyn_cast<pir::OpResult>().owner()->isa<paddle::dialect::FullIntArrayOp>()) {{\n    {name} = std::move(phi::IntArray(paddle::dialect::GetInt64Vector(\n                          {name}_.dyn_cast<pir::OpResult>().owner()\n                          ->dyn_cast<paddle::dialect::FullIntArrayOp>()\n                          .attribute("value"))));\n  }} else if ({name}_.type().isa<pir::VectorType>()) {{\n    size_t {name}_size = {name}_.type().dyn_cast<pir::VectorType>().size();\n    {name} = std::move(phi::IntArray(std::vector<int64_t>({name}_size, -1)));\n    {name}.SetFromTensor(true);\n  }} else if ({name}_.type().isa<paddle::dialect::DenseTensorType>()) {{\n    size_t {name}_size = phi::product({name}_.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());\n    {name} = std::move(phi::IntArray(std::vector<int64_t>({name}_size, -1)));\n    {name}.SetFromTensor(true);\n  }} else {{\n    PADDLE_THROW(phi::errors::Unimplemented("Only support VectorType or DenseTensorType"));\n  }}\n'
    CREATE_VECTOR_INT_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE = '  std::vector<int64_t> {name};\n  if ({name}_.dyn_cast<pir::OpResult>().owner()->isa<paddle::dialect::FullIntArrayOp>()) {{\n    {name} = paddle::dialect::GetInt64Vector(\n                    {name}_.dyn_cast<pir::OpResult>().owner()\n                    ->dyn_cast<paddle::dialect::FullIntArrayOp>()\n                    .attribute("value"));\n  }} else if ({name}_.type().isa<pir::VectorType>()) {{\n    size_t {name}_size = {name}_.type().dyn_cast<pir::VectorType>().size();\n    {name} = std::vector<int64_t>({name}_size, -1);\n  }} else if ({name}_.type().isa<paddle::dialect::DenseTensorType>()) {{\n    size_t {name}_size = phi::product({name}_.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());\n    {name} = std::vector<int64_t>({name}_size, -1);\n  }} else {{\n    PADDLE_THROW(phi::errors::Unimplemented("Only support VectorType or DenseTensorType"));\n  }}\n'
    CREATE_SCALAR_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE = '  phi::Scalar {name};\n  if ({name}_.dyn_cast<pir::OpResult>().owner()->isa<paddle::dialect::FullOp>()) {{\n    {name} = std::move(phi::Scalar({name}_.dyn_cast<pir::OpResult>().owner()\n                                  ->dyn_cast<paddle::dialect::FullOp>()\n                                  .attribute("value")\n                                  .dyn_cast<paddle::dialect::ScalarAttribute>()\n                                  .data()\n                                  .to<int>()));\n  }}\n  else {{\n    {name} = std::move(phi::Scalar(-1));\n    {name}.SetFromTensor(true);\n  }}\n'
    CREATE_OUTPUT_METATENSOR_TEMPLATE = '  paddle::dialect::IrTensor dense_{name};\n  paddle::dialect::IrMetaTensor meta_{name}(&dense_{name});\n'
    CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE = '  std::vector<paddle::dialect::IrTensor> vec_dense_{name}(({output_size}), paddle::dialect::IrTensor());\n  std::vector<paddle::dialect::IrMetaTensor> vec_meta_{name};\n  for (size_t i=0; i < static_cast<size_t>({output_size}); i++) {{\n    vec_meta_{name}.push_back(paddle::dialect::IrMetaTensor(&vec_dense_{name}[i]));\n  }}\n  std::vector<phi::MetaTensor*> meta_{name};\n  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{\n    meta_{name}.push_back(&vec_meta_{name}[i]);\n  }}\n'
    for idx in range(len(op_input_name_list)):
        if 'pir::VectorType' in op_input_type_list[idx]:
            if op_input_optional_list[idx] == 'false':
                build_output_str += '  pir::VectorType {name} = {name}_.type().dyn_cast<pir::VectorType>(); (void){name};\n'.format(name=op_input_name_list[idx])
        elif op_input_optional_list[idx] == 'false':
            build_output_str += '  paddle::dialect::DenseTensorType {name} = {name}_.type().dyn_cast<paddle::dialect::DenseTensorType>(); (void){name};\n'.format(name=op_input_name_list[idx])
    if mutable_attr_is_input:
        for idx in range(len(op_mutable_attribute_name_list)):
            attr_dtype = op_mutable_attribute_type_list[idx]
            if attr_dtype[0] == 'paddle::dialect::IntArrayAttribute':
                if op_class_name in _PREPARE_DATA_WITH_VECTOR_INT64_MTTABLE_ATTRIBUTE:
                    build_output_str += CREATE_VECTOR_INT_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE.format(name=op_mutable_attribute_name_list[idx])
                else:
                    build_output_str += CREATE_INTARRAY_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE.format(name=op_mutable_attribute_name_list[idx])
            elif attr_dtype[0] == 'paddle::dialect::ScalarAttribute':
                build_output_str += CREATE_SCALAR_MUTABLE_ATTRIBUE_WITH_UNKONW_DATA_TEMPLATE.format(name=op_mutable_attribute_name_list[idx], dtype=attr_dtype[1])
            elif attr_dtype[0] == 'pir::StrAttribute':
                build_output_str += ''
            else:
                assert 'mutable attribtue type is not right.'
        build_output_str += '\n'
    infer_meta_args = []
    for idx in range(len(op_infer_meta_map['param'])):
        if op_infer_meta_map['param'][idx] in op_input_name_list:
            if 'meta_' + op_infer_meta_map['param'][idx] not in infer_meta_args:
                if 'pir::VectorType' in op_input_type_list[op_input_name_list.index(op_infer_meta_map['param'][idx])]:
                    input_index = op_input_name_list.index(op_infer_meta_map['param'][idx])
                    if op_input_optional_list[input_index] == 'true':
                        build_output_str += CREATE_OPTIONAL_INPUT_VEC_METATENSOR_TEMPLATE.format(name=op_infer_meta_map['param'][idx])
                    else:
                        build_output_str += CREATE_INPUT_VEC_METATENSOR_TEMPLATE.format(name=op_infer_meta_map['param'][idx])
                else:
                    input_index = op_input_name_list.index(op_infer_meta_map['param'][idx])
                    if op_input_optional_list[input_index] == 'true':
                        build_output_str += CREATE_OPTIONAL_INPUT_METATENSOR_TEMPLATE.format(name=op_infer_meta_map['param'][idx])
                    else:
                        build_output_str += CREATE_INPUT_METATENSOR_TEMPLATE.format(name=op_infer_meta_map['param'][idx])
            infer_meta_args.append('meta_' + op_infer_meta_map['param'][idx])
        else:
            infer_meta_args.append(op_infer_meta_map['param'][idx])
    for idx in range(len(op_output_name_list)):
        if 'pir::VectorType' in op_output_type_list[idx]:
            build_output_str += CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE.format(name=op_output_name_list[idx], output_size=op_output_size_list[idx])
            infer_meta_args.append(f'meta_{op_output_name_list[idx]}')
        else:
            build_output_str += CREATE_OUTPUT_METATENSOR_TEMPLATE.format(name=op_output_name_list[idx])
            infer_meta_args.append(f'&meta_{op_output_name_list[idx]}')
    CREATE_INFER_META_FUNC_TEMPLATE = '\n  phi::{func}({args});\n'
    CREATE_INFER_META_FUNC_WITH_METACINFIG_TEMPLATE = '\n  phi::{func}({args}, phi::MetaConfig(false, false));\n'
    if op_infer_meta_map['func'] in _INFERMETA_NEED_META_CONFIG:
        build_output_str += CREATE_INFER_META_FUNC_WITH_METACINFIG_TEMPLATE.format(func=op_infer_meta_map['func'], args=', '.join(infer_meta_args))
    else:
        build_output_str += CREATE_INFER_META_FUNC_TEMPLATE.format(func=op_infer_meta_map['func'], args=', '.join(infer_meta_args))
    build_output_str += '\n  std::vector<pir::Type> argument_outputs;'
    CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE = '\n  pir::Type {name}_dense_tensor_type = paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_{name}.dtype()), dense_{name}.dims(), dense_{name}.layout(), dense_{name}.lod(), dense_{name}.offset());\n  argument_outputs.push_back({name}_dense_tensor_type);\n'
    CREATE_OUTPUT_INPLACE_OPTIONAL_DENSE_TENSOR_TEMPLATE = '\n  if ({input_name}_.impl() != nullptr) {{\n    pir::Type {output_name}_dense_tensor_type = paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_{output_name}.dtype()), dense_{output_name}.dims(), dense_{output_name}.layout(), dense_{output_name}.lod(), dense_{output_name}.offset());\n    argument_outputs.push_back({output_name}_dense_tensor_type);\n  }} else {{\n    pir::Type {output_name}_type;\n    argument_outputs.push_back({output_name}_type);\n  }}\n\n'
    CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE = '\n  std::vector<pir::Type> {name}_types;\n  for (size_t i=0; i < static_cast<size_t>({output_size}); i++) {{\n    {name}_types.push_back(paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(vec_dense_{name}[i].dtype()), vec_dense_{name}[i].dims(), vec_dense_{name}[i].layout(), vec_dense_{name}[i].lod(), vec_dense_{name}[i].offset()));\n  }}\n  pir::Type {name}_vector_type = pir::VectorType::get(pir::IrContext::Instance(), {name}_types);\n  argument_outputs.push_back({name}_vector_type);\n'
    for idx in range(len(op_output_name_list)):
        if 'pir::VectorType' in op_output_type_list[idx]:
            build_output_str += CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE.format(name=op_output_name_list[idx], output_size=op_output_size_list[idx])
        else:
            output_name = op_output_name_list[idx]
            has_input_inplace = op_inplace_map is not None and output_name in op_inplace_map.keys()
            if op_output_optional_list[idx] == 'true' and has_input_inplace:
                build_output_str += CREATE_OUTPUT_INPLACE_OPTIONAL_DENSE_TENSOR_TEMPLATE.format(input_name=op_inplace_map[output_name], output_name=output_name)
            else:
                build_output_str += CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE.format(name=output_name)
    build_output_str += '  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());\n'
    build_output_str += '  ::pir::PassStopGradientsDefaultly(argument);\n'
    return build_output_str

def gen_build_func_str(op_class_name, op_input_name_list, op_input_type_list, op_input_optional_list, op_attribute_name_list, op_attribute_type_list, op_attribute_build_arg_type_list, op_attribute_default_value_list, op_mutable_attribute_name_list, op_mutable_attribute_type_list, op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list, op_non_mutable_attribute_build_arg_type_list, op_non_mutable_attribute_default_value_list, op_output_name_list, op_output_type_list, op_output_size_list, op_output_optional_list, op_infer_meta_map, op_inplace_map, muta_attr_is_input=False, attr_args_is_map=False):
    if False:
        return 10
    build_args_for_declare = ''
    build_func = ''
    build_info_str = OP_INFO_TEMPLATE.format(op_name=op_class_name)
    build_args_for_declare = GenBuildInputArgsStr(op_input_name_list, op_attribute_name_list, op_attribute_build_arg_type_list, op_attribute_default_value_list, op_mutable_attribute_name_list, op_non_mutable_attribute_name_list, op_non_mutable_attribute_build_arg_type_list, op_non_mutable_attribute_default_value_list, True, muta_attr_is_input, attr_args_is_map)
    build_args_for_define = GenBuildInputArgsStr(op_input_name_list, op_attribute_name_list, op_attribute_build_arg_type_list, op_attribute_default_value_list, op_mutable_attribute_name_list, op_non_mutable_attribute_name_list, op_non_mutable_attribute_build_arg_type_list, op_non_mutable_attribute_default_value_list, False, muta_attr_is_input, attr_args_is_map)
    inset_full_for_mutable_attributes_str = ''
    if not muta_attr_is_input:
        inset_full_for_mutable_attributes_str = GenBuildInserFullForMutableAttribute(op_class_name, op_attribute_name_list, op_attribute_build_arg_type_list, op_mutable_attribute_name_list, op_mutable_attribute_type_list)
    build_inputs_str = GenBuildInputs(op_input_name_list, op_mutable_attribute_name_list)
    build_attributes_str = GenBuildAttributes(op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list)
    build_outputs_str = GenBuildOutputs(op_class_name, op_input_name_list, op_input_type_list, op_input_optional_list, op_mutable_attribute_name_list, op_mutable_attribute_type_list, op_output_name_list, op_output_type_list, op_output_size_list, op_output_optional_list, op_infer_meta_map, op_inplace_map, muta_attr_is_input)
    GET_ATTRIBUTES_FROM_MAP_TEMPLATE = '\n  IR_ENFORCE(\n      attributes.find("{attribute_name}") != attributes.end(),\n          "\'{attribute_name}\' Attribute is expected for {op_name}. ");\n  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<{attr_ir_type}>().data();\n'
    GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE = '\n  IR_ENFORCE(\n      attributes.find("{attribute_name}") != attributes.end(),\n          "\'{attribute_name}\' Attribute is expected for {op_name}. ");\n  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<pir::StrAttribute>().AsString();\n'
    GET_ARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = '\n  IR_ENFORCE(\n      attributes.find("{attribute_name}") != attributes.end(),\n          "\'{attribute_name}\' Attribute is expected for {op_name}. ");\n  {attr_type} {attribute_name};\n  for (size_t i = 0; i < attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().size(); i++) {{\n    {attribute_name}.push_back(attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().at(i).dyn_cast<{inner_type}>().{data_name}());\n  }}\n'
    GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = '\n  IR_ENFORCE(\n      attributes.find("{attribute_name}") != attributes.end(),\n          "\'{attribute_name}\' Attribute is expected for {op_name}. ");\n  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();\n'
    GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE = '\n  IR_ENFORCE(\n      attributes.find("{attribute_name}") != attributes.end(),\n          "\'{attribute_name}\' Attribute is expected for {op_name}. ");\n  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::ScalarAttribute>().data().to<{attr_type}>();\n'
    get_attributes_str = ''
    array_attr_str = 'pir::ArrayAttribute'
    attr_names = []
    attr_types = []
    attr_build_arg_types = []
    if not muta_attr_is_input:
        attr_names = op_attribute_name_list
        attr_types = op_attribute_type_list
        attr_build_arg_types = op_attribute_build_arg_type_list
    else:
        attr_names = op_non_mutable_attribute_name_list
        attr_types = op_non_mutable_attribute_type_list
        attr_build_arg_types = op_non_mutable_attribute_build_arg_type_list
    if attr_args_is_map:
        for idx in range(len(attr_names)):
            attr_type = attr_build_arg_types[idx]
            attr_type = attr_type.replace('const ', '')
            attr_type = attr_type.replace('&', '')
            if array_attr_str in attr_types[idx]:
                inner_type = attr_types[idx][len(array_attr_str) + 1:-1]
                data_name = 'data'
                if inner_type == 'pir::StrAttribute':
                    data_name = 'AsString'
                get_attributes_str += GET_ARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE.format(op_name=op_class_name, attr_type=attr_type, attribute_name=attr_names[idx], inner_type=inner_type, data_name=data_name)
            elif 'paddle::dialect::IntArrayAttribute' in attr_types[idx]:
                get_attributes_str += GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE.format(op_name=op_class_name, attr_type=attr_type, attribute_name=attr_names[idx])
            elif 'paddle::dialect::ScalarAttribute' in attr_types[idx]:
                get_attributes_str += GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE.format(op_name=op_class_name, attr_type=attr_type, attribute_name=attr_names[idx])
            elif 'pir::StrAttribute' in attr_types[idx]:
                get_attributes_str += GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE.format(op_name=op_class_name, attr_type=attr_type, attribute_name=attr_names[idx], attr_ir_type=attr_types[idx])
            else:
                get_attributes_str += GET_ATTRIBUTES_FROM_MAP_TEMPLATE.format(op_name=op_class_name, attr_type=attr_type, attribute_name=attr_names[idx], attr_ir_type=attr_types[idx])
    build_func = OP_BUILD_TEMPLATE.format(op_name=op_class_name, build_info=build_info_str, build_args=build_args_for_define, build_mutable_attributes=inset_full_for_mutable_attributes_str, get_attributes=get_attributes_str, build_inputs=build_inputs_str, build_attributes=build_attributes_str, build_outputs=build_outputs_str)
    return (build_args_for_declare, build_func)
OP_BUILD_BY_INVOKE_TEMPLATE = '\nvoid {op_name}::Build({build_args}) {{\n  {invoke_class}::Build(builder, argument{invoke_args});\n}}\n'

def gen_build_func_str_by_invoke(op_class_name, op_input_name_list, op_input_type_list, op_input_optional_list, op_attribute_name_list, op_attribute_type_list, op_attribute_build_arg_type_list, op_attribute_default_value_list, op_mutable_attribute_name_list, op_mutable_attribute_type_list, op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list, op_non_mutable_attribute_build_arg_type_list, op_non_mutable_attribute_default_value_list, op_invoke_class_name, op_invoke_map):
    if False:
        while True:
            i = 10
    build_args_for_declare = ''
    build_func = ''
    build_args_for_declare = GenBuildInputArgsStr(op_input_name_list, op_attribute_name_list, op_attribute_build_arg_type_list, op_attribute_default_value_list, op_mutable_attribute_name_list, op_non_mutable_attribute_name_list, op_non_mutable_attribute_build_arg_type_list, op_non_mutable_attribute_default_value_list, True, False, False)
    build_args_for_define = GenBuildInputArgsStr(op_input_name_list, op_attribute_name_list, op_attribute_build_arg_type_list, op_attribute_default_value_list, op_mutable_attribute_name_list, op_non_mutable_attribute_name_list, op_non_mutable_attribute_build_arg_type_list, op_non_mutable_attribute_default_value_list, False, False, False)
    invoke_args = op_invoke_map['args'].split(', ')
    invoke_args_str = ''
    for item in invoke_args:
        if item in op_input_name_list:
            invoke_args_str += ', ' + item + '_'
        elif '.dtype()' in item:
            invoke_args_str += ', paddle::dialect::TransToPhiDataType(' + item[:-8] + '_' + '.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype())'
        else:
            invoke_args_str += ', ' + item
    build_func = OP_BUILD_BY_INVOKE_TEMPLATE.format(op_name=op_class_name, build_args=build_args_for_define, invoke_class=op_invoke_class_name, invoke_args=invoke_args_str)
    return (build_args_for_declare, build_func)