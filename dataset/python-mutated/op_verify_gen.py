OP_VERIFY_TEMPLATE = '\nvoid {op_name}::VerifySig() {{\n  VLOG(4) << "Start Verifying inputs, outputs and attributes for: {op_name}.";\n  VLOG(4) << "Verifying inputs:";\n  {{\n  auto input_size = num_operands();\n  IR_ENFORCE(input_size == {inputs_size}u,\n                    "The size %d of inputs must be equal to {inputs_size}.", input_size);{inputs_type_check}\n  }}\n  VLOG(4) << "Verifying attributes:";\n  {{{attributes_check}\n  }}\n  VLOG(4) << "Verifying outputs:";\n  {{\n  auto output_size = num_results();\n  IR_ENFORCE(output_size == {outputs_size}u,\n                    "The size %d of outputs must be equal to {outputs_size}.", output_size);{outputs_type_check}\n  }}\n  VLOG(4) << "End Verifying for: {op_name}.";\n}}\n'
GRAD_OP_VERIFY_TEMPLATE = '\nvoid {op_name}::VerifySig() {{}}\n'
INPUT_TYPE_CHECK_TEMPLATE = '\n  IR_ENFORCE((*this)->operand_source({index}).type().isa<{standard}>(),\n                  "Type validation failed for the {index}th input.");'
INPUT_VECTORTYPE_CHECK_TEMPLATE = '\n  if (auto vec_type = (*this)->operand_source({index}).type().dyn_cast<pir::VectorType>()) {{\n      for (size_t i = 0; i < vec_type.size(); ++i) {{\n        IR_ENFORCE(vec_type[i].isa<{standard}>(),\n                       "Type validation failed for the {index}th input.");\n      }}\n  }}\n  else {{\n    IR_ENFORCE((*this)->operand_source({index}).type().isa<{standard}>(),\n                   "Type validation failed for the {index}th input.");\n  }}'
INPUT_OPTIONAL_TYPE_CHECK_TEMPLATE = '\n  if (auto val = (*this)->operand({index})) {{\n    IR_ENFORCE(val.type().isa<{standard}>(),\n                   "Type validation failed for the {index}th input.");\n  }}'
INPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE = '\n  if (auto val =  (*this)->operand({index})) {{\n    if (auto vec_type = val.type().dyn_cast<pir::VectorType>()) {{\n      for (size_t i = 0; i < vec_type.size(); i++) {{\n        IR_ENFORCE(vec_type[i].isa<{standard}>(),\n                          "Type validation failed for the {index}th input.");\n      }}\n    }}\n    else {{\n      IR_ENFORCE(val.type().isa<{standard}>(),\n                        "Type validation failed for the {index}th input.");\n    }}\n  }}'
ATTRIBUTE_CHECK_TEMPLATE = '\n  IR_ENFORCE(attributes.count("{attribute_name}")>0,\n                 "{attribute_name} does not exist.");\n  IR_ENFORCE(attributes.at("{attribute_name}").isa<{standard}>(),\n                 "Type of attribute: {attribute_name} is not {standard}.");\n'
ATTRIBUTE_VECTOR_CHECK_TEMPLATE = '\n  IR_ENFORCE(attributes.count("{attribute_name}")>0,\n                 "{attribute_name} does not exist.");\n  IR_ENFORCE(attributes.at("{attribute_name}").isa<pir::ArrayAttribute>(),\n                 "Type of attribute: {attribute_name} is not pir::ArrayAttribute.");\n  for (size_t i = 0; i < attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().size(); i++) {{\n    IR_ENFORCE(attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().at(i).isa<{standard}>(),\n                   "Type of attribute: {attribute_name} is not right.");\n  }}'
OUTPUT_TYPE_CHECK_TEMPLATE = '\n  IR_ENFORCE((*this)->result({index}).type().isa<{standard}>(),\n                 "Type validation failed for the {index}th output.");'
OUTPUT_VECTORTYPE_CHECK_TEMPLATE = '\n  auto output_{index}_type = (*this)->result({index}).type();\n  if (auto vec_type = output_{index}_type.dyn_cast<pir::VectorType>()) {{\n    for (size_t i = 0; i < vec_type.size(); i++) {{\n      IR_ENFORCE(vec_type[i].isa<{standard}>(),\n                     "Type validation failed for the {index}th output.");\n    }}\n  }}\n  else {{\n    IR_ENFORCE(output_{index}_type.isa<{standard}>(),\n                   "Type validation failed for the {index}th output.");\n  }}'
OUTPUT_OPTIONAL_TYPE_CHECK_TEMPLATE = '\n  if (auto output_{index}_type = (*this)->result({index}).type()) {{\n    IR_ENFORCE(output_{index}_type.isa<{standard}>(),\n                   "Type validation failed for the {index}th output.");\n  }}'
OUTPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE = '\n  if (auto output_{index}_type = (*this)->result({index}).type()) {{\n    if (auto vec_type = output_{index}_type.dyn_cast<pir::VectorType>()) {{\n      for (size_t i = 0; i < vec_type.size(); ++i) {{\n        IR_ENFORCE(vec_type[i].isa<{standard}>(),\n                       "Type validation failed for the {index}th output.");\n      }}\n    }}\n    else {{\n      IR_ENFORCE(output_{index}_type.isa<{standard}>(),\n                     "Type validation failed for the {index}th output.");\n    }}\n  }}'

def gen_inputs_type_check_str(op_input_type_list, op_input_optional_list, op_mutable_attribute_name_list, op_mutable_attribute_type_list):
    if False:
        print('Hello World!')
    if len(op_input_type_list) + len(op_mutable_attribute_name_list) == 0:
        inputs_type_check_str = '\n  // Inputs num is 0, not need to check inputs type.'
    else:
        inputs_type_check_str = ''
    vector_type_str = 'pir::VectorType<'
    for idx in range(len(op_input_type_list)):
        input_type = op_input_type_list[idx]
        is_optional = op_input_optional_list[idx]
        is_vector = False
        if input_type.startswith(vector_type_str):
            is_vector = True
            input_type = input_type[len(vector_type_str):-1]
        check_str = ''
        if is_optional == 'true':
            if is_vector:
                check_str = INPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE.format(index=idx, standard=input_type)
            else:
                check_str = INPUT_OPTIONAL_TYPE_CHECK_TEMPLATE.format(index=idx, standard=input_type)
        elif is_vector:
            check_str = INPUT_VECTORTYPE_CHECK_TEMPLATE.format(index=idx, standard=input_type)
        else:
            check_str = INPUT_TYPE_CHECK_TEMPLATE.format(index=idx, standard=input_type)
        inputs_type_check_str += check_str
    for idx in range(len(op_mutable_attribute_name_list)):
        mutable_attribute_type = op_mutable_attribute_type_list[idx][0]
        check_str = ''
        if mutable_attribute_type == 'paddle::dialect::ScalarAttribute':
            check_str = INPUT_TYPE_CHECK_TEMPLATE.format(index=idx + len(op_input_type_list), standard='paddle::dialect::DenseTensorType')
        else:
            check_str = INPUT_VECTORTYPE_CHECK_TEMPLATE.format(index=idx + len(op_input_type_list), standard='paddle::dialect::DenseTensorType')
        inputs_type_check_str += check_str
    return inputs_type_check_str

def gen_attributes_type_check_str(op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list):
    if False:
        for i in range(10):
            print('nop')
    if len(op_non_mutable_attribute_name_list) == 0:
        attributes_check_str = '\n  // Attributes num is 0, not need to check attributes type.'
    else:
        attributes_check_str = '\n  auto& attributes = this->attributes();'
    array_attr_str = 'pir::ArrayAttribute<'
    for idx in range(len(op_non_mutable_attribute_name_list)):
        attribute_name = op_non_mutable_attribute_name_list[idx]
        attribute_type = op_non_mutable_attribute_type_list[idx]
        if attribute_type.startswith(array_attr_str):
            attribute_type = attribute_type[len(array_attr_str):-1]
            attributes_check_str += ATTRIBUTE_VECTOR_CHECK_TEMPLATE.format(attribute_name=attribute_name, standard=attribute_type)
        else:
            attributes_check_str += ATTRIBUTE_CHECK_TEMPLATE.format(attribute_name=attribute_name, standard=attribute_type)
    return attributes_check_str

def gen_outputs_type_check_str(op_output_type_list, op_output_optional_list):
    if False:
        for i in range(10):
            print('nop')
    if len(op_output_type_list) == 0:
        outputs_type_check_str = '\n  // Outputs num is 0, not need to check outputs type.'
    else:
        outputs_type_check_str = ''
    vector_type_str = 'pir::VectorType<'
    for idx in range(len(op_output_type_list)):
        output_type = op_output_type_list[idx]
        is_optional = op_output_optional_list[idx]
        is_vector = False
        if output_type.startswith(vector_type_str):
            is_vector = True
            output_type = output_type[len(vector_type_str):-1]
        check_str = ''
        if is_optional == 'true':
            if is_vector:
                check_str = OUTPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE.format(index=idx, standard=output_type)
            else:
                check_str = OUTPUT_OPTIONAL_TYPE_CHECK_TEMPLATE.format(index=idx, standard=output_type)
        elif is_vector:
            check_str = OUTPUT_VECTORTYPE_CHECK_TEMPLATE.format(index=idx, standard=output_type)
        else:
            check_str = OUTPUT_TYPE_CHECK_TEMPLATE.format(index=idx, standard=output_type)
        outputs_type_check_str += check_str
    return outputs_type_check_str

def gen_verify_func_str(op_class_name, op_input_type_list, op_input_optional_list, op_mutable_attribute_name_list, op_mutable_attribute_type_list, op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list, op_output_type_list, op_output_optional_list):
    if False:
        while True:
            i = 10
    if 'GradOp' in op_class_name or 'Grad_Op' in op_class_name:
        return GRAD_OP_VERIFY_TEMPLATE.format(op_name=op_class_name)
    inputs_type_check_str = gen_inputs_type_check_str(op_input_type_list, op_input_optional_list, op_mutable_attribute_name_list, op_mutable_attribute_type_list)
    attributes_type_check_str = gen_attributes_type_check_str(op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list)
    outputs_type_check_str = gen_outputs_type_check_str(op_output_type_list, op_output_optional_list)
    return OP_VERIFY_TEMPLATE.format(op_name=op_class_name, inputs_size=len(op_input_type_list) + len(op_mutable_attribute_type_list), inputs_type_check=inputs_type_check_str, attributes_check=attributes_type_check_str, outputs_size=len(op_output_type_list), outputs_type_check=outputs_type_check_str)