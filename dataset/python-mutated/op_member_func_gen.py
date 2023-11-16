OP_GET_INPUT_TEMPLATE = '  pir::Value {input_name}() {{ return operand_source({input_index}); }}\n'
OP_GET_OUTPUT_TEMPLATE = '  pir::OpResult {output_name}() {{ return result({output_index}); }}\n'

def gen_op_get_inputs_outputs_str(op_input_name_list, op_mutable_attribute_name_list, op_output_name_list):
    if False:
        return 10
    op_get_inputs_outputs_str = ''
    for idx in range(len(op_input_name_list)):
        op_get_inputs_outputs_str += OP_GET_INPUT_TEMPLATE.format(input_name=op_input_name_list[idx], input_index=idx)
    for idx in range(len(op_mutable_attribute_name_list)):
        op_get_inputs_outputs_str += OP_GET_INPUT_TEMPLATE.format(input_name=op_mutable_attribute_name_list[idx], input_index=idx + len(op_input_name_list))
    for idx in range(len(op_output_name_list)):
        op_get_inputs_outputs_str += OP_GET_OUTPUT_TEMPLATE.format(output_name=op_output_name_list[idx], output_index=idx)
    return op_get_inputs_outputs_str