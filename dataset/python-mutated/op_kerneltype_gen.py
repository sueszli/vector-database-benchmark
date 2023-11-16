OP_GET_KERNEL_TYPE_FOR_VAR_TEMPLATE = '\nphi::DataType {op_name}::GetKernelTypeForVar(\n    const std::string& var_name,\n    const phi::DataType& tensor_dtype,\n    const phi::DataType& expected_kernel_dtype) {{\n  VLOG(4) << "Get KernelType for Var of op: {op_name}";\n  {data_transform_check}{complex_promote_check}\n  return expected_kernel_dtype;\n}}\n'
OP_DATA_TRANSFORM_CHECK_TEMPLATE = '\n{skip_trans}{support_trans}\n'
OP_SKIP_TRANSFORM_CHECK_TEMPLATE = '\n  // deal skip data transform\n  if ({skip_transform_check}){{\n    return expected_kernel_dtype;\n  }}\n'
OP_SUPPORT_TRANSFORM_CHECK_TEMPLATE = '\n  // deal support data transform\n  VLOG(8) << "SUPPORT_TRANSFORM: " << "{support_dtype_name};";\n  return tensor_dtype;\n'
OP_COMPLEX_PROMOTE_CHECK_TEMPLATE = '\n  // deal complex_promote\n  if (framework::IsComplexType(expected_kernel_dtype)) {{\n    // only promote inputs’s types when contains complex input\n    return tensor_dtype;\n  }}\n'

def get_data_transform_check_str(op_data_transform_map):
    if False:
        for i in range(10):
            print('nop')
    skip_trans_str = ''
    support_trans_str = ''
    if op_data_transform_map is not None:
        args = None
        if 'skip_transform' in op_data_transform_map:
            args = op_data_transform_map['skip_transform']
            if args is not None:
                if_cond_args = []
                for skip_arg in args:
                    if_cond_args.append('var_name == "' + skip_arg + '"')
                skip_trans_str = OP_SKIP_TRANSFORM_CHECK_TEMPLATE.format(skip_transform_check=' || '.join(if_cond_args))
        if 'support_trans_dtype' in op_data_transform_map:
            args = op_data_transform_map['support_trans_dtype']
            if args is not None:
                support_trans_str = OP_SUPPORT_TRANSFORM_CHECK_TEMPLATE.format(support_dtype_name=args)
    return OP_DATA_TRANSFORM_CHECK_TEMPLATE.format(skip_trans=skip_trans_str, support_trans=support_trans_str)

def get_complex_promote_check_str(op_compat_item):
    if False:
        for i in range(10):
            print('nop')
    complex_promote_check_str = ''
    if op_compat_item is not None and 'complex_promote' in op_compat_item and (op_compat_item['complex_promote'] is not None):
        complex_promote_check_str = OP_COMPLEX_PROMOTE_CHECK_TEMPLATE
    return complex_promote_check_str

def gen_kernel_type_for_var_str(op_class_name, op_data_transform_map, op_kernel_map, op_compat_item):
    if False:
        while True:
            i = 10
    complex_promote_check_str = get_complex_promote_check_str(op_compat_item)
    data_transform_check_str = get_data_transform_check_str(op_data_transform_map)
    return OP_GET_KERNEL_TYPE_FOR_VAR_TEMPLATE.format(op_name=op_class_name, data_transform_check=data_transform_check_str, complex_promote_check=complex_promote_check_str)