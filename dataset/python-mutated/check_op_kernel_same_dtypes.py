"""
Print all registered kernels of a python module in alphabet order.

Usage:
    python check_op_kernel_same_dtypes.py > all_kernels.txt
    python check_op_kernel_same_dtypes.py OP_KERNEL_DTYPE_DEV.spec OP_KERNEL_DTYPE_PR.spec > is_valid
"""
import collections
import re
import sys
import paddle

def get_all_kernels():
    if False:
        while True:
            i = 10
    all_kernels_info = paddle.framework.core._get_all_register_op_kernels()
    print(all_kernels_info)
    op_kernel_types = collections.defaultdict(list)
    for (op_type, op_infos) in all_kernels_info.items():
        pattern = re.compile('data_type\\[([^\\]]+)\\]')
        for op_info in op_infos:
            infos = pattern.findall(op_info)
            if infos is None or len(infos) == 0:
                continue
            register_type = infos[0].split(':')[-1]
            op_kernel_types[op_type].append(register_type.lower())
    for (op_type, op_kernels) in sorted(op_kernel_types.items(), key=lambda x: x[0]):
        print(op_type, ' '.join(sorted(op_kernels)))

def read_file(file_path):
    if False:
        return 10
    with open(file_path, 'r') as f:
        content = f.read()
        content = content.splitlines()
    return content

def print_diff(op_type, op_kernel_dtype_set, grad_op_kernel_dtype_set):
    if False:
        while True:
            i = 10
    if len(op_kernel_dtype_set) > len(grad_op_kernel_dtype_set):
        lack_dtypes = list(op_kernel_dtype_set - grad_op_kernel_dtype_set)
        print('{} supports [{}] now, but its grad op kernel not supported.'.format(op_type, ' '.join(lack_dtypes)))
    else:
        lack_dtypes = list(grad_op_kernel_dtype_set - op_kernel_dtype_set)
        print('{} supports [{}] now, but its forward op kernel not supported.'.format(op_type + '_grad', ' '.join(lack_dtypes)))

def contain_current_op(op_type, op_info_dict):
    if False:
        print('Hello World!')
    if not op_type.endswith('_grad'):
        return op_type + '_grad' in op_info_dict
    else:
        return op_type.rstrip('_grad') in op_info_dict

def check_change_or_add_op_kernel_dtypes_valid():
    if False:
        for i in range(10):
            print('nop')
    origin = read_file(sys.argv[1])
    new = read_file(sys.argv[2])
    origin_all_kernel_dtype_dict = {}
    for op_msg in origin:
        op_info = op_msg.split()
        origin_all_kernel_dtype_dict[op_info[0]] = set(op_info[1:])
    new_all_kernel_dtype_dict = {}
    for op_msg in new:
        op_info = op_msg.split()
        new_all_kernel_dtype_dict[op_info[0]] = set(op_info[1:])
    added_or_changed_op_info = {}
    for (op_type, dtype_set) in new_all_kernel_dtype_dict.items():
        if op_type in origin_all_kernel_dtype_dict:
            origin_dtype_set = origin_all_kernel_dtype_dict[op_type]
            if origin_dtype_set != dtype_set and (not contain_current_op(op_type, added_or_changed_op_info)):
                added_or_changed_op_info[op_type] = dtype_set
            else:
                continue
        elif not contain_current_op(op_type, added_or_changed_op_info):
            added_or_changed_op_info[op_type] = dtype_set
        else:
            continue
    for (op_type, dtype_set) in added_or_changed_op_info.items():
        if not op_type.endswith('_grad'):
            grad_op_type = op_type + '_grad'
            if grad_op_type in new_all_kernel_dtype_dict:
                grad_op_kernel_dtype_set = set(new_all_kernel_dtype_dict[grad_op_type])
                if dtype_set != grad_op_kernel_dtype_set:
                    print_diff(op_type, dtype_set, grad_op_kernel_dtype_set)
        else:
            forward_op_type = op_type.rstrip('_grad')
            if forward_op_type in new_all_kernel_dtype_dict:
                op_kernel_dtype_set = set(new_all_kernel_dtype_dict[forward_op_type])
                if op_kernel_dtype_set != dtype_set:
                    print_diff(forward_op_type, op_kernel_dtype_set, dtype_set)
if len(sys.argv) == 1:
    get_all_kernels()
elif len(sys.argv) == 3:
    check_change_or_add_op_kernel_dtypes_valid()
else:
    print('Usage:\n\tpython check_op_kernel_same_dtypes.py > all_kernels.txt\n\tpython check_op_kernel_same_dtypes.py OP_KERNEL_DTYPE_DEV.spec OP_KERNEL_DTYPE_PR.spec > diff')