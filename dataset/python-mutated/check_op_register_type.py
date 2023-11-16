"""
Print all registered kernels of a python module in alphabet order.

Usage:
    python check_op_register_type.py > all_kernels.txt
    python check_op_register_type.py OP_TYPE_DEV.spec OP_TYPE_PR.spec > is_valid
"""
import collections
import difflib
import re
import sys
from paddle import base
INTS = {'int', 'int64_t'}
FLOATS = {'float', 'double'}

def get_all_kernels():
    if False:
        i = 10
        return i + 15
    all_kernels_info = base.core._get_all_register_op_kernels()
    op_kernel_types = collections.defaultdict(list)
    for (op_type, op_infos) in all_kernels_info.items():
        is_grad_op = op_type.endswith('_grad')
        if is_grad_op:
            continue
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
        print('Hello World!')
    with open(file_path, 'r') as f:
        content = f.read()
        content = content.splitlines()
    return content

def print_diff(op_type, register_types):
    if False:
        while True:
            i = 10
    lack_types = set()
    if len(INTS - register_types) == 1:
        lack_types |= INTS - register_types
    if len(FLOATS - register_types) == 1:
        lack_types |= FLOATS - register_types
    print('{} only supports [{}] now, but lacks [{}].'.format(op_type, ' '.join(register_types), ' '.join(lack_types)))

def check_add_op_valid():
    if False:
        i = 10
        return i + 15
    origin = read_file(sys.argv[1])
    new = read_file(sys.argv[2])
    differ = difflib.Differ()
    result = differ.compare(origin, new)
    for each_diff in result:
        if each_diff[0] in ['+'] and len(each_diff) > 2:
            op_info = each_diff[1:].split()
            if len(op_info) < 2:
                continue
            register_types = set(op_info[1:])
            if len(FLOATS - register_types) == 1 or len(INTS - register_types) == 1:
                print_diff(op_info[0], register_types)
if len(sys.argv) == 1:
    get_all_kernels()
elif len(sys.argv) == 3:
    check_add_op_valid()
else:
    print('Usage:\n\tpython check_op_register_type.py > all_kernels.txt\n\tpython check_op_register_type.py OP_TYPE_DEV.spec OP_TYPE_PR.spec > diff')