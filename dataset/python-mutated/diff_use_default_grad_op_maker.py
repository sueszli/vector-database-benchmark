import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
from paddle import base

def generate_spec(filename):
    if False:
        for i in range(10):
            print('nop')
    with open(filename, 'w') as f:
        ops = base.core._get_use_default_grad_op_desc_maker_ops()
        for op in ops:
            f.write(op + '\n')

def read_spec(filename):
    if False:
        return 10
    with open(filename, 'r') as f:
        return {line.strip() for line in f.readlines()}

def get_spec_diff(dev_filename, pr_filename):
    if False:
        return 10
    ops_dev = read_spec(dev_filename)
    ops_pr = read_spec(pr_filename)
    added_ops = []
    removed_ops = []
    for op in ops_pr:
        if op not in ops_dev:
            added_ops.append(op)
        else:
            removed_ops.append(op)
    return added_ops
if len(sys.argv) == 2:
    generate_spec(sys.argv[1])
elif len(sys.argv) == 3:
    added_ops = get_spec_diff(sys.argv[1], sys.argv[2])
    if added_ops:
        print(added_ops)
else:
    print('Usage 1: python diff_use_default_grad_op_maker.py [filepath] to generate new spec file\nUsage 2: python diff_use_default_grad_op_maker.py [dev_filepath] [pr_filepath] to diff spec file')
    sys.exit(1)