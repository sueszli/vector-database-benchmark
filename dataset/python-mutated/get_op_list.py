import argparse
import os
import re
import numpy as np
import paddle
from paddle.inference import _get_phi_kernel_name
paddle.enable_static()

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='', help='Directory of the inference models that named with pdmodel.')
    parser.add_argument('--op_list', type=str, default='', help='List of ops like "conv2d;pool2d;relu".')
    return parser.parse_args()

def get_model_ops(model_file, ops_set):
    if False:
        print('Hello World!')
    model_bytes = paddle.static.load_from_file(model_file)
    pg = paddle.static.deserialize_program(model_bytes)
    for i in range(0, pg.desc.num_blocks()):
        block = pg.desc.block(i)
        size = block.op_size()
        for j in range(0, size):
            ops_set.add(block.op(j).type())

def get_model_phi_kernels(ops_set):
    if False:
        i = 10
        return i + 15
    phi_set = set()
    phi_raw_list = ['add', 'subtract', 'multiply', 'multiply_sr', 'divide', 'maximum', 'minimum', 'remainder', 'floor_divide', 'elementwise_pow']
    phi_odd_dist = {'batch_norm': 'batch_norm_infer'}
    for op in ops_set:
        print(op)
        phi_kernel = _get_phi_kernel_name(op)
        print(phi_kernel)
        phi_set.add(phi_kernel)
        if phi_kernel in phi_raw_list:
            phi_set.add(phi_kernel + '_raw')
        if phi_kernel in phi_odd_dist.keys():
            phi_set.add(phi_odd_dist[phi_kernel])
    return phi_set
if __name__ == '__main__':
    args = parse_args()
    ops_set = set()
    if args.op_list != '':
        op_list = args.op_list.split(';')
        for op in op_list:
            ops_set.add(op)
    if args.model_dir != '':
        for (root, dirs, files) in os.walk(args.model_dir, topdown=True):
            for name in files:
                if re.match('.*pdmodel', name):
                    get_model_ops(os.path.join(root, name), ops_set)
    phi_set = get_model_phi_kernels(ops_set)
    ops = ';'.join(ops_set)
    kernels = ';'.join(phi_set)
    print('op_list: ', ops)
    print('kernel_list: ', kernels)
    ops = np.array([ops])
    kernels = np.array([kernels])
    np.savetxt('op_list.txt', ops, fmt='%s')
    np.savetxt('kernel_list.txt', kernels, fmt='%s')