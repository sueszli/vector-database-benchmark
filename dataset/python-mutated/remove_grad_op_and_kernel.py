"""
This script simply removes grad ops and kernels. You should use this script
when cmake ON_INFER=ON, which can greatly reduce the volume of the inference library.
"""
import argparse
import glob
import os
import re
import reduce_lib_size_util

def parse_args():
    if False:
        while True:
            i = 10
    'Parse input arguments.'
    parser = argparse.ArgumentParser(description='Remove grad op and kernels.')
    parser.add_argument('--only_kernel', action='store_true', default=False)
    parser.add_argument('--dry_run', action='store_true', default=False)
    args = parser.parse_args()
    return args

def find_type_files(cur_dir, file_type, file_list=[]):
    if False:
        i = 10
        return i + 15
    next_level_dirs = os.listdir(cur_dir)
    for next_level_name in next_level_dirs:
        next_level_dir = os.path.join(cur_dir, next_level_name)
        if os.path.isfile(next_level_dir):
            if os.path.splitext(next_level_dir)[1] == file_type:
                file_list.append(next_level_dir)
        elif os.path.isdir(next_level_dir):
            find_type_files(next_level_dir, file_type, file_list)
    return file_list

def remove_grad_op_and_kernel(content, pattern1, pattern2):
    if False:
        for i in range(10):
            print('nop')
    res = []
    first_match = re.findall(pattern1, content, flags=re.DOTALL)
    for match in first_match:
        res.extend(re.findall(pattern2, match, flags=re.DOTALL))
    return (res, len(res))

def update_operator_cmake(cmake_file):
    if False:
        return 10
    'Update operator cmake.\n    Args:\n        cmake_file (str): cmake file path.\n    '
    pat1 = 'add_subdirectory(optimizers)'
    pat2 = 'register_operators\\(EXCLUDES.*?py_func_op.*?\\)'
    code1 = 'if(ON_INFER)\nadd_subdirectory(optimizers)\nendif()'
    code2 = 'if(ON_INFER)\nfile(GLOB LOSS_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*loss_op.cc")\nstring(REPLACE ".cc" "" LOSS_OPS "${LOSS_OPS}")\nendif()'
    with open(cmake_file, 'r') as f:
        content = ''.join(f.readlines())
        content = content.replace(pat1, code1)
        match = re.findall(pat2, content, flags=re.DOTALL)
        content = content.replace(match[0], code2 + '\n' + match[0].replace('py_func_op', 'py_func_op ${LOSS_OPS}'))
    with open(cmake_file, 'w') as f:
        f.write(content)
if __name__ == '__main__':
    args = parse_args()
    tool_dir = os.path.dirname(os.path.abspath(__file__))
    all_op = glob.glob(os.path.join(tool_dir, '../paddle/fluid/operators/**/*.cc'), recursive=True)
    all_op += glob.glob(os.path.join(tool_dir, '../paddle/fluid/operators/**/*.cu'), recursive=True)
    spec_ops = ['activation_op.cc']
    (register_op_count, register_op_cpu_kernel_count, register_op_cuda_kernel_count, register_op_xpu_kernel_count) = (0, 0, 0, 0)
    (register_op_kernel_count, register_op_kernel_with_custom_type_count) = (0, 0)
    for op_file in all_op:
        op_pattern1 = 'REGISTER_OPERATOR\\(.*?\\);?'
        op_pattern2 = 'REGISTER_OPERATOR\\(.*?_grad,.*?\\);?'
        if args.only_kernel:
            op_pattern1 = 'DISABLE_REMOVE_GRAD_OP_' + op_pattern1
            op_pattern2 = 'DISABLE_REMOVE_GRAD_OP_' + op_pattern2
        cpu_kernel_pattern1 = 'REGISTER_OP_CPU_KERNEL\\(.*?\\);?|REGISTER_OP_CPU_KERNEL_FUNCTOR\\(.*?\\);?'
        cpu_kernel_pattern2 = 'REGISTER_OP_CPU_KERNEL\\(.*?_grad,.*?\\);?|REGISTER_OP_CPU_KERNEL_FUNCTOR\\(.*?_grad,.*?\\);?'
        gpu_kernel_pattern1 = 'REGISTER_OP_CUDA_KERNEL\\(.*?\\);?|REGISTER_OP_CUDA_KERNEL_FUNCTOR\\(.*?\\);?'
        gpu_kernel_pattern2 = 'REGISTER_OP_CUDA_KERNEL\\(.*?_grad,.*?\\);?|REGISTER_OP_CUDA_KERNEL_FUNCTOR\\(.*?_grad,.*?\\);?'
        xpu_kernel_pattern1 = 'REGISTER_OP_XPU_KERNEL\\(.*?\\);?'
        xpu_kernel_pattern2 = 'REGISTER_OP_XPU_KERNEL\\(.*?_grad,.*?\\);?'
        op_kernel_pattern1 = 'REGISTER_OP_KERNEL\\(.*?\\);?'
        op_kernel_pattern2 = 'REGISTER_OP_KERNEL\\(.*?_grad,.*?\\);?'
        custom_pattern1 = 'REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE\\(.*?\\);?'
        custom_pattern2 = 'REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE\\(.*?_grad,.*?\\);?'
        op_name = os.path.split(op_file)[1]
        if op_name in spec_ops:
            op_pattern1 = op_pattern1[:-1]
            op_pattern2 = op_pattern2[:-1]
            cpu_kernel_pattern1 = cpu_kernel_pattern1[:-1]
            cpu_kernel_pattern2 = cpu_kernel_pattern2[:-1]
            gpu_kernel_pattern1 = gpu_kernel_pattern1[:-1]
            gpu_kernel_pattern2 = gpu_kernel_pattern2[:-1]
            xpu_kernel_pattern1 = xpu_kernel_pattern1[:-1]
            xpu_kernel_pattern2 = xpu_kernel_pattern2[:-1]
            op_kernel_pattern1 = op_kernel_pattern1[:-1]
            op_kernel_pattern2 = op_kernel_pattern2[:-1]
            custom_pattern1 = custom_pattern1[:-1]
            custom_pattern2 = custom_pattern2[:-1]
        all_matches = []
        with open(op_file, 'r', encoding='utf-8') as f:
            content = ''.join(f.readlines())
            (op, op_count) = remove_grad_op_and_kernel(content, op_pattern1, op_pattern2)
            (cpu_kernel, cpu_kernel_count) = remove_grad_op_and_kernel(content, cpu_kernel_pattern1, cpu_kernel_pattern2)
            (gpu_kernel, gpu_kernel_count) = remove_grad_op_and_kernel(content, gpu_kernel_pattern1, gpu_kernel_pattern2)
            (xpu_kernel, xpu_kernel_count) = remove_grad_op_and_kernel(content, xpu_kernel_pattern1, xpu_kernel_pattern2)
            (op_kernel, op_kernel_count) = remove_grad_op_and_kernel(content, op_kernel_pattern1, op_kernel_pattern2)
            (custom_kernel, custom_kernel_count) = remove_grad_op_and_kernel(content, custom_pattern1, custom_pattern2)
            register_op_count += op_count
            register_op_cpu_kernel_count += cpu_kernel_count
            register_op_cuda_kernel_count += gpu_kernel_count
            register_op_xpu_kernel_count += xpu_kernel_count
            register_op_kernel_count += op_kernel_count
            register_op_kernel_with_custom_type_count += custom_kernel_count
            all_matches.extend(op)
            all_matches.extend(cpu_kernel)
            all_matches.extend(gpu_kernel)
            all_matches.extend(xpu_kernel)
            all_matches.extend(op_kernel)
            all_matches.extend(custom_kernel)
        for to_remove in all_matches:
            content = content.replace(to_remove, '')
            if args.dry_run:
                print(op_file, to_remove)
        if not args.dry_run:
            with open(op_file, 'w', encoding='utf-8') as f:
                f.write(content)
    cmake_file = os.path.join(tool_dir, '../paddle/fluid/operators/CMakeLists.txt')
    update_operator_cmake(cmake_file)
    register_pd_kernel_count = reduce_lib_size_util.remove_grad_kernels(args.dry_run)
    print('We erase all grad op and kernel for Paddle-Inference lib.')
    print('%50s%10s' % ('type', 'count'))
    print('%50s%10s' % ('REGISTER_OPERATOR', register_op_count))
    print('%50s%10s' % ('REGISTER_OP_CPU_KERNEL', register_op_cpu_kernel_count))
    print('%50s%10s' % ('REGISTER_OP_CUDA_KERNEL', register_op_cuda_kernel_count))
    print('%50s%10s' % ('REGISTER_OP_XPU_KERNEL', register_op_xpu_kernel_count))
    print('%50s%10s' % ('REGISTER_OP_KERNEL', register_op_kernel_count))
    print('%50s%10s' % ('REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE', register_op_kernel_with_custom_type_count))
    print('%50s%10s' % ('REGISTER_OP_PD_KERNEL', register_pd_kernel_count))