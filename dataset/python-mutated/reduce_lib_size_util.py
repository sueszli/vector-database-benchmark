"""
This script simply removes grad kernels. You should use this script
when cmake ON_INFER=ON, which can greatly reduce the volume of the inference library.
"""
import glob
import os

def is_balanced(content):
    if False:
        return 10
    '\n    Check whether sequence contains valid parenthesis.\n    Args:\n       content (str): content of string.\n\n    Returns:\n        boolean: True if sequence contains valid parenthesis.\n    '
    if content.find('{') == -1:
        return False
    stack = []
    (push_chars, pop_chars) = ('({', ')}')
    for c in content:
        if c in push_chars:
            stack.append(c)
        elif c in pop_chars:
            if not len(stack):
                return False
            else:
                stack_top = stack.pop()
                balancing_bracket = push_chars[pop_chars.index(c)]
                if stack_top != balancing_bracket:
                    return False
    return not stack

def grad_kernel_definition(content, kernel_pattern, grad_pattern):
    if False:
        i = 10
        return i + 15
    '\n    Args:\n       content(str): file content\n       kernel_pattern(str): kernel pattern\n       grad_pattern(str): grad pattern\n\n    Returns:\n        (list, int): grad kernel definitions in file and count.\n    '
    results = []
    count = 0
    start = 0
    lens = len(content)
    while True:
        index = content.find(kernel_pattern, start)
        if index == -1:
            return (results, count)
        i = index + 1
        while i <= lens:
            check_str = content[index:i]
            if is_balanced(check_str):
                if check_str.find(grad_pattern) != -1:
                    results.append(check_str)
                    count += 1
                start = i
                break
            i += 1
        else:
            return (results, count)

def remove_grad_kernels(dry_run=False):
    if False:
        while True:
            i = 10
    '\n    Args:\n       dry_run(bool): whether just print\n\n    Returns:\n        int: number of kernel(grad) removed\n    '
    pd_kernel_pattern = 'PD_REGISTER_STRUCT_KERNEL'
    register_op_pd_kernel_count = 0
    matches = []
    tool_dir = os.path.dirname(os.path.abspath(__file__))
    all_op = glob.glob(os.path.join(tool_dir, '../paddle/fluid/operators/**/*.cc'), recursive=True)
    all_op += glob.glob(os.path.join(tool_dir, '../paddle/fluid/operators/**/*.cu'), recursive=True)
    for op_file in all_op:
        with open(op_file, 'r', encoding='utf-8') as f:
            content = ''.join(f.readlines())
            (pd_kernel, pd_kernel_count) = grad_kernel_definition(content, pd_kernel_pattern, '_grad,')
            register_op_pd_kernel_count += pd_kernel_count
            matches.extend(pd_kernel)
        for to_remove in matches:
            content = content.replace(to_remove, '')
            if dry_run:
                print(op_file, to_remove)
        if not dry_run:
            with open(op_file, 'w', encoding='utf-8') as f:
                f.write(content)
    return register_op_pd_kernel_count