import os
import re
import sys
from typing import List
__all__ = ['check_code_for_cuda_kernel_launches', 'check_cuda_kernel_launches']
exclude_files: List[str] = []
kernel_launch_start = re.compile('^.*<<<[^>]+>>>\\s*\\(', flags=re.MULTILINE)
has_check = re.compile('\\s*;(?![^;}]*C10_CUDA_KERNEL_LAUNCH_CHECK\\(\\);)', flags=re.MULTILINE)

def find_matching_paren(s: str, startpos: int) -> int:
    if False:
        i = 10
        return i + 15
    'Given a string "prefix (unknown number of characters) suffix"\n    and the position of the first `(` returns the index of the character\n    1 past the `)`, accounting for paren nesting\n    '
    opening = 0
    for (i, c) in enumerate(s[startpos:]):
        if c == '(':
            opening += 1
        elif c == ')':
            opening -= 1
            if opening == 0:
                return startpos + i + 1
    raise IndexError('Closing parens not found!')

def should_exclude_file(filename) -> bool:
    if False:
        for i in range(10):
            print('nop')
    for exclude_suffix in exclude_files:
        if filename.endswith(exclude_suffix):
            return True
    return False

def check_code_for_cuda_kernel_launches(code, filename=None):
    if False:
        print('Hello World!')
    'Checks code for CUDA kernel launches without cuda error checks.\n\n    Args:\n        filename - Filename of file containing the code. Used only for display\n                   purposes, so you can put anything here.\n        code     - The code to check\n\n    Returns:\n        The number of unsafe kernel launches in the code\n    '
    if filename is None:
        filename = '##Python Function Call##'
    code = enumerate(code.split('\n'))
    code = [f'{lineno}: {linecode}' for (lineno, linecode) in code]
    code = '\n'.join(code)
    num_launches_without_checks = 0
    for m in kernel_launch_start.finditer(code):
        end_paren = find_matching_paren(code, m.end() - 1)
        if has_check.match(code, end_paren):
            num_launches_without_checks += 1
            context = code[m.start():end_paren + 1]
            print(f"Missing C10_CUDA_KERNEL_LAUNCH_CHECK in '{filename}'. Context:\n{context}", file=sys.stderr)
    return num_launches_without_checks

def check_file(filename):
    if False:
        while True:
            i = 10
    'Checks a file for CUDA kernel launches without cuda error checks\n\n    Args:\n        filename - File to check\n\n    Returns:\n        The number of unsafe kernel launches in the file\n    '
    if not filename.endswith(('.cu', '.cuh')):
        return 0
    if should_exclude_file(filename):
        return 0
    with open(filename) as fo:
        contents = fo.read()
        unsafeCount = check_code_for_cuda_kernel_launches(contents, filename)
    return unsafeCount

def check_cuda_kernel_launches():
    if False:
        i = 10
        return i + 15
    'Checks all pytorch code for CUDA kernel launches without cuda error checks\n\n    Returns:\n        The number of unsafe kernel launches in the codebase\n    '
    torch_dir = os.path.dirname(os.path.realpath(__file__))
    torch_dir = os.path.dirname(torch_dir)
    torch_dir = os.path.dirname(torch_dir)
    kernels_without_checks = 0
    files_without_checks = []
    for (root, dirnames, filenames) in os.walk(torch_dir):
        if root == os.path.join(torch_dir, 'build') or root == os.path.join(torch_dir, 'torch/include'):
            dirnames[:] = []
            continue
        for x in filenames:
            filename = os.path.join(root, x)
            file_result = check_file(filename)
            if file_result > 0:
                kernels_without_checks += file_result
                files_without_checks.append(filename)
    if kernels_without_checks > 0:
        count_str = f"Found {kernels_without_checks} instances in {len(files_without_checks)} files where kernel launches didn't have checks."
        print(count_str, file=sys.stderr)
        print('Files without checks:', file=sys.stderr)
        for x in files_without_checks:
            print(f'\t{x}', file=sys.stderr)
        print(count_str, file=sys.stderr)
    return kernels_without_checks
if __name__ == '__main__':
    unsafe_launches = check_cuda_kernel_launches()
    sys.exit(0 if unsafe_launches == 0 else 1)