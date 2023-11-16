import os
import subprocess
import sys
from typing import List

def run_cmd(cmd: List[str]) -> None:
    if False:
        print('Hello World!')
    print(f'Running: {cmd}')
    result = subprocess.run(cmd, capture_output=True)
    (stdout, stderr) = (result.stdout.decode('utf-8').strip(), result.stderr.decode('utf-8').strip())
    print(stdout)
    print(stderr)
    if result.returncode != 0:
        print(f'Failed to run {cmd}')
        sys.exit(1)

def update_submodules() -> None:
    if False:
        print('Hello World!')
    run_cmd(['git', 'submodule', 'update', '--init', '--recursive'])

def gen_compile_commands() -> None:
    if False:
        i = 10
        return i + 15
    os.environ['USE_NCCL'] = '0'
    os.environ['CC'] = 'clang'
    os.environ['CXX'] = 'clang++'
    run_cmd([sys.executable, 'setup.py', '--cmake-only', 'build'])

def run_autogen() -> None:
    if False:
        return 10
    run_cmd([sys.executable, '-m', 'torchgen.gen', '-s', 'aten/src/ATen', '-d', 'build/aten/src/ATen', '--per-operator-headers'])
    run_cmd([sys.executable, 'tools/setup_helpers/generate_code.py', '--native-functions-path', 'aten/src/ATen/native/native_functions.yaml', '--tags-path', 'aten/src/ATen/native/tags.yaml', '--gen-lazy-ts-backend'])

def generate_build_files() -> None:
    if False:
        for i in range(10):
            print('nop')
    update_submodules()
    gen_compile_commands()
    run_autogen()
if __name__ == '__main__':
    generate_build_files()