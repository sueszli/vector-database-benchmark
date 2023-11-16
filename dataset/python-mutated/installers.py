"""Module for obtaining various versions of Python.

Currently this is a thin shim around pyenv, but it would be nice to have
this work on Windows as well by using Anaconda (as our build already
does).
"""
import os
import shutil
import subprocess
from hypothesistooling import scripts
from hypothesistooling.junkdrawer import once
HOME = os.environ['HOME']

def __python_executable(version):
    if False:
        print('Hello World!')
    return os.path.join(scripts.SNAKEPIT, version, 'bin', 'python')

def python_executable(version):
    if False:
        for i in range(10):
            print('nop')
    ensure_python(version)
    return __python_executable(version)
PYTHONS = set()

def ensure_python(version):
    if False:
        return 10
    if version in PYTHONS:
        return
    scripts.run_script('ensure-python.sh', version)
    target = __python_executable(version)
    assert os.path.exists(target), target
    PYTHONS.add(version)
STACK = os.path.join(HOME, '.local', 'bin', 'stack')
GHC = os.path.join(HOME, '.local', 'bin', 'ghc')
SHELLCHECK = shutil.which('shellcheck') or os.path.join(HOME, '.local', 'bin', 'shellcheck')

def ensure_stack():
    if False:
        i = 10
        return i + 15
    if os.path.exists(STACK):
        return
    subprocess.check_call('mkdir -p ~/.local/bin', shell=True)
    subprocess.check_call("curl -L https://www.stackage.org/stack/linux-x86_64 | tar xz --wildcards --strip-components=1 -C $HOME/.local/bin '*/stack'", shell=True)

@once
def update_stack():
    if False:
        for i in range(10):
            print('nop')
    ensure_stack()
    subprocess.check_call([STACK, 'update'])

@once
def ensure_ghc():
    if False:
        print('Hello World!')
    if os.path.exists(GHC):
        return
    update_stack()
    subprocess.check_call([STACK, 'setup'])

@once
def ensure_shellcheck():
    if False:
        i = 10
        return i + 15
    if os.path.exists(SHELLCHECK):
        return
    update_stack()
    ensure_ghc()
    subprocess.check_call([STACK, 'install', 'ShellCheck'])

@once
def ensure_rustup():
    if False:
        print('Hello World!')
    scripts.run_script('ensure-rustup.sh')