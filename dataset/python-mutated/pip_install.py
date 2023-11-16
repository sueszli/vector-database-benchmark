from __future__ import absolute_import
from __future__ import print_function
import os
import subprocess
import sys

def find_tools_path():
    if False:
        print('Hello World!')
    return os.path.dirname(os.path.realpath(__file__))

def call_with_print(command, env):
    if False:
        print('Hello World!')
    assert env is not None
    print(command)
    subprocess.check_call(command, shell=True, env=env)

def pip_install_with_print(args_str, env):
    if False:
        while True:
            i = 10
    command = ['"', sys.executable, '" -m pip install --disable-pip-version-check ', args_str]
    call_with_print(''.join(command), env=env)

def pip_constrained_environ():
    if False:
        while True:
            i = 10
    tools_path = find_tools_path()
    repo_path = os.path.dirname(tools_path)
    if os.environ.get('CERTBOT_OLDEST') == '1':
        constraints_path = os.path.normpath(os.path.join(repo_path, 'tools', 'oldest_constraints.txt'))
    else:
        constraints_path = os.path.normpath(os.path.join(repo_path, 'tools', 'requirements.txt'))
    env = os.environ.copy()
    env['PIP_CONSTRAINT'] = constraints_path
    return env

def pipstrap(env=None):
    if False:
        i = 10
        return i + 15
    if env is None:
        env = pip_constrained_environ()
    pip_install_with_print('pip setuptools wheel', env=env)

def main(args):
    if False:
        i = 10
        return i + 15
    env = pip_constrained_environ()
    pipstrap(env)
    pip_install_with_print(' '.join(args), env=env)
if __name__ == '__main__':
    main(sys.argv[1:])