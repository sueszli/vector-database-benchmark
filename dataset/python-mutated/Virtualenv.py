""" Creating virtualenvs and running commands in them.

"""
import os
import sys
from contextlib import contextmanager
from nuitka.__past__ import unicode
from nuitka.Tracing import my_print
from nuitka.utils.Execution import check_call, executeProcess
from nuitka.utils.FileOperations import getDirectoryRealPath, removeDirectory, withDirectoryChange

class Virtualenv(object):

    def __init__(self, env_dir):
        if False:
            for i in range(10):
                print('nop')
        self.env_dir = os.path.abspath(env_dir)

    def runCommand(self, commands, style=None):
        if False:
            for i in range(10):
                print('nop')
        if type(commands) in (str, unicode):
            commands = [commands]
        with withDirectoryChange(self.env_dir):
            if os.name == 'nt':
                commands = ['call scripts\\activate.bat'] + commands
            else:
                commands = ['. bin/activate'] + commands
            command = ' && '.join(commands)
            if style is not None:
                my_print('Executing: %s' % command, style=style)
            assert os.system(command) == 0, command

    def runCommandWithOutput(self, commands, style=None):
        if False:
            return 10
        '\n        Returns the stdout,stderr,exit_code from running command\n        '
        if type(commands) in (str, unicode):
            commands = [commands]
        with withDirectoryChange(self.env_dir):
            if os.name == 'nt':
                commands = ['call scripts\\activate.bat'] + commands
            else:
                commands = ['. bin/activate'] + commands
            command = ' && '.join(commands)
            if style is not None:
                my_print('Executing: %s' % command, style=style)
            return executeProcess(command=command, shell=True)

    def getVirtualenvDir(self):
        if False:
            i = 10
            return i + 15
        return self.env_dir

@contextmanager
def withVirtualenv(env_name, base_dir=None, python=None, delete=True, style=None):
    if False:
        i = 10
        return i + 15
    'Create a virtualenv and change into it.\n\n    Activating for actual use will be your task.\n    '
    if style is not None:
        my_print('Creating a virtualenv:')
    if python is None:
        python = sys.executable
    python = os.path.join(getDirectoryRealPath(os.path.dirname(python)), os.path.basename(python))
    if base_dir is not None:
        env_dir = os.path.join(base_dir, env_name)
    else:
        env_dir = env_name
    removeDirectory(env_dir, ignore_errors=False)
    with withDirectoryChange(base_dir, allow_none=True):
        command = [python, '-m', 'virtualenv', env_name]
        if style is not None:
            my_print('Executing: %s' % ' '.join(command), style=style)
        check_call(command)
        yield Virtualenv(env_dir)
    if delete:
        removeDirectory(env_dir, ignore_errors=False)