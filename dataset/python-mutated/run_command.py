import os
import subprocess
import sys
from typing import List
import pytest
from scripts.make_scripts import script_path_for
from tests.support.help_reformatting import reformat_help_message
from tests.support.my_path import MyPath
from trashcli import base_dir

class CmdResult:

    def __init__(self, stdout, stderr, exit_code):
        if False:
            while True:
                i = 10
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.all = [stdout, stderr, exit_code]

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.all)

    def output(self):
        if False:
            i = 10
            return i + 15
        return self._format([self.stdout, self.stderr])

    def last_line_of_stderr(self):
        if False:
            for i in range(10):
                print('nop')
        return last_line_of(self.stderr)

    def last_line_of_stdout(self):
        if False:
            return 10
        return last_line_of(self.stdout)

    def reformatted_help(self):
        if False:
            return 10
        return reformat_help_message(self.stdout)

    @staticmethod
    def _format(outs):
        if False:
            for i in range(10):
                print('nop')
        outs = [out for out in outs if out != '']
        return ''.join([out.rstrip('\n') + '\n' for out in outs])

    def clean_vol_and_grep(self, pattern, fake_vol):
        if False:
            i = 10
            return i + 15
        matching_lines = self._grep(self.stderr_lines(), pattern)
        return self._replace(fake_vol, '/vol', matching_lines)

    @staticmethod
    def _grep(lines, pattern):
        if False:
            while True:
                i = 10
        return [line for line in lines if pattern in line]

    def clean_temp_dir(self, temp_dir):
        if False:
            print('Hello World!')
        return self._replace(temp_dir, '', self.stderr_lines())

    def clean_tmp_and_grep(self, temp_dir, pattern):
        if False:
            return 10
        return self._grep(self.clean_temp_dir(temp_dir), pattern)

    def stderr_lines(self):
        if False:
            print('Hello World!')
        return self.stderr.splitlines()

    @staticmethod
    def _replace(pattern, replacement, lines):
        if False:
            print('Hello World!')
        return [line.replace(pattern, replacement) for line in lines]

def run_command(cwd, command, args=None, input='', env=None):
    if False:
        return 10
    if env is None:
        env = {}
    if args is None:
        args = []
    command_full_path = script_path_for(command)
    env['PYTHONPATH'] = base_dir
    process = subprocess.Popen([sys.executable, command_full_path] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=merge_dicts(os.environ, env))
    (stdout, stderr) = process.communicate(input=input.encode('utf-8'))
    return CmdResult(stdout.decode('utf-8'), stderr.decode('utf-8'), process.returncode)

@pytest.fixture
def temp_dir():
    if False:
        i = 10
        return i + 15
    temp_dir = MyPath.make_temp_dir()
    yield temp_dir
    temp_dir.clean_up()

def merge_dicts(x, y):
    if False:
        i = 10
        return i + 15
    z = x.copy()
    z.update(y)
    return z

def last_line_of(stdout):
    if False:
        i = 10
        return i + 15
    if len(stdout.splitlines()) > 0:
        return stdout.splitlines()[-1]
    else:
        return ''

def first_line_of(out):
    if False:
        for i in range(10):
            print('nop')
    return out.splitlines()[0]