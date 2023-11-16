"""
Runs all the scripts in the e2e/scripts folder in "bare" mode - that is,
using `python [script]` as opposed to `streamlit run [script]`.

If any script exits with a non-zero status, this will also exit
with a non-zero status.
"""
import os
import subprocess
import sys
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from typing import Set
import click
E2E_DIR = 'e2e/scripts'
EXCLUDED_FILENAMES: Set[str] = set()
EXCLUDED_FILENAMES.add('st_experimental_rerun.py')
os.environ['MPLBACKEND'] = 'Agg'

def _command_to_string(command):
    if False:
        print('Hello World!')
    if isinstance(command, list):
        return ' '.join(command)
    else:
        return command

def _get_filenames(dir):
    if False:
        print('Hello World!')
    dir = os.path.abspath(dir)
    return [os.path.join(dir, filename) for filename in sorted(os.listdir(dir)) if filename.endswith('.py') and filename not in EXCLUDED_FILENAMES]

def run_commands(section_header, commands):
    if False:
        while True:
            i = 10
    'Run a list of commands, displaying them within the given section.'
    pool = ThreadPool(processes=4)
    lock = Lock()
    failed_commands = []

    def process_command(arg):
        if False:
            return 10
        (i, command) = arg
        vars = {'section_header': section_header, 'total': len(commands), 'command': _command_to_string(command), 'v': i + 1}
        click.secho('\nRunning %(section_header)s %(v)s/%(total)s : %(command)s' % vars, bold=True)
        result = subprocess.call(command.split(' '), stdout=subprocess.DEVNULL, stderr=None)
        if result != 0:
            with lock:
                failed_commands.append(command)
    pool.map(process_command, enumerate(commands))
    return failed_commands

def main():
    if False:
        return 10
    filenames = _get_filenames(E2E_DIR)
    commands = ['python %s' % filename for filename in filenames]
    failed = run_commands('bare scripts', commands)
    if len(failed) == 0:
        click.secho('All scripts succeeded!', fg='green', bold=True)
        sys.exit(0)
    else:
        click.secho('\n'.join((_command_to_string(command) for command in failed)), fg='red')
        click.secho('\n%s failed scripts' % len(failed), fg='red', bold=True)
        sys.exit(-1)
if __name__ == '__main__':
    main()