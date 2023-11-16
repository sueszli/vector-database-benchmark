"""
Test specific for the --no-color option
"""
import os
import shutil
import subprocess
import sys
import pytest
from tests.lib import PipTestEnvironment

@pytest.mark.skipif(shutil.which('script') is None, reason="no 'script' executable")
def test_no_color(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Ensure colour output disabled when --no-color is passed.'
    pip_command = 'pip uninstall {} noSuchPackage'
    if sys.platform == 'darwin':
        command = f'script -q /tmp/pip-test-no-color.txt {pip_command}'
    else:
        command = f'script -q /tmp/pip-test-no-color.txt --command "{pip_command}"'

    def get_run_output(option: str='') -> str:
        if False:
            return 10
        cmd = command.format(option)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.communicate()
        try:
            with open('/tmp/pip-test-no-color.txt') as output_file:
                retval = output_file.read()
            return retval
        finally:
            os.unlink('/tmp/pip-test-no-color.txt')
    assert '\x1b[3' in get_run_output(''), 'Expected color in output'
    assert '\x1b[3' not in get_run_output('--no-color'), 'Expected no color in output'