from __future__ import annotations
import sys
from unittest.mock import MagicMock
module_name = 'pwndbg.commands'
module = MagicMock(__name__=module_name, load_commands=lambda : None)
sys.modules[module_name] = module
import os
import tempfile
import mocks.gdb
import mocks.gdblib
from pwndbg.lib.which import which

def test_basic():
    if False:
        while True:
            i = 10
    assert which('ls') == '/bin/ls'

def test_nonexistent():
    if False:
        for i in range(10):
            print('nop')
    assert which('definitely-not-a-real-command') is None

def test_dir():
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'test_file')
        with open(path, 'w') as f:
            f.write('test')
        os.chmod(path, 493)
        assert which(path) == path