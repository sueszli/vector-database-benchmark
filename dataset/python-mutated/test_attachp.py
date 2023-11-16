from __future__ import annotations
import os
import re
import subprocess
import tempfile
import pytest
from .utils import run_gdb_with_script
can_attach = False
if os.getuid() == 0:
    can_attach = True
else:
    with open('/proc/sys/kernel/yama/ptrace_scope') as f:
        result = f.read()
        if len(result) >= 1 and result[0] == '0':
            can_attach = True
REASON_CANNOT_ATTACH = 'Test skipped due to inability to attach (needs sudo or sysctl -w kernel.yama.ptrace_scope=0'

@pytest.fixture
def launched_bash_binary():
    if False:
        print('Hello World!')
    path = tempfile.mktemp()
    subprocess.check_output(['cp', '/bin/bash', path])
    process = subprocess.Popen([path], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    yield (process.pid, path)
    process.kill()
    os.remove(path)

@pytest.mark.skipif(can_attach is False, reason=REASON_CANNOT_ATTACH)
def test_attachp_command_attaches_to_procname(launched_bash_binary):
    if False:
        print('Hello World!')
    (pid, binary_path) = launched_bash_binary
    binary_name = binary_path.split('/')[-1]
    result = run_gdb_with_script(pyafter=f'attachp {binary_name}')
    matches = re.search('Attaching to ([0-9]+)', result).groups()
    assert matches == (str(pid),)
    assert re.search(f'Detaching from program: {binary_path}, process {pid}', result)

@pytest.mark.skipif(can_attach is False, reason=REASON_CANNOT_ATTACH)
def test_attachp_command_attaches_to_pid(launched_bash_binary):
    if False:
        i = 10
        return i + 15
    (pid, binary_path) = launched_bash_binary
    result = run_gdb_with_script(pyafter=f'attachp {pid}')
    matches = re.search('Attaching to ([0-9]+)', result).groups()
    assert matches == (str(pid),)
    assert re.search(f'Detaching from program: {binary_path}, process {pid}', result)

@pytest.mark.skipif(can_attach is False, reason=REASON_CANNOT_ATTACH)
def test_attachp_command_attaches_to_procname_too_many_pids(launched_bash_binary):
    if False:
        for i in range(10):
            print('nop')
    (pid, binary_path) = launched_bash_binary
    process = subprocess.Popen([binary_path], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    binary_name = binary_path.split('/')[-1]
    result = run_gdb_with_script(pyafter=f'attachp {binary_name}')
    process.kill()
    matches = re.search('Found pids: ([0-9]+), ([0-9]+) \\(use `attach <pid>`\\)', result).groups()
    matches = list(map(int, matches))
    matches.sort()
    expected_pids = [pid, process.pid]
    expected_pids.sort()
    assert matches == expected_pids

@pytest.mark.skipif(can_attach is False, reason=REASON_CANNOT_ATTACH)
def test_attachp_command_nonexistent_procname():
    if False:
        i = 10
        return i + 15
    result = run_gdb_with_script(pyafter='attachp some-nonexistent-process-name')
    assert 'Process some-nonexistent-process-name not found' in result

def test_attachp_command_no_pids():
    if False:
        while True:
            i = 10
    try:
        result = run_gdb_with_script(pyafter='attachp 99999999', timeout=5)
    except subprocess.TimeoutExpired:
        return
    assert 'Error: ptrace: No such process.' in result