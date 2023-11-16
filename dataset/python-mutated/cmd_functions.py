from __future__ import annotations
import os
import select
import shlex
import subprocess
import sys
from ansible.module_utils.common.text.converters import to_bytes

def run_cmd(cmd, live=False, readsize=10):
    if False:
        for i in range(10):
            print('nop')
    cmdargs = shlex.split(cmd)
    cmdargs = [to_bytes(a, errors='surrogate_or_strict') for a in cmdargs]
    p = subprocess.Popen(cmdargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = b''
    stderr = b''
    rpipes = [p.stdout, p.stderr]
    while True:
        (rfd, wfd, efd) = select.select(rpipes, [], rpipes, 1)
        if p.stdout in rfd:
            dat = os.read(p.stdout.fileno(), readsize)
            if live:
                sys.stdout.buffer.write(dat)
            stdout += dat
            if dat == b'':
                rpipes.remove(p.stdout)
        if p.stderr in rfd:
            dat = os.read(p.stderr.fileno(), readsize)
            stderr += dat
            if live:
                sys.stdout.buffer.write(dat)
            if dat == b'':
                rpipes.remove(p.stderr)
        if (not rpipes or not rfd) and p.poll() is not None:
            break
        elif not rpipes and p.poll() is None:
            p.wait()
    return (p.returncode, stdout, stderr)