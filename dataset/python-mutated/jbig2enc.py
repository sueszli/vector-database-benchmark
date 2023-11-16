"""Interface to jbig2 executable."""
from __future__ import annotations
from subprocess import PIPE
from packaging.version import Version
from ocrmypdf.exceptions import MissingDependencyError
from ocrmypdf.subprocess import get_version, run

def version() -> Version:
    if False:
        for i in range(10):
            print('nop')
    return Version(get_version('jbig2', regex='jbig2enc (\\d+(\\.\\d+)*).*'))

def available():
    if False:
        while True:
            i = 10
    try:
        version()
    except MissingDependencyError:
        return False
    return True

def convert_group(cwd, infiles, out_prefix, threshold):
    if False:
        return 10
    args = ['jbig2', '-b', out_prefix, '--symbol-mode', '-t', str(threshold), '--pdf']
    args.extend(infiles)
    proc = run(args, cwd=cwd, stdout=PIPE, stderr=PIPE)
    proc.check_returncode()
    return proc

def convert_single(cwd, infile, outfile, threshold):
    if False:
        for i in range(10):
            print('nop')
    args = ['jbig2', '--pdf', '-t', str(threshold), infile]
    with open(outfile, 'wb') as fstdout:
        proc = run(args, cwd=cwd, stdout=fstdout, stderr=PIPE)
    proc.check_returncode()
    return proc