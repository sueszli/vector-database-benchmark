"""Various utilities common to IPython release and maintenance tools.
"""
import os
import sys
from pathlib import Path
cd = os.chdir
archive_user = 'ipython@archive.ipython.org'
archive_dir = 'archive.ipython.org'
archive = '%s:%s' % (archive_user, archive_dir)
build_command = '{python} -m build'.format(python=sys.executable)

def sh(cmd):
    if False:
        for i in range(10):
            print('nop')
    'Run system command in shell, raise SystemExit if it returns an error.'
    print('$', cmd)
    stat = os.system(cmd)
    if stat:
        raise SystemExit('Command %s failed with code: %s' % (cmd, stat))

def get_ipdir():
    if False:
        while True:
            i = 10
    "Get IPython directory from command line, or assume it's the one above."
    ipdir = Path(__file__).parent / os.pardir
    ipdir = ipdir.resolve()
    cd(ipdir)
    if not Path('IPython').is_dir() and Path('setup.py').is_file():
        raise SystemExit('Invalid ipython directory: %s' % ipdir)
    return ipdir

def execfile(fname, globs, locs=None):
    if False:
        print('Hello World!')
    locs = locs or globs
    exec(compile(open(fname, encoding='utf-8').read(), fname, 'exec'), globs, locs)