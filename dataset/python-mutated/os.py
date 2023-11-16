"""Xonsh extension of the standard library os module, using xonsh for
subprocess calls"""
import sys
from xonsh.built_ins import subproc_uncaptured
from xonsh.dirstack import with_pushd
indir = with_pushd
'alias to push_d context manager'

def rmtree(dirname, force=False):
    if False:
        print('Hello World!')
    'Remove a directory, even if it has read-only files (Windows).\n    Git creates read-only files that must be removed on teardown. See\n    https://stackoverflow.com/questions/2656322  for more info.\n\n    Parameters\n    ----------\n    dirname : str\n        Directory to be removed\n    force : bool\n        If True force removal, defaults to False\n    '
    if sys.platform == 'win32':
        cmd_args = '/S/Q'
        subproc_uncaptured(['rmdir', cmd_args, dirname])
    else:
        cmd_args = '-r'
        if force:
            cmd_args += 'f'
        subproc_uncaptured(['rm', cmd_args, dirname])