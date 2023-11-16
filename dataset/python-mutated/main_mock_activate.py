"""Mock CLI implementation for `conda activate`.

A mock implementation of the activate shell command for better UX.
"""
from argparse import SUPPRESS
from .. import CondaError

def configure_parser(sub_parsers):
    if False:
        return 10
    p = sub_parsers.add_parser('activate', help='Activate a conda environment.')
    p.set_defaults(func='conda.cli.main_mock_activate.execute')
    p.add_argument('args', action='store', nargs='*', help=SUPPRESS)

def execute(args, parser):
    if False:
        return 10
    raise CondaError("Run 'conda init' before 'conda activate'")