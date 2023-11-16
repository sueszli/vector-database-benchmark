"""Mock CLI implementation for `conda deactivate`.

A mock implementation of the deactivate shell command for better UX.
"""
from .. import CondaError

def configure_parser(sub_parsers):
    if False:
        for i in range(10):
            print('nop')
    p = sub_parsers.add_parser('deactivate', help='Deactivate the current active conda environment.')
    p.set_defaults(func='conda.cli.main_mock_deactivate.execute')

def execute(args, parser):
    if False:
        i = 10
        return i + 15
    raise CondaError("Run 'conda init' before 'conda deactivate'")