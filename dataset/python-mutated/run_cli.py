from feeluown.utils import aio
from feeluown.cli import climain

def run_cli(args):
    if False:
        return 10
    aio.run(climain(args))