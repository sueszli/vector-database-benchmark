"""PEP 621 compatible entry point used when `conda init` has not updated the user shell profile."""
import os
import sys
from logging import getLogger
log = getLogger(__name__)

def pip_installed_post_parse_hook(args, p):
    if False:
        print('Hello World!')
    from .. import CondaError
    if args.cmd not in ('init', 'info'):
        raise CondaError("Conda has not been initialized.\n\nTo enable full conda functionality, please run 'conda init'.\nFor additional information, see 'conda init --help'.\n")

def main(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    from .main import main
    os.environ['CONDA_PIP_UNINITIALIZED'] = 'true'
    kwargs['post_parse_hook'] = pip_installed_post_parse_hook
    return main(*args, **kwargs)
if __name__ == '__main__':
    sys.exit(main())