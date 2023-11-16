__package__ = 'archivebox.cli'
__command__ = 'archivebox setup'
import sys
import argparse
from typing import Optional, List, IO
from ..main import setup
from ..util import docstring
from ..config import OUTPUT_DIR
from ..logging_util import SmartFormatter, reject_stdin

@docstring(setup.__doc__)
def main(args: Optional[List[str]]=None, stdin: Optional[IO]=None, pwd: Optional[str]=None) -> None:
    if False:
        return 10
    parser = argparse.ArgumentParser(prog=__command__, description=setup.__doc__, add_help=True, formatter_class=SmartFormatter)
    command = parser.parse_args(args or ())
    reject_stdin(__command__, stdin)
    setup(out_dir=pwd or OUTPUT_DIR)
if __name__ == '__main__':
    main(args=sys.argv[1:], stdin=sys.stdin)