"""This script produces the expression parser."""
from __future__ import annotations
import argparse
import os
import subprocess
from typing import Optional, Sequence
from . import common
from . import setup
_PARSER = argparse.ArgumentParser(description="\nRun this script from the oppia root folder:\n    python -m scripts.create_expression_parser\nThe root folder MUST be named 'oppia'.\n")

def main(args: Optional[Sequence[str]]=None) -> None:
    if False:
        return 10
    'Produces the expression parser.'
    unused_parsed_args = _PARSER.parse_args(args=args)
    setup.main(args=[])
    expression_parser_definition = os.path.join('core', 'templates', 'expressions', 'parser.pegjs')
    expression_parser_js = os.path.join('core', 'templates', 'expressions', 'parser.js')
    common.install_npm_library('pegjs', '0.8.0', common.OPPIA_TOOLS_DIR)
    subprocess.check_call([os.path.join(common.NODE_MODULES_PATH, 'pegjs', 'bin', 'pegjs'), expression_parser_definition, expression_parser_js])
    print('Done!')
if __name__ == '__main__':
    main()