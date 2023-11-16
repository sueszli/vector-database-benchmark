""" Provide a ``main`` function to run bokeh commands.

The following are equivalent:

* Running the ``bokeh`` command line script:

  .. code-block:: sh

      bokeh serve --show app.py

* Using ``python -m bokeh``:

  .. code-block:: sh

      python -m bokeh serve --show app.py

* Executing ``main`` programmatically:

  .. code-block:: python

      from bokeh.command.bootstrap import main

      main(["bokeh", "serve", "--show", "app.py"])

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import argparse
import sys
from typing import Sequence
from bokeh import __version__
from bokeh.settings import settings
from bokeh.util.strings import nice_join
from . import subcommands
from .util import die
__all__ = ('main',)

def main(argv: Sequence[str]) -> None:
    if False:
        i = 10
        return i + 15
    ' Execute the Bokeh command.\n\n    Args:\n        argv (seq[str]) : a list of command line arguments to process\n\n    Returns:\n        None\n\n    The first item in ``argv`` is typically "bokeh", and the second should\n    be the name of one of the available subcommands:\n\n    * :ref:`info <bokeh.command.subcommands.info>`\n    * :ref:`json <bokeh.command.subcommands.json>`\n    * :ref:`sampledata <bokeh.command.subcommands.sampledata>`\n    * :ref:`secret <bokeh.command.subcommands.secret>`\n    * :ref:`serve <bokeh.command.subcommands.serve>`\n    * :ref:`static <bokeh.command.subcommands.static>`\n\n    '
    if len(argv) == 1:
        die('ERROR: Must specify subcommand, one of: %s' % nice_join([x.name for x in subcommands.all]))
    parser = argparse.ArgumentParser(prog=argv[0], epilog="See '<command> --help' to read about a specific subcommand.")
    parser.add_argument('-v', '--version', action='version', version=__version__)
    subs = parser.add_subparsers(help='Sub-commands')
    for cls in subcommands.all:
        subparser = subs.add_parser(cls.name, help=cls.help)
        subcommand = cls(parser=subparser)
        subparser.set_defaults(invoke=subcommand.invoke)
    args = parser.parse_args(argv[1:])
    try:
        ret = args.invoke(args)
    except Exception as e:
        if settings.dev:
            raise
        else:
            die('ERROR: ' + str(e))
    if ret is False:
        sys.exit(1)
    elif ret is not True and isinstance(ret, int) and (ret != 0):
        sys.exit(ret)