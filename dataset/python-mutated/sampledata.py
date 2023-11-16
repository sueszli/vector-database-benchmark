""" Download the Bokeh sample data sets to local disk.

To download the Bokeh sample data sets, execute

.. code-block:: sh

    bokeh sampledata

on the command line.

Executing this command is equivalent to running the Python code

.. code-block:: python

    import bokeh.sampledata

    bokeh.sampledata.download()

See :ref:`bokeh.sampledata` for more information on the specific data sets
included in Bokeh's sample data.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from argparse import Namespace
from bokeh import sampledata
from ..subcommand import Subcommand
__all__ = ('Sampledata',)

class Sampledata(Subcommand):
    """ Subcommand to download bokeh sample data sets.

    """
    name = 'sampledata'
    help = 'Download the bokeh sample data sets'

    def invoke(self, args: Namespace) -> None:
        if False:
            i = 10
            return i + 15
        '\n\n        '
        sampledata.download()