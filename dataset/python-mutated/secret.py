""" Generate new secret keys that can be used by the Bokeh server to
cryptographically sign session IDs.

To generate a new secret key for use with Bokeh server, execute

.. code-block:: sh

    bokeh secret

on the command line. The key will be printed to standard output.

The secret key can be provided to the ``bokeh serve`` command with
the ``BOKEH_SECRET_KEY`` environment variable.

.. warning::
    You must keep the secret secret! Protect it like a root password.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from argparse import Namespace
from bokeh.util.token import generate_secret_key
from ..subcommand import Subcommand
__all__ = ('Secret',)

class Secret(Subcommand):
    """ Subcommand to generate a new secret key.

    """
    name = 'secret'
    help = 'Create a Bokeh secret key for use with Bokeh server'

    def invoke(self, args: Namespace) -> None:
        if False:
            return 10
        '\n\n        '
        key = generate_secret_key()
        print(key)