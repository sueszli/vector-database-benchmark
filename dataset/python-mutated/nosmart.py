"""Implementation of Transport that never has a smart medium.

This is mainly useful with HTTP transports, which sometimes have a smart medium
and sometimes don't.  By using this decorator, you can force those transports
to never have a smart medium.
"""
from __future__ import absolute_import
from bzrlib import errors
from bzrlib.transport import decorator

class NoSmartTransportDecorator(decorator.TransportDecorator):
    """A decorator for transports that disables get_smart_medium."""

    @classmethod
    def _get_url_prefix(self):
        if False:
            i = 10
            return i + 15
        return 'nosmart+'

    def get_smart_medium(self):
        if False:
            i = 10
            return i + 15
        raise errors.NoSmartMedium(self)

def get_test_permutations():
    if False:
        print('Hello World!')
    'Return the permutations to be used in testing.'
    from bzrlib.tests import test_server
    return [(NoSmartTransportDecorator, test_server.NoSmartTransportServer)]