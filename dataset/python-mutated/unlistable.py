"""Transport implementation that disables listing to simulate HTTP cheaply."""
from __future__ import absolute_import
from bzrlib.transport import Transport
from bzrlib.transport import decorator

class UnlistableTransportDecorator(decorator.TransportDecorator):
    """A transport that disables file listing for testing."""

    @classmethod
    def _get_url_prefix(self):
        if False:
            while True:
                i = 10
        "Unlistable transports are identified by 'unlistable+'"
        return 'unlistable+'

    def iter_files_recursive(self):
        if False:
            return 10
        Transport.iter_files_recursive(self)

    def listable(self):
        if False:
            while True:
                i = 10
        return False

    def list_dir(self, relpath):
        if False:
            while True:
                i = 10
        Transport.list_dir(self, relpath)

def get_test_permutations():
    if False:
        i = 10
        return i + 15
    'Return the permutations to be used in testing.'
    from bzrlib.tests import test_server
    return [(UnlistableTransportDecorator, test_server.UnlistableServer)]