"""Implementation of Transport that prevents access to locations above a set
root.
"""
from __future__ import absolute_import
from bzrlib.transport import pathfilter, register_transport

class ChrootServer(pathfilter.PathFilteringServer):
    """User space 'chroot' facility.

    The server's get_url returns the url for a chroot transport mapped to the
    backing transport. The url is of the form chroot-xxx:/// so parent
    directories of the backing transport are not visible. The chroot url will
    not allow '..' sequences to result in requests to the chroot affecting
    directories outside the backing transport.

    PathFilteringServer does all the path sanitation needed to enforce a
    chroot, so this is a simple subclass of PathFilteringServer that ignores
    filter_func.
    """

    def __init__(self, backing_transport):
        if False:
            return 10
        pathfilter.PathFilteringServer.__init__(self, backing_transport, None)

    def _factory(self, url):
        if False:
            print('Hello World!')
        return ChrootTransport(self, url)

    def start_server(self):
        if False:
            print('Hello World!')
        self.scheme = 'chroot-%d:///' % id(self)
        register_transport(self.scheme, self._factory)

class ChrootTransport(pathfilter.PathFilteringTransport):
    """A ChrootTransport.

    Please see ChrootServer for details.
    """

    def _filter(self, relpath):
        if False:
            i = 10
            return i + 15
        return self._relpath_from_server_root(relpath)

def get_test_permutations():
    if False:
        return 10
    'Return the permutations to be used in testing.'
    from bzrlib.tests import test_server
    return [(ChrootTransport, test_server.TestingChrootServer)]