"""A transport decorator that filters all paths that are passed to it."""
from __future__ import absolute_import
from bzrlib import urlutils
from bzrlib.transport import register_transport, Server, Transport, unregister_transport

class PathFilteringServer(Server):
    """Transport server for PathFilteringTransport.

    It holds the backing_transport and filter_func for PathFilteringTransports.
    All paths will be passed through filter_func before calling into the
    backing_transport.

    Note that paths returned from the backing transport are *not* altered in
    anyway.  So, depending on the filter_func, PathFilteringTransports might
    not conform to the usual expectations of Transport behaviour; e.g. 'name'
    in t.list_dir('dir') might not imply t.has('dir/name') is True!  A filter
    that merely prefixes a constant path segment will be essentially
    transparent, whereas a filter that does rot13 to paths will break
    expectations and probably cause confusing errors.  So choose your
    filter_func with care.
    """

    def __init__(self, backing_transport, filter_func):
        if False:
            return 10
        'Constructor.\n\n        :param backing_transport: a transport\n        :param filter_func: a callable that takes paths, and translates them\n            into paths for use with the backing transport.\n        '
        self.backing_transport = backing_transport
        self.filter_func = filter_func

    def _factory(self, url):
        if False:
            i = 10
            return i + 15
        return PathFilteringTransport(self, url)

    def get_url(self):
        if False:
            for i in range(10):
                print('nop')
        return self.scheme

    def start_server(self):
        if False:
            for i in range(10):
                print('nop')
        self.scheme = 'filtered-%d:///' % id(self)
        register_transport(self.scheme, self._factory)

    def stop_server(self):
        if False:
            while True:
                i = 10
        unregister_transport(self.scheme, self._factory)

class PathFilteringTransport(Transport):
    """A PathFilteringTransport.

    Please see PathFilteringServer for details.
    """

    def __init__(self, server, base):
        if False:
            i = 10
            return i + 15
        self.server = server
        if not base.endswith('/'):
            base += '/'
        Transport.__init__(self, base)
        self.base_path = self.base[len(self.server.scheme) - 1:]
        self.scheme = self.server.scheme

    def _relpath_from_server_root(self, relpath):
        if False:
            while True:
                i = 10
        unfiltered_path = urlutils.URL._combine_paths(self.base_path, relpath)
        if not unfiltered_path.startswith('/'):
            raise ValueError(unfiltered_path)
        return unfiltered_path[1:]

    def _filter(self, relpath):
        if False:
            for i in range(10):
                print('nop')
        return self.server.filter_func(self._relpath_from_server_root(relpath))

    def _call(self, methodname, relpath, *args):
        if False:
            print('Hello World!')
        'Helper for Transport methods of the form:\n            operation(path, [other args ...])\n        '
        backing_method = getattr(self.server.backing_transport, methodname)
        return backing_method(self._filter(relpath), *args)

    def abspath(self, relpath):
        if False:
            for i in range(10):
                print('nop')
        return self.scheme + self._relpath_from_server_root(relpath)

    def append_file(self, relpath, f, mode=None):
        if False:
            i = 10
            return i + 15
        return self._call('append_file', relpath, f, mode)

    def _can_roundtrip_unix_modebits(self):
        if False:
            while True:
                i = 10
        return self.server.backing_transport._can_roundtrip_unix_modebits()

    def clone(self, relpath):
        if False:
            i = 10
            return i + 15
        return self.__class__(self.server, self.abspath(relpath))

    def delete(self, relpath):
        if False:
            for i in range(10):
                print('nop')
        return self._call('delete', relpath)

    def delete_tree(self, relpath):
        if False:
            return 10
        return self._call('delete_tree', relpath)

    def external_url(self):
        if False:
            return 10
        'See bzrlib.transport.Transport.external_url.'
        return self.server.backing_transport.external_url()

    def get(self, relpath):
        if False:
            return 10
        return self._call('get', relpath)

    def has(self, relpath):
        if False:
            return 10
        return self._call('has', relpath)

    def is_readonly(self):
        if False:
            print('Hello World!')
        return self.server.backing_transport.is_readonly()

    def iter_files_recursive(self):
        if False:
            print('Hello World!')
        backing_transport = self.server.backing_transport.clone(self._filter('.'))
        return backing_transport.iter_files_recursive()

    def listable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.server.backing_transport.listable()

    def list_dir(self, relpath):
        if False:
            return 10
        return self._call('list_dir', relpath)

    def lock_read(self, relpath):
        if False:
            while True:
                i = 10
        return self._call('lock_read', relpath)

    def lock_write(self, relpath):
        if False:
            print('Hello World!')
        return self._call('lock_write', relpath)

    def mkdir(self, relpath, mode=None):
        if False:
            while True:
                i = 10
        return self._call('mkdir', relpath, mode)

    def open_write_stream(self, relpath, mode=None):
        if False:
            print('Hello World!')
        return self._call('open_write_stream', relpath, mode)

    def put_file(self, relpath, f, mode=None):
        if False:
            return 10
        return self._call('put_file', relpath, f, mode)

    def rename(self, rel_from, rel_to):
        if False:
            while True:
                i = 10
        return self._call('rename', rel_from, self._filter(rel_to))

    def rmdir(self, relpath):
        if False:
            while True:
                i = 10
        return self._call('rmdir', relpath)

    def stat(self, relpath):
        if False:
            for i in range(10):
                print('nop')
        return self._call('stat', relpath)

def get_test_permutations():
    if False:
        while True:
            i = 10
    'Return the permutations to be used in testing.'
    from bzrlib.tests import test_server
    return [(PathFilteringTransport, test_server.TestingPathFilteringServer)]