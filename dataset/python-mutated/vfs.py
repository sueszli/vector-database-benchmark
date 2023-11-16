"""VFS operations for the smart server.

This module defines the smart server methods that are low-level file operations
-- i.e. methods that operate directly on files and directories, rather than
higher-level concepts like branches and revisions.

These methods, plus 'hello' and 'get_bundle', are version 1 of the smart server
protocol, as implemented in bzr 0.11 and later.
"""
from __future__ import absolute_import
import os
from bzrlib import errors
from bzrlib import urlutils
from bzrlib.smart import request

def _deserialise_optional_mode(mode):
    if False:
        print('Hello World!')
    if mode == '':
        return None
    else:
        return int(mode)

def vfs_enabled():
    if False:
        i = 10
        return i + 15
    'Is the VFS enabled ?\n\n    the VFS is disabled when the BZR_NO_SMART_VFS environment variable is set.\n\n    :return: True if it is enabled.\n    '
    return not 'BZR_NO_SMART_VFS' in os.environ

class VfsRequest(request.SmartServerRequest):
    """Base class for VFS requests.

    VFS requests are disabled if vfs_enabled() returns False.
    """

    def _check_enabled(self):
        if False:
            print('Hello World!')
        if not vfs_enabled():
            raise errors.DisabledMethod(self.__class__.__name__)

    def translate_client_path(self, relpath):
        if False:
            print('Hello World!')
        x = request.SmartServerRequest.translate_client_path(self, relpath)
        return str(urlutils.unescape(x))

class HasRequest(VfsRequest):

    def do(self, relpath):
        if False:
            print('Hello World!')
        relpath = self.translate_client_path(relpath)
        r = self._backing_transport.has(relpath) and 'yes' or 'no'
        return request.SuccessfulSmartServerResponse((r,))

class GetRequest(VfsRequest):

    def do(self, relpath):
        if False:
            for i in range(10):
                print('nop')
        relpath = self.translate_client_path(relpath)
        backing_bytes = self._backing_transport.get_bytes(relpath)
        return request.SuccessfulSmartServerResponse(('ok',), backing_bytes)

class AppendRequest(VfsRequest):

    def do(self, relpath, mode):
        if False:
            for i in range(10):
                print('nop')
        relpath = self.translate_client_path(relpath)
        self._relpath = relpath
        self._mode = _deserialise_optional_mode(mode)

    def do_body(self, body_bytes):
        if False:
            while True:
                i = 10
        old_length = self._backing_transport.append_bytes(self._relpath, body_bytes, self._mode)
        return request.SuccessfulSmartServerResponse(('appended', '%d' % old_length))

class DeleteRequest(VfsRequest):

    def do(self, relpath):
        if False:
            return 10
        relpath = self.translate_client_path(relpath)
        self._backing_transport.delete(relpath)
        return request.SuccessfulSmartServerResponse(('ok',))

class IterFilesRecursiveRequest(VfsRequest):

    def do(self, relpath):
        if False:
            i = 10
            return i + 15
        if not relpath.endswith('/'):
            relpath += '/'
        relpath = self.translate_client_path(relpath)
        transport = self._backing_transport.clone(relpath)
        filenames = transport.iter_files_recursive()
        return request.SuccessfulSmartServerResponse(('names',) + tuple(filenames))

class ListDirRequest(VfsRequest):

    def do(self, relpath):
        if False:
            return 10
        if not relpath.endswith('/'):
            relpath += '/'
        relpath = self.translate_client_path(relpath)
        filenames = self._backing_transport.list_dir(relpath)
        return request.SuccessfulSmartServerResponse(('names',) + tuple(filenames))

class MkdirRequest(VfsRequest):

    def do(self, relpath, mode):
        if False:
            while True:
                i = 10
        relpath = self.translate_client_path(relpath)
        self._backing_transport.mkdir(relpath, _deserialise_optional_mode(mode))
        return request.SuccessfulSmartServerResponse(('ok',))

class MoveRequest(VfsRequest):

    def do(self, rel_from, rel_to):
        if False:
            print('Hello World!')
        rel_from = self.translate_client_path(rel_from)
        rel_to = self.translate_client_path(rel_to)
        self._backing_transport.move(rel_from, rel_to)
        return request.SuccessfulSmartServerResponse(('ok',))

class PutRequest(VfsRequest):

    def do(self, relpath, mode):
        if False:
            i = 10
            return i + 15
        relpath = self.translate_client_path(relpath)
        self._relpath = relpath
        self._mode = _deserialise_optional_mode(mode)

    def do_body(self, body_bytes):
        if False:
            return 10
        self._backing_transport.put_bytes(self._relpath, body_bytes, self._mode)
        return request.SuccessfulSmartServerResponse(('ok',))

class PutNonAtomicRequest(VfsRequest):

    def do(self, relpath, mode, create_parent, dir_mode):
        if False:
            for i in range(10):
                print('nop')
        relpath = self.translate_client_path(relpath)
        self._relpath = relpath
        self._dir_mode = _deserialise_optional_mode(dir_mode)
        self._mode = _deserialise_optional_mode(mode)
        self._create_parent = create_parent == 'T'

    def do_body(self, body_bytes):
        if False:
            for i in range(10):
                print('nop')
        self._backing_transport.put_bytes_non_atomic(self._relpath, body_bytes, mode=self._mode, create_parent_dir=self._create_parent, dir_mode=self._dir_mode)
        return request.SuccessfulSmartServerResponse(('ok',))

class ReadvRequest(VfsRequest):

    def do(self, relpath):
        if False:
            return 10
        relpath = self.translate_client_path(relpath)
        self._relpath = relpath

    def do_body(self, body_bytes):
        if False:
            print('Hello World!')
        'accept offsets for a readv request.'
        offsets = self._deserialise_offsets(body_bytes)
        backing_bytes = ''.join((bytes for (offset, bytes) in self._backing_transport.readv(self._relpath, offsets)))
        return request.SuccessfulSmartServerResponse(('readv',), backing_bytes)

    def _deserialise_offsets(self, text):
        if False:
            while True:
                i = 10
        offsets = []
        for line in text.split('\n'):
            if not line:
                continue
            (start, length) = line.split(',')
            offsets.append((int(start), int(length)))
        return offsets

class RenameRequest(VfsRequest):

    def do(self, rel_from, rel_to):
        if False:
            i = 10
            return i + 15
        rel_from = self.translate_client_path(rel_from)
        rel_to = self.translate_client_path(rel_to)
        self._backing_transport.rename(rel_from, rel_to)
        return request.SuccessfulSmartServerResponse(('ok',))

class RmdirRequest(VfsRequest):

    def do(self, relpath):
        if False:
            return 10
        relpath = self.translate_client_path(relpath)
        self._backing_transport.rmdir(relpath)
        return request.SuccessfulSmartServerResponse(('ok',))

class StatRequest(VfsRequest):

    def do(self, relpath):
        if False:
            for i in range(10):
                print('nop')
        if not relpath.endswith('/'):
            relpath += '/'
        relpath = self.translate_client_path(relpath)
        stat = self._backing_transport.stat(relpath)
        return request.SuccessfulSmartServerResponse(('stat', str(stat.st_size), oct(stat.st_mode)))