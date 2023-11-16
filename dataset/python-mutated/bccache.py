"""
    jinja2.bccache
    ~~~~~~~~~~~~~~

    This module implements the bytecode cache system Jinja is optionally
    using.  This is useful if you have very complex template situations and
    the compiliation of all those templates slow down your application too
    much.

    Situations where this is useful are often forking web applications that
    are initialized on the first request.

    :copyright: (c) 2017 by the Jinja Team.
    :license: BSD.
"""
from os import path, listdir
import os
import sys
import stat
import errno
import marshal
import tempfile
import fnmatch
from hashlib import sha1
from jinja2.utils import open_if_exists
from jinja2._compat import BytesIO, pickle, PY2, text_type
if not PY2:
    marshal_dump = marshal.dump
    marshal_load = marshal.load
else:

    def marshal_dump(code, f):
        if False:
            while True:
                i = 10
        if isinstance(f, file):
            marshal.dump(code, f)
        else:
            f.write(marshal.dumps(code))

    def marshal_load(f):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(f, file):
            return marshal.load(f)
        return marshal.loads(f.read())
bc_version = 3
bc_magic = 'j2'.encode('ascii') + pickle.dumps(bc_version, 2) + pickle.dumps(sys.version_info[0] << 24 | sys.version_info[1])

class Bucket(object):
    """Buckets are used to store the bytecode for one template.  It's created
    and initialized by the bytecode cache and passed to the loading functions.

    The buckets get an internal checksum from the cache assigned and use this
    to automatically reject outdated cache material.  Individual bytecode
    cache subclasses don't have to care about cache invalidation.
    """

    def __init__(self, environment, key, checksum):
        if False:
            print('Hello World!')
        self.environment = environment
        self.key = key
        self.checksum = checksum
        self.reset()

    def reset(self):
        if False:
            while True:
                i = 10
        'Resets the bucket (unloads the bytecode).'
        self.code = None

    def load_bytecode(self, f):
        if False:
            for i in range(10):
                print('nop')
        'Loads bytecode from a file or file like object.'
        magic = f.read(len(bc_magic))
        if magic != bc_magic:
            self.reset()
            return
        checksum = pickle.load(f)
        if self.checksum != checksum:
            self.reset()
            return
        try:
            self.code = marshal_load(f)
        except (EOFError, ValueError, TypeError):
            self.reset()
            return

    def write_bytecode(self, f):
        if False:
            print('Hello World!')
        'Dump the bytecode into the file or file like object passed.'
        if self.code is None:
            raise TypeError("can't write empty bucket")
        f.write(bc_magic)
        pickle.dump(self.checksum, f, 2)
        marshal_dump(self.code, f)

    def bytecode_from_string(self, string):
        if False:
            for i in range(10):
                print('nop')
        'Load bytecode from a string.'
        self.load_bytecode(BytesIO(string))

    def bytecode_to_string(self):
        if False:
            while True:
                i = 10
        'Return the bytecode as string.'
        out = BytesIO()
        self.write_bytecode(out)
        return out.getvalue()

class BytecodeCache(object):
    """To implement your own bytecode cache you have to subclass this class
    and override :meth:`load_bytecode` and :meth:`dump_bytecode`.  Both of
    these methods are passed a :class:`~jinja2.bccache.Bucket`.

    A very basic bytecode cache that saves the bytecode on the file system::

        from os import path

        class MyCache(BytecodeCache):

            def __init__(self, directory):
                self.directory = directory

            def load_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                if path.exists(filename):
                    with open(filename, 'rb') as f:
                        bucket.load_bytecode(f)

            def dump_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                with open(filename, 'wb') as f:
                    bucket.write_bytecode(f)

    A more advanced version of a filesystem based bytecode cache is part of
    Jinja2.
    """

    def load_bytecode(self, bucket):
        if False:
            for i in range(10):
                print('nop')
        'Subclasses have to override this method to load bytecode into a\n        bucket.  If they are not able to find code in the cache for the\n        bucket, it must not do anything.\n        '
        raise NotImplementedError()

    def dump_bytecode(self, bucket):
        if False:
            while True:
                i = 10
        'Subclasses have to override this method to write the bytecode\n        from a bucket back to the cache.  If it unable to do so it must not\n        fail silently but raise an exception.\n        '
        raise NotImplementedError()

    def clear(self):
        if False:
            print('Hello World!')
        'Clears the cache.  This method is not used by Jinja2 but should be\n        implemented to allow applications to clear the bytecode cache used\n        by a particular environment.\n        '

    def get_cache_key(self, name, filename=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the unique hash key for this template name.'
        hash = sha1(name.encode('utf-8'))
        if filename is not None:
            filename = '|' + filename
            if isinstance(filename, text_type):
                filename = filename.encode('utf-8')
            hash.update(filename)
        return hash.hexdigest()

    def get_source_checksum(self, source):
        if False:
            return 10
        'Returns a checksum for the source.'
        return sha1(source.encode('utf-8')).hexdigest()

    def get_bucket(self, environment, name, filename, source):
        if False:
            i = 10
            return i + 15
        'Return a cache bucket for the given template.  All arguments are\n        mandatory but filename may be `None`.\n        '
        key = self.get_cache_key(name, filename)
        checksum = self.get_source_checksum(source)
        bucket = Bucket(environment, key, checksum)
        self.load_bytecode(bucket)
        return bucket

    def set_bucket(self, bucket):
        if False:
            print('Hello World!')
        'Put the bucket into the cache.'
        self.dump_bytecode(bucket)

class FileSystemBytecodeCache(BytecodeCache):
    """A bytecode cache that stores bytecode on the filesystem.  It accepts
    two arguments: The directory where the cache items are stored and a
    pattern string that is used to build the filename.

    If no directory is specified a default cache directory is selected.  On
    Windows the user's temp directory is used, on UNIX systems a directory
    is created for the user in the system temp directory.

    The pattern can be used to have multiple separate caches operate on the
    same directory.  The default pattern is ``'__jinja2_%s.cache'``.  ``%s``
    is replaced with the cache key.

    >>> bcc = FileSystemBytecodeCache('/tmp/jinja_cache', '%s.cache')

    This bytecode cache supports clearing of the cache using the clear method.
    """

    def __init__(self, directory=None, pattern='__jinja2_%s.cache'):
        if False:
            i = 10
            return i + 15
        if directory is None:
            directory = self._get_default_cache_dir()
        self.directory = directory
        self.pattern = pattern

    def _get_default_cache_dir(self):
        if False:
            while True:
                i = 10

        def _unsafe_dir():
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('Cannot determine safe temp directory.  You need to explicitly provide one.')
        tmpdir = tempfile.gettempdir()
        if os.name == 'nt':
            return tmpdir
        if not hasattr(os, 'getuid'):
            _unsafe_dir()
        dirname = '_jinja2-cache-%d' % os.getuid()
        actual_dir = os.path.join(tmpdir, dirname)
        try:
            os.mkdir(actual_dir, stat.S_IRWXU)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.chmod(actual_dir, stat.S_IRWXU)
            actual_dir_stat = os.lstat(actual_dir)
            if actual_dir_stat.st_uid != os.getuid() or not stat.S_ISDIR(actual_dir_stat.st_mode) or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU:
                _unsafe_dir()
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        actual_dir_stat = os.lstat(actual_dir)
        if actual_dir_stat.st_uid != os.getuid() or not stat.S_ISDIR(actual_dir_stat.st_mode) or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU:
            _unsafe_dir()
        return actual_dir

    def _get_cache_filename(self, bucket):
        if False:
            return 10
        return path.join(self.directory, self.pattern % bucket.key)

    def load_bytecode(self, bucket):
        if False:
            print('Hello World!')
        f = open_if_exists(self._get_cache_filename(bucket), 'rb')
        if f is not None:
            try:
                bucket.load_bytecode(f)
            finally:
                f.close()

    def dump_bytecode(self, bucket):
        if False:
            return 10
        f = open(self._get_cache_filename(bucket), 'wb')
        try:
            bucket.write_bytecode(f)
        finally:
            f.close()

    def clear(self):
        if False:
            return 10
        from os import remove
        files = fnmatch.filter(listdir(self.directory), self.pattern % '*')
        for filename in files:
            try:
                remove(path.join(self.directory, filename))
            except OSError:
                pass

class MemcachedBytecodeCache(BytecodeCache):
    """This class implements a bytecode cache that uses a memcache cache for
    storing the information.  It does not enforce a specific memcache library
    (tummy's memcache or cmemcache) but will accept any class that provides
    the minimal interface required.

    Libraries compatible with this class:

    -   `cachelib <https://github.com/pallets/cachelib>`_
    -   `python-memcached <https://pypi.org/project/python-memcached/>`_

    (Unfortunately the django cache interface is not compatible because it
    does not support storing binary data, only unicode.  You can however pass
    the underlying cache client to the bytecode cache which is available
    as `django.core.cache.cache._client`.)

    The minimal interface for the client passed to the constructor is this:

    .. class:: MinimalClientInterface

        .. method:: set(key, value[, timeout])

            Stores the bytecode in the cache.  `value` is a string and
            `timeout` the timeout of the key.  If timeout is not provided
            a default timeout or no timeout should be assumed, if it's
            provided it's an integer with the number of seconds the cache
            item should exist.

        .. method:: get(key)

            Returns the value for the cache key.  If the item does not
            exist in the cache the return value must be `None`.

    The other arguments to the constructor are the prefix for all keys that
    is added before the actual cache key and the timeout for the bytecode in
    the cache system.  We recommend a high (or no) timeout.

    This bytecode cache does not support clearing of used items in the cache.
    The clear method is a no-operation function.

    .. versionadded:: 2.7
       Added support for ignoring memcache errors through the
       `ignore_memcache_errors` parameter.
    """

    def __init__(self, client, prefix='jinja2/bytecode/', timeout=None, ignore_memcache_errors=True):
        if False:
            return 10
        self.client = client
        self.prefix = prefix
        self.timeout = timeout
        self.ignore_memcache_errors = ignore_memcache_errors

    def load_bytecode(self, bucket):
        if False:
            i = 10
            return i + 15
        try:
            code = self.client.get(self.prefix + bucket.key)
        except Exception:
            if not self.ignore_memcache_errors:
                raise
            code = None
        if code is not None:
            bucket.bytecode_from_string(code)

    def dump_bytecode(self, bucket):
        if False:
            while True:
                i = 10
        args = (self.prefix + bucket.key, bucket.bytecode_to_string())
        if self.timeout is not None:
            args += (self.timeout,)
        try:
            self.client.set(*args)
        except Exception:
            if not self.ignore_memcache_errors:
                raise