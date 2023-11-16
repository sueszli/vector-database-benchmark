"""Interface for accessing the file system with automatic caching.

The idea is to cache the results of any file system state reads during
a single transaction. This has two main benefits:

* This avoids redundant syscalls, as we won't perform the same OS
  operations multiple times.

* This makes it easier to reason about concurrent FS updates, as different
  operations targeting the same paths can't report different state during
  a transaction.

Note that this only deals with reading state, not writing.

Properties maintained by the API:

* The contents of the file are always from the same or later time compared
  to the reported mtime of the file, even if mtime is queried after reading
  a file.

* Repeating an operation produces the same result as the first one during
  a transaction.

* Call flush() to start a new transaction (flush the caches).

The API is a bit limited. It's easy to add new cached operations, however.
You should perform all file system reads through the API to actually take
advantage of the benefits.
"""
from __future__ import annotations
import os
import stat
from mypy_extensions import mypyc_attr
from mypy.util import hash_digest

@mypyc_attr(allow_interpreted_subclasses=True)
class FileSystemCache:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.package_root: list[str] = []
        self.flush()

    def set_package_root(self, package_root: list[str]) -> None:
        if False:
            return 10
        self.package_root = package_root

    def flush(self) -> None:
        if False:
            i = 10
            return i + 15
        'Start another transaction and empty all caches.'
        self.stat_cache: dict[str, os.stat_result] = {}
        self.stat_error_cache: dict[str, OSError] = {}
        self.listdir_cache: dict[str, list[str]] = {}
        self.listdir_error_cache: dict[str, OSError] = {}
        self.isfile_case_cache: dict[str, bool] = {}
        self.exists_case_cache: dict[str, bool] = {}
        self.read_cache: dict[str, bytes] = {}
        self.read_error_cache: dict[str, Exception] = {}
        self.hash_cache: dict[str, str] = {}
        self.fake_package_cache: set[str] = set()

    def stat(self, path: str) -> os.stat_result:
        if False:
            for i in range(10):
                print('nop')
        if path in self.stat_cache:
            return self.stat_cache[path]
        if path in self.stat_error_cache:
            raise copy_os_error(self.stat_error_cache[path])
        try:
            st = os.stat(path)
        except OSError as err:
            if self.init_under_package_root(path):
                try:
                    return self._fake_init(path)
                except OSError:
                    pass
            self.stat_error_cache[path] = copy_os_error(err)
            raise err
        self.stat_cache[path] = st
        return st

    def init_under_package_root(self, path: str) -> bool:
        if False:
            return 10
        "Is this path an __init__.py under a package root?\n\n        This is used to detect packages that don't contain __init__.py\n        files, which is needed to support Bazel.  The function should\n        only be called for non-existing files.\n\n        It will return True if it refers to a __init__.py file that\n        Bazel would create, so that at runtime Python would think the\n        directory containing it is a package.  For this to work you\n        must pass one or more package roots using the --package-root\n        flag.\n\n        As an exceptional case, any directory that is a package root\n        itself will not be considered to contain a __init__.py file.\n        This is different from the rules Bazel itself applies, but is\n        necessary for mypy to properly distinguish packages from other\n        directories.\n\n        See https://docs.bazel.build/versions/master/be/python.html,\n        where this behavior is described under legacy_create_init.\n        "
        if not self.package_root:
            return False
        (dirname, basename) = os.path.split(path)
        if basename != '__init__.py':
            return False
        if not os.path.basename(dirname).isidentifier():
            return False
        try:
            st = self.stat(dirname)
        except OSError:
            return False
        else:
            if not stat.S_ISDIR(st.st_mode):
                return False
        ok = False
        (drive, path) = os.path.splitdrive(path)
        if os.path.isabs(path):
            path = os.path.relpath(path)
        path = os.path.normpath(path)
        for root in self.package_root:
            if path.startswith(root):
                if path == root + basename:
                    ok = False
                    break
                else:
                    ok = True
        return ok

    def _fake_init(self, path: str) -> os.stat_result:
        if False:
            return 10
        'Prime the cache with a fake __init__.py file.\n\n        This makes code that looks for path believe an empty file by\n        that name exists.  Should only be called after\n        init_under_package_root() returns True.\n        '
        (dirname, basename) = os.path.split(path)
        assert basename == '__init__.py', path
        assert not os.path.exists(path), path
        dirname = os.path.normpath(dirname)
        st = self.stat(dirname)
        seq: list[float] = list(st)
        seq[stat.ST_MODE] = stat.S_IFREG | 292
        seq[stat.ST_INO] = 1
        seq[stat.ST_NLINK] = 1
        seq[stat.ST_SIZE] = 0
        st = os.stat_result(seq)
        self.stat_cache[path] = st
        self.fake_package_cache.add(dirname)
        return st

    def listdir(self, path: str) -> list[str]:
        if False:
            return 10
        path = os.path.normpath(path)
        if path in self.listdir_cache:
            res = self.listdir_cache[path]
            if path in self.fake_package_cache and '__init__.py' not in res:
                res.append('__init__.py')
            return res
        if path in self.listdir_error_cache:
            raise copy_os_error(self.listdir_error_cache[path])
        try:
            results = os.listdir(path)
        except OSError as err:
            self.listdir_error_cache[path] = copy_os_error(err)
            raise err
        self.listdir_cache[path] = results
        if path in self.fake_package_cache and '__init__.py' not in results:
            results.append('__init__.py')
        return results

    def isfile(self, path: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        try:
            st = self.stat(path)
        except OSError:
            return False
        return stat.S_ISREG(st.st_mode)

    def isfile_case(self, path: str, prefix: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Return whether path exists and is a file.\n\n        On case-insensitive filesystems (like Mac or Windows) this returns\n        False if the case of path's last component does not exactly match\n        the case found in the filesystem.\n\n        We check also the case of other path components up to prefix.\n        For example, if path is 'user-stubs/pack/mod.pyi' and prefix is 'user-stubs',\n        we check that the case of 'pack' and 'mod.py' matches exactly, 'user-stubs' will be\n        case insensitive on case insensitive filesystems.\n\n        The caller must ensure that prefix is a valid file system prefix of path.\n        "
        if not self.isfile(path):
            return False
        if path in self.isfile_case_cache:
            return self.isfile_case_cache[path]
        (head, tail) = os.path.split(path)
        if not tail:
            self.isfile_case_cache[path] = False
            return False
        try:
            names = self.listdir(head)
            res = tail in names
        except OSError:
            res = False
        if res:
            res = self.exists_case(head, prefix)
        self.isfile_case_cache[path] = res
        return res

    def exists_case(self, path: str, prefix: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Return whether path exists - checking path components in case sensitive\n        fashion, up to prefix.\n        '
        if path in self.exists_case_cache:
            return self.exists_case_cache[path]
        (head, tail) = os.path.split(path)
        if not head.startswith(prefix) or not tail:
            self.exists_case_cache[path] = True
            return True
        try:
            names = self.listdir(head)
            res = tail in names
        except OSError:
            res = False
        if res:
            res = self.exists_case(head, prefix)
        self.exists_case_cache[path] = res
        return res

    def isdir(self, path: str) -> bool:
        if False:
            i = 10
            return i + 15
        try:
            st = self.stat(path)
        except OSError:
            return False
        return stat.S_ISDIR(st.st_mode)

    def exists(self, path: str) -> bool:
        if False:
            print('Hello World!')
        try:
            self.stat(path)
        except FileNotFoundError:
            return False
        return True

    def read(self, path: str) -> bytes:
        if False:
            print('Hello World!')
        if path in self.read_cache:
            return self.read_cache[path]
        if path in self.read_error_cache:
            raise self.read_error_cache[path]
        self.stat(path)
        (dirname, basename) = os.path.split(path)
        dirname = os.path.normpath(dirname)
        if basename == '__init__.py' and dirname in self.fake_package_cache:
            data = b''
        else:
            try:
                with open(path, 'rb') as f:
                    data = f.read()
            except OSError as err:
                self.read_error_cache[path] = err
                raise
        self.read_cache[path] = data
        self.hash_cache[path] = hash_digest(data)
        return data

    def hash_digest(self, path: str) -> str:
        if False:
            while True:
                i = 10
        if path not in self.hash_cache:
            self.read(path)
        return self.hash_cache[path]

    def samefile(self, f1: str, f2: str) -> bool:
        if False:
            i = 10
            return i + 15
        s1 = self.stat(f1)
        s2 = self.stat(f2)
        return os.path.samestat(s1, s2)

def copy_os_error(e: OSError) -> OSError:
    if False:
        return 10
    new = OSError(*e.args)
    new.errno = e.errno
    new.strerror = e.strerror
    new.filename = e.filename
    if e.filename2:
        new.filename2 = e.filename2
    return new