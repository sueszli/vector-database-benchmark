"""
:class:`LocalTarget` provides a concrete implementation of a :py:class:`~luigi.target.Target` class that uses files on the local file system
"""
import os
import random
import shutil
import tempfile
import io
import warnings
import errno
from luigi.format import FileWrapper, get_default_format
from luigi.target import FileAlreadyExists, MissingParentDirectory, NotADirectory, FileSystem, FileSystemTarget, AtomicLocalFile

class atomic_file(AtomicLocalFile):
    """Simple class that writes to a temp file and moves it on close()
    Also cleans up the temp file if close is not invoked
    """

    def move_to_final_destination(self):
        if False:
            i = 10
            return i + 15
        os.rename(self.tmp_path, self.path)

    def generate_tmp_path(self, path):
        if False:
            while True:
                i = 10
        return path + '-luigi-tmp-%09d' % random.randrange(0, 10000000000)

class LocalFileSystem(FileSystem):
    """
    Wrapper for access to file system operations.

    Work in progress - add things as needed.
    """

    def copy(self, old_path, new_path, raise_if_exists=False):
        if False:
            for i in range(10):
                print('nop')
        if raise_if_exists and os.path.exists(new_path):
            raise RuntimeError('Destination exists: %s' % new_path)
        d = os.path.dirname(new_path)
        if d and (not os.path.exists(d)):
            self.mkdir(d)
        shutil.copy(old_path, new_path)

    def exists(self, path):
        if False:
            i = 10
            return i + 15
        return os.path.exists(path)

    def mkdir(self, path, parents=True, raise_if_exists=False):
        if False:
            while True:
                i = 10
        if self.exists(path):
            if raise_if_exists:
                raise FileAlreadyExists()
            elif not self.isdir(path):
                raise NotADirectory()
            else:
                return
        if parents:
            try:
                os.makedirs(path)
            except OSError as err:
                if err.errno != errno.EEXIST:
                    raise
        else:
            if not os.path.exists(os.path.dirname(path)):
                raise MissingParentDirectory()
            os.mkdir(path)

    def isdir(self, path):
        if False:
            while True:
                i = 10
        return os.path.isdir(path)

    def listdir(self, path):
        if False:
            for i in range(10):
                print('nop')
        for (dir_, _, files) in os.walk(path):
            assert dir_.startswith(path)
            for name in files:
                yield os.path.join(dir_, name)

    def remove(self, path, recursive=True):
        if False:
            while True:
                i = 10
        if recursive and self.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    def move(self, old_path, new_path, raise_if_exists=False):
        if False:
            print('Hello World!')
        '\n        Move file atomically. If source and destination are located\n        on different filesystems, atomicity is approximated\n        but cannot be guaranteed.\n        '
        if raise_if_exists and os.path.exists(new_path):
            raise FileAlreadyExists('Destination exists: %s' % new_path)
        d = os.path.dirname(new_path)
        if d and (not os.path.exists(d)):
            self.mkdir(d)
        try:
            os.rename(old_path, new_path)
        except OSError as err:
            if err.errno == errno.EXDEV:
                new_path_tmp = '%s-%09d' % (new_path, random.randint(0, 999999999))
                shutil.copy(old_path, new_path_tmp)
                os.rename(new_path_tmp, new_path)
                os.remove(old_path)
            else:
                raise err

    def rename_dont_move(self, path, dest):
        if False:
            return 10
        "\n        Rename ``path`` to ``dest``, but don't move it into the ``dest``\n        folder (if it is a folder). This method is just a wrapper around the\n        ``move`` method of LocalTarget.\n        "
        self.move(path, dest, raise_if_exists=True)

class LocalTarget(FileSystemTarget):
    fs = LocalFileSystem()

    def __init__(self, path=None, format=None, is_tmp=False):
        if False:
            print('Hello World!')
        if format is None:
            format = get_default_format()
        if not path:
            if not is_tmp:
                raise Exception('path or is_tmp must be set')
            path = os.path.join(tempfile.gettempdir(), 'luigi-tmp-%09d' % random.randint(0, 999999999))
        super(LocalTarget, self).__init__(path)
        self.format = format
        self.is_tmp = is_tmp

    def makedirs(self):
        if False:
            return 10
        '\n        Create all parent folders if they do not exist.\n        '
        normpath = os.path.normpath(self.path)
        parentfolder = os.path.dirname(normpath)
        if parentfolder:
            try:
                os.makedirs(parentfolder)
            except OSError:
                pass

    def open(self, mode='r'):
        if False:
            while True:
                i = 10
        rwmode = mode.replace('b', '').replace('t', '')
        if rwmode == 'w':
            self.makedirs()
            return self.format.pipe_writer(atomic_file(self.path))
        elif rwmode == 'r':
            fileobj = FileWrapper(io.BufferedReader(io.FileIO(self.path, mode)))
            return self.format.pipe_reader(fileobj)
        else:
            raise Exception("mode must be 'r' or 'w' (got: %s)" % mode)

    def move(self, new_path, raise_if_exists=False):
        if False:
            while True:
                i = 10
        self.fs.move(self.path, new_path, raise_if_exists=raise_if_exists)

    def move_dir(self, new_path):
        if False:
            i = 10
            return i + 15
        self.move(new_path)

    def remove(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.remove(self.path)

    def copy(self, new_path, raise_if_exists=False):
        if False:
            print('Hello World!')
        self.fs.copy(self.path, new_path, raise_if_exists)

    @property
    def fn(self):
        if False:
            while True:
                i = 10
        warnings.warn('Use LocalTarget.path to reference filename', DeprecationWarning, stacklevel=2)
        return self.path

    def __del__(self):
        if False:
            print('Hello World!')
        if hasattr(self, 'is_tmp') and self.is_tmp and self.exists():
            self.remove()