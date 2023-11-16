""" Contexts for *with* statement providing temporary directories
"""
import os
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp

@contextmanager
def tempdir():
    if False:
        i = 10
        return i + 15
    "Create and return a temporary directory. This has the same\n    behavior as mkdtemp but can be used as a context manager.\n\n    Upon exiting the context, the directory and everything contained\n    in it are removed.\n\n    Examples\n    --------\n    >>> import os\n    >>> with tempdir() as tmpdir:\n    ...     fname = os.path.join(tmpdir, 'example_file.txt')\n    ...     with open(fname, 'wt') as fobj:\n    ...         _ = fobj.write('a string\\n')\n    >>> os.path.exists(tmpdir)\n    False\n    "
    d = mkdtemp()
    yield d
    rmtree(d)

@contextmanager
def in_tempdir():
    if False:
        return 10
    " Create, return, and change directory to a temporary directory\n\n    Examples\n    --------\n    >>> import os\n    >>> my_cwd = os.getcwd()\n    >>> with in_tempdir() as tmpdir:\n    ...     _ = open('test.txt', 'wt').write('some text')\n    ...     assert os.path.isfile('test.txt')\n    ...     assert os.path.isfile(os.path.join(tmpdir, 'test.txt'))\n    >>> os.path.exists(tmpdir)\n    False\n    >>> os.getcwd() == my_cwd\n    True\n    "
    pwd = os.getcwd()
    d = mkdtemp()
    os.chdir(d)
    yield d
    os.chdir(pwd)
    rmtree(d)

@contextmanager
def in_dir(dir=None):
    if False:
        for i in range(10):
            print('nop')
    ' Change directory to given directory for duration of ``with`` block\n\n    Useful when you want to use `in_tempdir` for the final test, but\n    you are still debugging. For example, you may want to do this in the end:\n\n    >>> with in_tempdir() as tmpdir:\n    ...     # do something complicated which might break\n    ...     pass\n\n    But, indeed, the complicated thing does break, and meanwhile, the\n    ``in_tempdir`` context manager wiped out the directory with the\n    temporary files that you wanted for debugging. So, while debugging, you\n    replace with something like:\n\n    >>> with in_dir() as tmpdir: # Use working directory by default\n    ...     # do something complicated which might break\n    ...     pass\n\n    You can then look at the temporary file outputs to debug what is happening,\n    fix, and finally replace ``in_dir`` with ``in_tempdir`` again.\n    '
    cwd = os.getcwd()
    if dir is None:
        yield cwd
        return
    os.chdir(dir)
    yield dir
    os.chdir(cwd)