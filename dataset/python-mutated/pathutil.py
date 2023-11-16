"""
Provides some command utility functions.

TODO:
  matcher that ignores empty lines and whitespace and has contains comparison
"""
from __future__ import absolute_import, print_function
import os.path
import sys
import codecs

def realpath_with_context(path, context):
    if False:
        while True:
            i = 10
    '\n    Convert a path into its realpath:\n\n      * For relative path: use :attr:`context.workdir` as root directory\n      * For absolute path: Pass-through without any changes.\n\n    :param path: Filepath to convert (as string).\n    :param context: Behave context object (with :attr:`context.workdir`)\n    :return: Converted path.\n    '
    if not os.path.isabs(path):
        assert context.workdir
        path = os.path.join(context.workdir, os.path.normpath(path))
    return path

def posixpath_normpath(pathname):
    if False:
        print('Hello World!')
    '\n    Convert path into POSIX path:\n\n      * Normalize path\n      * Replace backslash with slash\n\n    :param pathname: Pathname (as string)\n    :return: Normalized POSIX path.\n    '
    backslash = '\\'
    pathname2 = os.path.normpath(pathname) or '.'
    if backslash in pathname2:
        pathname2 = pathname2.replace(backslash, '/')
    return pathname2

def ensure_makedirs(directory, max_iterations=3):
    if False:
        for i in range(10):
            print('nop')
    iteration = 0
    exception_text = None
    for iteration in range(max_iterations):
        try:
            os.makedirs(directory)
        except OSError as e:
            if iteration >= max_iterations:
                raise
            else:
                exception_text = '%s:%s' % (e.__class__.__name__, e)
        if os.path.isdir(directory):
            return
    assert os.path.isdir(directory), 'FAILED: ensure_makedirs(%r) (after %s iterations):\n%s' % (directory, max_iterations, exception_text)

def read_file_contents(filename, context=None, encoding=None):
    if False:
        print('Hello World!')
    filename_ = realpath_with_context(filename, context)
    assert os.path.exists(filename_)
    with open(filename_, 'r') as file_:
        file_contents = file_.read()
    return file_contents

def create_textfile_with_contents(filename, contents, encoding='utf-8'):
    if False:
        while True:
            i = 10
    '\n    Creates a textual file with the provided contents in the workdir.\n    Overwrites an existing file.\n    '
    ensure_directory_exists(os.path.dirname(filename))
    if os.path.exists(filename):
        os.remove(filename)
    outstream = codecs.open(filename, 'w', encoding)
    outstream.write(contents)
    if contents and (not contents.endswith('\n')):
        outstream.write('\n')
    outstream.flush()
    outstream.close()
    assert os.path.exists(filename), 'ENSURE file exists: %s' % filename

def ensure_file_exists(filename, context=None):
    if False:
        return 10
    real_filename = filename
    if context:
        real_filename = realpath_with_context(filename, context)
    if not os.path.exists(real_filename):
        create_textfile_with_contents(real_filename, '')
    assert os.path.exists(real_filename), 'ENSURE file exists: %s' % filename

def ensure_directory_exists(dirname, context=None):
    if False:
        for i in range(10):
            print('nop')
    'Ensures that a directory exits.\n    If it does not exist, it is automatically created.\n    '
    real_dirname = dirname
    if context:
        real_dirname = realpath_with_context(dirname, context)
    if not os.path.exists(real_dirname):
        mas_iterations = 2
        if sys.platform.startswith('win'):
            mas_iterations = 10
        ensure_makedirs(real_dirname, mas_iterations)
    assert os.path.exists(real_dirname), 'ENSURE dir exists: %s' % dirname
    assert os.path.isdir(real_dirname), 'ENSURE isa dir: %s' % dirname