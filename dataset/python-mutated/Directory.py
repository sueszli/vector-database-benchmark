import os
from coalib.parsing.Globbing import relative_recursive_glob
from coala_utils.decorators import generate_eq

@generate_eq('path', 'timestamp')
class Directory:
    """
    The ``Directory`` class acts as an interface to directories and
    provides useful information about them such as:

        * The directory path.
        * The parent directory path.
        * The last modified timestamp of the directory.
        * The children contained inside a directory.

    >>> import os
    >>> d = Directory('tests/io/DirectoryTestDir/')

    Get the number of files and sub-directories at the top level:

    >>> len(d.get_children())
    3

    Get the number of all the files and sub-directories recursively:
    >>> len(d.get_children_recursively())
    5

    Get the path of the ``Directory`` object:

    >>> os.path.basename(d.path).endswith('DirectoryTestDir')
    True

    Get the parent directory:

    >>> os.path.basename(d.parent).endswith('io')
    True

    Get the last modified timestamp of the directory:

    >>> d.timestamp == os.path.getmtime(d.path)
    True
    """

    def __init__(self, path):
        if False:
            print('Hello World!')
        self._path = os.path.abspath(path)
        self._parent = os.path.abspath(os.path.dirname(self._path))
        self._timestamp = os.path.getmtime(self._path)

    def get_children(self):
        if False:
            print('Hello World!')
        '\n        :return:\n            A list of all the files and sub-directories cotained at the\n            top level of the directory.\n        '
        return os.listdir(self._path)

    def get_children_recursively(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :return:\n            A list of all the files and sub-directories contained\n            inside a directory.\n        '
        return list(relative_recursive_glob(self._path, '**'))[1:]

    @property
    def path(self):
        if False:
            print('Hello World!')
        '\n        :return:\n            The directory path.\n        '
        return self._path

    @property
    def parent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :return:\n            The parent directory.\n        '
        return self._parent

    @property
    def timestamp(self):
        if False:
            return 10
        '\n        :return:\n            The last modified timestamp of the directory.\n        '
        return self._timestamp