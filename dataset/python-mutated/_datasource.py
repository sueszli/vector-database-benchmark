"""A file interface for handling local and remote data files.

The goal of datasource is to abstract some of the file system operations
when dealing with data files so the researcher doesn't have to know all the
low-level details.  Through datasource, a researcher can obtain and use a
file with one function call, regardless of location of the file.

DataSource is meant to augment standard python libraries, not replace them.
It should work seamlessly with standard file IO operations and the os
module.

DataSource files can originate locally or remotely:

- local files : '/home/guido/src/local/data.txt'
- URLs (http, ftp, ...) : 'http://www.scipy.org/not/real/data.txt'

DataSource files can also be compressed or uncompressed.  Currently only
gzip, bz2 and xz are supported.

Example::

    >>> # Create a DataSource, use os.curdir (default) for local storage.
    >>> from numpy import DataSource
    >>> ds = DataSource()
    >>>
    >>> # Open a remote file.
    >>> # DataSource downloads the file, stores it locally in:
    >>> #     './www.google.com/index.html'
    >>> # opens the file and returns a file object.
    >>> fp = ds.open('http://www.google.com/') # doctest: +SKIP
    >>>
    >>> # Use the file as you normally would
    >>> fp.read() # doctest: +SKIP
    >>> fp.close() # doctest: +SKIP

"""
import os
from .._utils import set_module
_open = open

def _check_mode(mode, encoding, newline):
    if False:
        print('Hello World!')
    'Check mode and that encoding and newline are compatible.\n\n    Parameters\n    ----------\n    mode : str\n        File open mode.\n    encoding : str\n        File encoding.\n    newline : str\n        Newline for text files.\n\n    '
    if 't' in mode:
        if 'b' in mode:
            raise ValueError('Invalid mode: %r' % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

class _FileOpeners:
    """
    Container for different methods to open (un-)compressed files.

    `_FileOpeners` contains a dictionary that holds one method for each
    supported file format. Attribute lookup is implemented in such a way
    that an instance of `_FileOpeners` itself can be indexed with the keys
    of that dictionary. Currently uncompressed files as well as files
    compressed with ``gzip``, ``bz2`` or ``xz`` compression are supported.

    Notes
    -----
    `_file_openers`, an instance of `_FileOpeners`, is made available for
    use in the `_datasource` module.

    Examples
    --------
    >>> import gzip
    >>> np.lib._datasource._file_openers.keys()
    [None, '.bz2', '.gz', '.xz', '.lzma']
    >>> np.lib._datasource._file_openers['.gz'] is gzip.open
    True

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._loaded = False
        self._file_openers = {None: open}

    def _load(self):
        if False:
            for i in range(10):
                print('nop')
        if self._loaded:
            return
        try:
            import bz2
            self._file_openers['.bz2'] = bz2.open
        except ImportError:
            pass
        try:
            import gzip
            self._file_openers['.gz'] = gzip.open
        except ImportError:
            pass
        try:
            import lzma
            self._file_openers['.xz'] = lzma.open
            self._file_openers['.lzma'] = lzma.open
        except (ImportError, AttributeError):
            pass
        self._loaded = True

    def keys(self):
        if False:
            print('Hello World!')
        "\n        Return the keys of currently supported file openers.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        keys : list\n            The keys are None for uncompressed files and the file extension\n            strings (i.e. ``'.gz'``, ``'.xz'``) for supported compression\n            methods.\n\n        "
        self._load()
        return list(self._file_openers.keys())

    def __getitem__(self, key):
        if False:
            return 10
        self._load()
        return self._file_openers[key]
_file_openers = _FileOpeners()

def open(path, mode='r', destpath=os.curdir, encoding=None, newline=None):
    if False:
        return 10
    "\n    Open `path` with `mode` and return the file object.\n\n    If ``path`` is an URL, it will be downloaded, stored in the\n    `DataSource` `destpath` directory and opened from there.\n\n    Parameters\n    ----------\n    path : str or pathlib.Path\n        Local file path or URL to open.\n    mode : str, optional\n        Mode to open `path`. Mode 'r' for reading, 'w' for writing, 'a' to\n        append. Available modes depend on the type of object specified by\n        path.  Default is 'r'.\n    destpath : str, optional\n        Path to the directory where the source file gets downloaded to for\n        use.  If `destpath` is None, a temporary directory will be created.\n        The default path is the current directory.\n    encoding : {None, str}, optional\n        Open text file with given encoding. The default encoding will be\n        what `open` uses.\n    newline : {None, str}, optional\n        Newline to use when reading text file.\n\n    Returns\n    -------\n    out : file object\n        The opened file.\n\n    Notes\n    -----\n    This is a convenience function that instantiates a `DataSource` and\n    returns the file object from ``DataSource.open(path)``.\n\n    "
    ds = DataSource(destpath)
    return ds.open(path, mode, encoding=encoding, newline=newline)

@set_module('numpy.lib.npyio')
class DataSource:
    """
    DataSource(destpath='.')

    A generic data source file (file, http, ftp, ...).

    DataSources can be local files or remote files/URLs.  The files may
    also be compressed or uncompressed. DataSource hides some of the
    low-level details of downloading the file, allowing you to simply pass
    in a valid file path (or URL) and obtain a file object.

    Parameters
    ----------
    destpath : str or None, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.

    Notes
    -----
    URLs require a scheme string (``http://``) to be used, without it they
    will fail::

        >>> repos = np.lib.npyio.DataSource()
        >>> repos.exists('www.google.com/index.html')
        False
        >>> repos.exists('http://www.google.com/index.html')
        True

    Temporary directories are deleted when the DataSource is deleted.

    Examples
    --------
    ::

        >>> ds = np.lib.npyio.DataSource('/home/guido')
        >>> urlname = 'http://www.google.com/'
        >>> gfile = ds.open('http://www.google.com/')
        >>> ds.abspath(urlname)
        '/home/guido/www.google.com/index.html'

        >>> ds = np.lib.npyio.DataSource(None)  # use with temporary file
        >>> ds.open('/home/guido/foobar.txt')
        <open file '/home/guido.foobar.txt', mode 'r' at 0x91d4430>
        >>> ds.abspath('/home/guido/foobar.txt')
        '/tmp/.../home/guido/foobar.txt'

    """

    def __init__(self, destpath=os.curdir):
        if False:
            i = 10
            return i + 15
        'Create a DataSource with a local path at destpath.'
        if destpath:
            self._destpath = os.path.abspath(destpath)
            self._istmpdest = False
        else:
            import tempfile
            self._destpath = tempfile.mkdtemp()
            self._istmpdest = True

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_istmpdest') and self._istmpdest:
            import shutil
            shutil.rmtree(self._destpath)

    def _iszip(self, filename):
        if False:
            print('Hello World!')
        'Test if the filename is a zip file by looking at the file extension.\n\n        '
        (fname, ext) = os.path.splitext(filename)
        return ext in _file_openers.keys()

    def _iswritemode(self, mode):
        if False:
            for i in range(10):
                print('nop')
        'Test if the given mode will open a file for writing.'
        _writemodes = ('w', '+')
        for c in mode:
            if c in _writemodes:
                return True
        return False

    def _splitzipext(self, filename):
        if False:
            for i in range(10):
                print('nop')
        'Split zip extension from filename and return filename.\n\n        Returns\n        -------\n        base, zip_ext : {tuple}\n\n        '
        if self._iszip(filename):
            return os.path.splitext(filename)
        else:
            return (filename, None)

    def _possible_names(self, filename):
        if False:
            for i in range(10):
                print('nop')
        'Return a tuple containing compressed filename variations.'
        names = [filename]
        if not self._iszip(filename):
            for zipext in _file_openers.keys():
                if zipext:
                    names.append(filename + zipext)
        return names

    def _isurl(self, path):
        if False:
            return 10
        'Test if path is a net location.  Tests the scheme and netloc.'
        from urllib.parse import urlparse
        (scheme, netloc, upath, uparams, uquery, ufrag) = urlparse(path)
        return bool(scheme and netloc)

    def _cache(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Cache the file specified by path.\n\n        Creates a copy of the file in the datasource cache.\n\n        '
        import shutil
        from urllib.request import urlopen
        upath = self.abspath(path)
        if not os.path.exists(os.path.dirname(upath)):
            os.makedirs(os.path.dirname(upath))
        if self._isurl(path):
            with urlopen(path) as openedurl:
                with _open(upath, 'wb') as f:
                    shutil.copyfileobj(openedurl, f)
        else:
            shutil.copyfile(path, upath)
        return upath

    def _findfile(self, path):
        if False:
            print('Hello World!')
        'Searches for ``path`` and returns full path if found.\n\n        If path is an URL, _findfile will cache a local copy and return the\n        path to the cached file.  If path is a local file, _findfile will\n        return a path to that local file.\n\n        The search will include possible compressed versions of the file\n        and return the first occurrence found.\n\n        '
        if not self._isurl(path):
            filelist = self._possible_names(path)
            filelist += self._possible_names(self.abspath(path))
        else:
            filelist = self._possible_names(self.abspath(path))
            filelist = filelist + self._possible_names(path)
        for name in filelist:
            if self.exists(name):
                if self._isurl(name):
                    name = self._cache(name)
                return name
        return None

    def abspath(self, path):
        if False:
            print('Hello World!')
        '\n        Return absolute path of file in the DataSource directory.\n\n        If `path` is an URL, then `abspath` will return either the location\n        the file exists locally or the location it would exist when opened\n        using the `open` method.\n\n        Parameters\n        ----------\n        path : str or pathlib.Path\n            Can be a local file or a remote URL.\n\n        Returns\n        -------\n        out : str\n            Complete path, including the `DataSource` destination directory.\n\n        Notes\n        -----\n        The functionality is based on `os.path.abspath`.\n\n        '
        from urllib.parse import urlparse
        splitpath = path.split(self._destpath, 2)
        if len(splitpath) > 1:
            path = splitpath[1]
        (scheme, netloc, upath, uparams, uquery, ufrag) = urlparse(path)
        netloc = self._sanitize_relative_path(netloc)
        upath = self._sanitize_relative_path(upath)
        return os.path.join(self._destpath, netloc, upath)

    def _sanitize_relative_path(self, path):
        if False:
            i = 10
            return i + 15
        'Return a sanitised relative path for which\n        os.path.abspath(os.path.join(base, path)).startswith(base)\n        '
        last = None
        path = os.path.normpath(path)
        while path != last:
            last = path
            path = path.lstrip(os.sep).lstrip('/')
            path = path.lstrip(os.pardir).lstrip('..')
            (drive, path) = os.path.splitdrive(path)
        return path

    def exists(self, path):
        if False:
            i = 10
            return i + 15
        "\n        Test if path exists.\n\n        Test if `path` exists as (and in this order):\n\n        - a local file.\n        - a remote URL that has been downloaded and stored locally in the\n          `DataSource` directory.\n        - a remote URL that has not been downloaded, but is valid and\n          accessible.\n\n        Parameters\n        ----------\n        path : str or pathlib.Path\n            Can be a local file or a remote URL.\n\n        Returns\n        -------\n        out : bool\n            True if `path` exists.\n\n        Notes\n        -----\n        When `path` is an URL, `exists` will return True if it's either\n        stored locally in the `DataSource` directory, or is a valid remote\n        URL.  `DataSource` does not discriminate between the two, the file\n        is accessible if it exists in either location.\n\n        "
        if os.path.exists(path):
            return True
        from urllib.request import urlopen
        from urllib.error import URLError
        upath = self.abspath(path)
        if os.path.exists(upath):
            return True
        if self._isurl(path):
            try:
                netfile = urlopen(path)
                netfile.close()
                del netfile
                return True
            except URLError:
                return False
        return False

    def open(self, path, mode='r', encoding=None, newline=None):
        if False:
            print('Hello World!')
        "\n        Open and return file-like object.\n\n        If `path` is an URL, it will be downloaded, stored in the\n        `DataSource` directory and opened from there.\n\n        Parameters\n        ----------\n        path : str or pathlib.Path\n            Local file path or URL to open.\n        mode : {'r', 'w', 'a'}, optional\n            Mode to open `path`.  Mode 'r' for reading, 'w' for writing,\n            'a' to append. Available modes depend on the type of object\n            specified by `path`. Default is 'r'.\n        encoding : {None, str}, optional\n            Open text file with given encoding. The default encoding will be\n            what `open` uses.\n        newline : {None, str}, optional\n            Newline to use when reading text file.\n\n        Returns\n        -------\n        out : file object\n            File object.\n\n        "
        if self._isurl(path) and self._iswritemode(mode):
            raise ValueError('URLs are not writeable')
        found = self._findfile(path)
        if found:
            (_fname, ext) = self._splitzipext(found)
            if ext == 'bz2':
                mode.replace('+', '')
            return _file_openers[ext](found, mode=mode, encoding=encoding, newline=newline)
        else:
            raise FileNotFoundError(f'{path} not found.')

class Repository(DataSource):
    """
    Repository(baseurl, destpath='.')

    A data repository where multiple DataSource's share a base
    URL/directory.

    `Repository` extends `DataSource` by prepending a base URL (or
    directory) to all the files it handles. Use `Repository` when you will
    be working with multiple files from one base URL.  Initialize
    `Repository` with the base URL, then refer to each file by its filename
    only.

    Parameters
    ----------
    baseurl : str
        Path to the local directory or remote location that contains the
        data files.
    destpath : str or None, optional
        Path to the directory where the source file gets downloaded to for
        use.  If `destpath` is None, a temporary directory will be created.
        The default path is the current directory.

    Examples
    --------
    To analyze all files in the repository, do something like this
    (note: this is not self-contained code)::

        >>> repos = np.lib._datasource.Repository('/home/user/data/dir/')
        >>> for filename in filelist:
        ...     fp = repos.open(filename)
        ...     fp.analyze()
        ...     fp.close()

    Similarly you could use a URL for a repository::

        >>> repos = np.lib._datasource.Repository('http://www.xyz.edu/data')

    """

    def __init__(self, baseurl, destpath=os.curdir):
        if False:
            return 10
        'Create a Repository with a shared url or directory of baseurl.'
        DataSource.__init__(self, destpath=destpath)
        self._baseurl = baseurl

    def __del__(self):
        if False:
            i = 10
            return i + 15
        DataSource.__del__(self)

    def _fullpath(self, path):
        if False:
            print('Hello World!')
        'Return complete path for path.  Prepends baseurl if necessary.'
        splitpath = path.split(self._baseurl, 2)
        if len(splitpath) == 1:
            result = os.path.join(self._baseurl, path)
        else:
            result = path
        return result

    def _findfile(self, path):
        if False:
            print('Hello World!')
        'Extend DataSource method to prepend baseurl to ``path``.'
        return DataSource._findfile(self, self._fullpath(path))

    def abspath(self, path):
        if False:
            print('Hello World!')
        '\n        Return absolute path of file in the Repository directory.\n\n        If `path` is an URL, then `abspath` will return either the location\n        the file exists locally or the location it would exist when opened\n        using the `open` method.\n\n        Parameters\n        ----------\n        path : str or pathlib.Path\n            Can be a local file or a remote URL. This may, but does not\n            have to, include the `baseurl` with which the `Repository` was\n            initialized.\n\n        Returns\n        -------\n        out : str\n            Complete path, including the `DataSource` destination directory.\n\n        '
        return DataSource.abspath(self, self._fullpath(path))

    def exists(self, path):
        if False:
            return 10
        "\n        Test if path exists prepending Repository base URL to path.\n\n        Test if `path` exists as (and in this order):\n\n        - a local file.\n        - a remote URL that has been downloaded and stored locally in the\n          `DataSource` directory.\n        - a remote URL that has not been downloaded, but is valid and\n          accessible.\n\n        Parameters\n        ----------\n        path : str or pathlib.Path\n            Can be a local file or a remote URL. This may, but does not\n            have to, include the `baseurl` with which the `Repository` was\n            initialized.\n\n        Returns\n        -------\n        out : bool\n            True if `path` exists.\n\n        Notes\n        -----\n        When `path` is an URL, `exists` will return True if it's either\n        stored locally in the `DataSource` directory, or is a valid remote\n        URL.  `DataSource` does not discriminate between the two, the file\n        is accessible if it exists in either location.\n\n        "
        return DataSource.exists(self, self._fullpath(path))

    def open(self, path, mode='r', encoding=None, newline=None):
        if False:
            return 10
        "\n        Open and return file-like object prepending Repository base URL.\n\n        If `path` is an URL, it will be downloaded, stored in the\n        DataSource directory and opened from there.\n\n        Parameters\n        ----------\n        path : str or pathlib.Path\n            Local file path or URL to open. This may, but does not have to,\n            include the `baseurl` with which the `Repository` was\n            initialized.\n        mode : {'r', 'w', 'a'}, optional\n            Mode to open `path`.  Mode 'r' for reading, 'w' for writing,\n            'a' to append. Available modes depend on the type of object\n            specified by `path`. Default is 'r'.\n        encoding : {None, str}, optional\n            Open text file with given encoding. The default encoding will be\n            what `open` uses.\n        newline : {None, str}, optional\n            Newline to use when reading text file.\n\n        Returns\n        -------\n        out : file object\n            File object.\n\n        "
        return DataSource.open(self, self._fullpath(path), mode, encoding=encoding, newline=newline)

    def listdir(self):
        if False:
            return 10
        '\n        List files in the source Repository.\n\n        Returns\n        -------\n        files : list of str or pathlib.Path\n            List of file names (not containing a directory part).\n\n        Notes\n        -----\n        Does not currently work for remote repositories.\n\n        '
        if self._isurl(self._baseurl):
            raise NotImplementedError('Directory listing of URLs, not supported yet.')
        else:
            return os.listdir(self._baseurl)