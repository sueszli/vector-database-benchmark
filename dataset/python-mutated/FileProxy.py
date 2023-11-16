import logging
from os import path
from coala_utils.FileUtils import detect_encoding
from coala_utils.decorators import enforce_signature, generate_eq

@generate_eq('filename')
class FileProxy:
    """
    ``FileProxy`` is responsible for providing access to
    contents of files and also provides methods to update
    the in memory content register of the said file.

    >>> import logging
    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile(delete=False)
    >>> file.write(b'bears')
    5
    >>> file.close()

    To create a new file proxy instance of a file:

    >>> proxy = FileProxy(file.name, None, 'coala')
    >>> proxy.contents()
    'coala'

    >>> proxy.lines()
    ('coala',)


    You can replace the file contents in-memory by using
    the replace method on a file proxy. Version tracking
    is a simple way FileProxy provides to handle external
    incremental updates to the contents.

    >>> proxy.replace('coala-update', 1)
    True
    >>> proxy.contents()
    'coala-update'
    >>> proxy.version
    1

    File Proxy instances can also be initialized from files
    using FileProxy.from_file(). Binary files are also
    supported using from_file.

    >>> proxy2 = FileProxy.from_file(file.name, None)
    >>> proxy2.contents()
    'bears'
    """

    def __init__(self, filename, workspace=None, contents=''):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the FileProxy instance with the passed\n        parameters. A FileProxy instance always starts at\n        a fresh state with a negative version indicating\n        that no updating operation has been performed on it.\n\n        :param filename:\n            The name of the file to create a FileProxy of.\n            The filename is internally normcased.\n        :param workspace:\n            The workspace/project this file belongs to.\n            Can be None.\n        :param contents:\n            The contents of the file to initialize the\n            instance with. Integrity of the content or the\n            sync state is never checked during initialization.\n        '
        logging.debug(f'File proxy for {filename} created')
        if not path.isabs(filename) or filename.endswith(path.sep):
            raise ValueError('expecting absolute filename')
        self._version = -1
        self._contents = contents
        self._filename = path.normcase(filename)
        self._workspace = workspace and path.normcase(workspace)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :return:\n            Return a string representation of a file proxy\n            with information about its version and filename.\n        '
        return f'<FileProxy {self._filename}, {self._version}>'

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        '\n        :return:\n            Returns hash of the instance.\n        '
        return hash(self.filename)

    def replace(self, contents, version):
        if False:
            print('Hello World!')
        '\n        The method replaces the content of the proxy\n        entirely and does not push the change to the\n        history. It is similar to updating the proxy\n        with the range spanning to the entire content.\n\n        :param contents:\n            The new contents of the proxy.\n        :param version:\n            The version number proxy upgrades to after\n            the update. This needs to be greater than\n            the current version number.\n        :return:\n            Returns a boolean indicating the status of\n            the update.\n        '
        if version > self._version:
            self._contents = contents
            self._version = version
            logging.debug(f"File proxy for {self.filename} updated to version '{self.version}'.")
            return True
        logging.debug(f'Updating file proxy for {self.filename} failed.')
        return False

    def get_disk_contents(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :return:\n            Returns the contents of a copy of the file\n            on the disk. It might not be in sync with\n            the editor version of the file.\n        '
        with open(self.filename, 'r', encoding=detect_encoding(self.filename)) as disk:
            return disk.read()

    def contents(self):
        if False:
            i = 10
            return i + 15
        '\n        :return:\n            Returns the current contents of the proxy.\n        '
        return self._contents

    def lines(self):
        if False:
            while True:
                i = 10
        '\n        :return:\n            Returns the tuple of lines from the contents\n            of current proxy instance.\n        '
        return tuple(self.contents().splitlines(True))

    def clear(self):
        if False:
            return 10
        '\n        Clearing a proxy essentially means emptying the\n        contents of the proxy instance.\n        '
        self._contents = ''
        self._version = -1

    @property
    def filename(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :return:\n            Returns the complete normcased file name.\n        '
        return self._filename

    @property
    def workspace(self):
        if False:
            print('Hello World!')
        '\n        :return:\n            Returns the normcased workspace of the file.\n        '
        return self._workspace

    @property
    def version(self):
        if False:
            return 10
        '\n        :return:\n            Returns the current edit version of the file.\n        '
        return self._version

    @classmethod
    def from_file(cls, filename, workspace, binary=False):
        if False:
            i = 10
            return i + 15
        '\n        Construct a FileProxy instance from an existing\n        file on the drive.\n\n        :param filename:\n            The name of the file to be represented by\n            the proxy instance.\n        :param workspace:\n            The workspace the file belongs to. This can\n            be none representing that the the directory\n            server is currently serving from is the workspace.\n        :return:\n            Returns a FileProxy instance of the file with\n            the content synced from a disk copy.\n        '
        if not binary:
            with open(filename, 'r', encoding=detect_encoding(filename)) as reader:
                return cls(filename, workspace, reader.read())
        else:
            with open(filename, 'rb') as reader:
                return cls(filename, workspace, reader.read())

class FileProxyMap:
    """
    FileProxyMap handles a collection of proxies
    and provides a mechanism to reliably resolve
    missing proxies.

    >>> import tempfile
    >>> file = tempfile.NamedTemporaryFile(delete=False)
    >>> file.write(b'coala')
    5
    >>> file.close()

    You can initialize an empty proxy map using

    >>> proxymap = FileProxyMap()

    Or, you can pass it a list of FileProxy instances
    to build a map from. Addition/Recplacing of a file proxy
    to the map can be done using the add() method and likewise
    deletion can be done using the remove() method. FileProxyMap
    provides a handy function called `resolve()` that can retrive
    a file proxy if it is available in the map or build one for you.

    >>> proxy = proxymap.resolve(file.name, None, binary=False)
    >>> proxy.contents()
    'coala'

    >>> proxy2 = proxymap.resolve(file.name, None)
    >>> proxy.contents()
    'coala'

    >>> proxy == proxy2
    True
    """

    def __init__(self, file_proxies=[]):
        if False:
            print('Hello World!')
        '\n        :param file_proxies:\n            A list of FileProxy instances to initialize\n            the ProxyMap with.\n        '
        self._map = {proxy.filename: proxy for proxy in file_proxies}

    @enforce_signature
    def add(self, proxy: FileProxy, replace=False):
        if False:
            while True:
                i = 10
        '\n        Add a proxy instance to the map or replaces\n        optionally if it already exists.\n\n        :param proxy:\n            The proxy instance to register in the map.\n        :param replace:\n            A boolean flag indicating if the proxy should\n            replace an existing proxy of the same file.\n        :return:\n            Boolean true if registering of the proxy was\n            successful else false.\n        '
        if self._map.get(proxy.filename) is not None:
            if replace:
                self._map[proxy.filename] = proxy
                return True
            return False
        self._map[proxy.filename] = proxy
        return True

    def remove(self, filename):
        if False:
            print('Hello World!')
        '\n        Remove the proxy associated with a file from the\n        proxy map.\n\n        :param filename:\n            The name of the file to remove the proxy\n            associated with.\n        '
        filename = path.normcase(filename)
        if self.get(filename):
            del self._map[filename]

    def get(self, filename):
        if False:
            return 10
        '\n        :param filename:\n            The name of file to get the associated proxy instance.\n        :return:\n            A file proxy instance or None if not available.\n        '
        filename = path.normcase(filename)
        return self._map.get(filename)

    def resolve(self, filename, workspace=None, hard_sync=True, binary=False):
        if False:
            print('Hello World!')
        '\n        Resolve tries to find an available proxy or creates one\n        if there is no available proxy for the said file.\n\n        :param filename:\n            The filename to search for in the map or to create\n            a proxy instance using.\n        :param workspace:\n            Used in case the lookup fails and a new instance is\n            being initialized.\n        :hard_sync:\n            Boolean flag indicating if the file should be initialized\n            from the file on disk or fail otherwise.\n        :return:\n            Returns a proxy instance or raises associated exceptions.\n        '
        filename = path.normcase(filename)
        proxy = self.get(filename)
        if proxy is not None:
            return proxy
        try:
            proxy = FileProxy.from_file(filename, workspace, binary=binary)
        except (OSError, ValueError) as ex:
            if hard_sync:
                raise ex
            proxy = FileProxy(filename, workspace)
            self.add(proxy)
        return proxy

class FileDictGenerator:
    """
    FileDictGenerator is an interface definition class to provide
    structure of a file dict buildable classes.
    """

    def get_file_dict(self, filename_list, *args, **kargs):
        if False:
            while True:
                i = 10
        '\n        A method to\n\n        :param filename_list: A list of file names as strings to build\n                              the file dictionary from.\n        :return:              A dict mapping from file names to a tuple\n                              of lines of file contents.\n        '
        raise NotImplementedError('get_file_dict() needs to be implemented')