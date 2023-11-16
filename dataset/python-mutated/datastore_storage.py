from collections import namedtuple
import re
from .exceptions import DataException

class CloseAfterUse(object):
    """
    Class that can be used to wrap data and a closer (cleanup code).
    This class should be used in a with statement and, when the with
    scope exits, `close` will be called on the closer object
    """

    def __init__(self, data, closer=None):
        if False:
            print('Hello World!')
        self.data = data
        self._closer = closer

    def __enter__(self):
        if False:
            return 10
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        if self._closer:
            self._closer.close()

class DataStoreStorage(object):
    """
    A DataStoreStorage defines the interface of communication between the
    higher-level datastores and the actual storage system.

    Both the ContentAddressedStore and the TaskDataStore use these methods to
    read/write/list from the actual storage system. These methods are meant to
    be low-level; they are in a class to provide better abstraction but this
    class itself is not meant to be initialized.
    """
    TYPE = None
    datastore_root = None
    path_rexp = None
    list_content_result = namedtuple('list_content_result', 'path is_file')

    def __init__(self, root=None):
        if False:
            for i in range(10):
                print('nop')
        self.datastore_root = root if root else self.datastore_root

    @classmethod
    def get_datastore_root_from_config(cls, echo, create_on_absent=True):
        if False:
            while True:
                i = 10
        "Returns a default choice for datastore_root from metaflow_config\n\n        Parameters\n        ----------\n        echo : function\n            Function to use to print out messages\n        create_on_absent : bool, optional\n            Create the datastore root if it doesn't exist, by default True\n        "
        raise NotImplementedError

    @classmethod
    def get_datastore_root_from_location(cls, path, flow_name):
        if False:
            for i in range(10):
                print('nop')
        'Extracts the datastore_root location from a path using\n        a content-addressed store.\n\n        NOTE: This leaks some detail of the content-addressed store so not ideal\n\n        This method will raise an exception if the flow_name is not as expected\n\n        Parameters\n        ----------\n        path : str\n            Location from which to extract the datastore root value\n        flow_name : str\n            Flow name (for verification purposes)\n\n        Returns\n        -------\n        str\n            The datastore_root value that can be used to initialize an instance\n            of this datastore storage.\n\n        Raises\n        ------\n        DataException\n            Raised if the path is not a valid path from this datastore.\n        '
        if cls.path_rexp is None:
            cls.path_rexp = re.compile(cls.path_join('(?P<root>.*)', '(?P<flow_name>[_a-zA-Z][_a-zA-Z0-9]+)', 'data', '(?P<init>[0-9a-f]{2})', '(?:r_)?(?P=init)[0-9a-f]{38}'))
        m = cls.path_rexp.match(path)
        if not m or m.group('flow_name') != flow_name:
            raise DataException("Location '%s' does not correspond to a valid location for flow '%s'." % (path, flow_name))
        return m.group('root')

    @classmethod
    def path_join(cls, *components):
        if False:
            for i in range(10):
                print('nop')
        if len(components) == 0:
            return ''
        component = components[0].rstrip('/')
        components = [component] + [c.strip('/') for c in components[1:]]
        return '/'.join(components)

    @classmethod
    def path_split(cls, path):
        if False:
            print('Hello World!')
        return path.split('/')

    @classmethod
    def basename(cls, path):
        if False:
            for i in range(10):
                print('nop')
        return path.split('/')[-1]

    @classmethod
    def dirname(cls, path):
        if False:
            print('Hello World!')
        return path.rsplit('/', 1)[0]

    def full_uri(self, path):
        if False:
            print('Hello World!')
        return self.path_join(self.datastore_root, path)

    def is_file(self, paths):
        if False:
            return 10
        '\n        Returns True or False depending on whether path refers to a valid\n        file-like object\n\n        This method returns False if path points to a directory\n\n        Parameters\n        ----------\n        path : List[string]\n            Path to the object\n\n        Returns\n        -------\n        List[bool]\n        '
        raise NotImplementedError

    def info_file(self, path):
        if False:
            while True:
                i = 10
        '\n        Returns a tuple where the first element is True or False depending on\n        whether path refers to a valid file-like object (like is_file) and the\n        second element is a dictionary of metadata associated with the file or\n        None if the file does not exist or there is no metadata.\n\n        Parameters\n        ----------\n        path : string\n            Path to the object\n\n        Returns\n        -------\n        tuple\n            (bool, dict)\n        '
        raise NotImplementedError

    def size_file(self, path):
        if False:
            while True:
                i = 10
        "\n        Returns file size at the indicated 'path', or None if file can not be found.\n\n        Parameters\n        ----------\n        path : string\n            Path to the object\n\n        Returns\n        -------\n        Optional\n            int\n        "
        raise NotImplementedError

    def list_content(self, paths):
        if False:
            return 10
        "\n        Lists the content of the datastore in the directory indicated by 'paths'.\n\n        This is similar to executing a 'ls'; it will only list the content one\n        level down and simply returns the paths to the elements present as well\n        as whether or not those elements are files (if not, they are further\n        directories that can be traversed)\n\n        The path returned always include the path passed in. As an example,\n        if your filesystem contains the files: A/b.txt A/c.txt and the directory\n        A/D, on return, you would get, for an input of ['A']:\n        [('A/b.txt', True), ('A/c.txt', True), ('A/D', False)]\n\n        Parameters\n        ----------\n        paths : List[string]\n            Directories to list\n\n        Returns\n        -------\n        List[list_content_result]\n            Content of the directory\n        "
        raise NotImplementedError

    def save_bytes(self, path_and_bytes_iter, overwrite=False, len_hint=0):
        if False:
            while True:
                i = 10
        '\n        Creates objects and stores them in the datastore.\n\n        If overwrite is False, any existing object will not be overwritten and\n        an error will be returned.\n\n        The objects are specified in an iterator over (path, obj) tuples where\n        the path is the path to store the object and the value is a file-like\n        object from which bytes can be read.\n\n        Parameters\n        ----------\n        path_and_bytes_iter : Iterator[(string, (RawIOBase|BufferedIOBase, metadata))]\n            Iterator over objects to store; the first element in the outermost\n            tuple is the path to store the bytes at. The second element in the\n            outermost tuple is either a RawIOBase or BufferedIOBase or a tuple\n            where the first element is a RawIOBase or BufferedIOBase and the\n            second element is a dictionary of metadata to associate with the\n            object.\n            Keys for the metadata must be ascii only string and elements\n            can be anything that can be converted to a string using json.dumps.\n            If you have no metadata, you can simply pass a RawIOBase or\n            BufferedIOBase.\n        overwrite : bool\n            True if the objects can be overwritten. Defaults to False.\n            Even when False, it is NOT an error condition to see an existing object.\n            Simply do not perform the upload operation.\n        len_hint : int\n            Estimated number of items produced by the iterator\n\n        Returns\n        -------\n        None\n        '
        raise NotImplementedError

    def load_bytes(self, keys):
        if False:
            return 10
        '\n        Gets objects from the datastore\n\n        Note that objects may be fetched in parallel so if order is important\n        for your consistency model, the caller is responsible for calling this\n        multiple times in the proper order.\n\n        Parameters\n        ----------\n        keys : List[string]\n            Keys to fetch\n\n        Returns\n        -------\n        CloseAfterUse :\n            A CloseAfterUse which should be used in a with statement. The data\n            in the CloseAfterUse will be an iterator over (key, file_path, metadata)\n            tuples. File_path and metadata will be None if the key was missing.\n            Metadata will be None if no metadata is present; otherwise it is\n            a dictionary of metadata associated with the object.\n\n            Note that the file at `file_path` may no longer be accessible outside\n            the scope of the returned object.\n\n            The order of items in the list is not to be relied on (ie: rely on the key\n            in the returned tuple and not on the order of the list). This function will,\n            however, return as many elements as passed in even in the presence of\n            duplicate keys.\n        '
        raise NotImplementedError