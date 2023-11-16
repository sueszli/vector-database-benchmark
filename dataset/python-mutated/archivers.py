from zipfile import ZipFile

class ZipFileArchiver(object):
    """
    An archiver used to generate .zip files.
    This wraps Python's built in :class:`zipfile.ZipFile`
    methods to operate exactly like :class:`tarfile.TarFile` does.
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Create a :class:`.ZipFileArchiver` instance. We create a new\n        :class:`zipfile.ZipFile` and store it to the ``zipfile`` member. \n        '
        self.zipfile = ZipFile(*args, **kwargs)

    @classmethod
    def open(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Open the archive. This must be a classmethod.\n        '
        return ZipFileArchiver(*args, **kwargs)

    def add(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Add file to the archive.\n        '
        self.zipfile.write(*args, **kwargs)

    def extractall(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Extract all files from the archive.\n        '
        self.zipfile.extractall(*args, **kwargs)

    def close(self):
        if False:
            i = 10
            return i + 15
        '\n        Close the archive.\n        '
        self.zipfile.close()