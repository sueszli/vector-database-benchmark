"""
RetrieveTask module
"""
import os
import tempfile
from urllib.request import urlretrieve
from urllib.parse import urlparse
from .url import UrlTask

class RetrieveTask(UrlTask):
    """
    Task that retrieves urls (local or remote) to a local directory.
    """

    def register(self, directory=None, flatten=True):
        if False:
            i = 10
            return i + 15
        '\n        Adds retrieve parameters to task.\n\n        Args:\n            directory: local directory used to store retrieved files\n            flatten: flatten input directory structure, defaults to True\n        '
        if not directory:
            self.tempdir = tempfile.TemporaryDirectory()
            directory = self.tempdir.name
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.flatten = flatten

    def prepare(self, element):
        if False:
            return 10
        path = urlparse(element).path
        if self.flatten:
            path = os.path.join(self.directory, os.path.basename(path))
        else:
            path = os.path.join(self.directory, os.path.normpath(path.lstrip('/')))
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
        urlretrieve(element, path)
        return path