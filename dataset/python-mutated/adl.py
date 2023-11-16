"""Utilities for working with Azure data lake storage"""
import re
from azure.datalake.store import core, lib

class ADL:
    """
    Represents an Azure Data Lake

    Methods
    -------
    The following are wrapped utilities for Azure storage:
    - read
    - listdir
    - write
    """

    def __init__(self):
        if False:
            return 10
        self.token = None

    @classmethod
    def _split_url(cls, url):
        if False:
            return 10
        match = re.match('adl://(.*)\\.azuredatalakestore\\.net\\/(.*)$', url)
        if not match:
            raise Exception(f"Invalid ADL url '{url}'")
        else:
            return (match.group(1), match.group(2))

    def _get_token(self):
        if False:
            return 10
        if self.token is None:
            self.token = lib.auth()
        return self.token

    def _create_adapter(self, store_name):
        if False:
            i = 10
            return i + 15
        return core.AzureDLFileSystem(self._get_token(), store_name=store_name)

    def listdir(self, url):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of the files under the specified path'
        (store_name, path) = self._split_url(url)
        adapter = self._create_adapter(store_name)
        return ['adl://{store_name}.azuredatalakestore.net/{path_to_child}'.format(store_name=store_name, path_to_child=path_to_child) for path_to_child in adapter.ls(path)]

    def read(self, url):
        if False:
            i = 10
            return i + 15
        'Read storage at a given url'
        (store_name, path) = self._split_url(url)
        adapter = self._create_adapter(store_name)
        lines = []
        with adapter.open(path) as f:
            for line in f:
                lines.append(line.decode())
        return lines

    def write(self, buf, url):
        if False:
            return 10
        'Write buffer to storage at a given url'
        (store_name, path) = self._split_url(url)
        adapter = self._create_adapter(store_name)
        with adapter.open(path, 'wb') as f:
            f.write(buf.encode())