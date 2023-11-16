from terminal.models import default_storage
from .base import BaseStorageHandler

class FTPFileStorageHandler(BaseStorageHandler):
    NAME = 'FTP'

    def get_file_path(self, **kwargs):
        if False:
            print('Hello World!')
        return (self.obj.filepath, self.obj.filepath)

    def find_local(self):
        if False:
            while True:
                i = 10
        local_path = self.obj.filepath
        if default_storage.exists(local_path):
            url = default_storage.url(local_path)
            return (local_path, url)
        return (None, None)