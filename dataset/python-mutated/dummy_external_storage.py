from django.core.files.base import File
from django.core.files.storage import FileSystemStorage, Storage
from django.utils.deconstruct import deconstructible

@deconstructible
class DummyExternalStorage(Storage):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.wrapped = FileSystemStorage(*args, **kwargs)

    def path(self, name):
        if False:
            while True:
                i = 10
        raise NotImplementedError("This backend doesn't support absolute paths.")

    def _open(self, name, mode='rb'):
        if False:
            print('Hello World!')
        return DummyExternalStorageFile(open(self.wrapped.path(name), mode))

    def _save(self, name, content):
        if False:
            print('Hello World!')
        file_pos = content.tell()
        if file_pos != 0:
            raise ValueError('Content file pointer should be at 0 - got %d instead' % file_pos)
        return self.wrapped._save(name, content)

    def delete(self, name):
        if False:
            while True:
                i = 10
        self.wrapped.delete(name)

    def exists(self, name):
        if False:
            return 10
        return self.wrapped.exists(name)

    def listdir(self, path):
        if False:
            for i in range(10):
                print('nop')
        return self.wrapped.listdir(path)

    def size(self, name):
        if False:
            i = 10
            return i + 15
        return self.wrapped.size(name)

    def url(self, name):
        if False:
            while True:
                i = 10
        return self.wrapped.url(name)

    def accessed_time(self, name):
        if False:
            i = 10
            return i + 15
        return self.wrapped.accessed_time(name)

    def created_time(self, name):
        if False:
            while True:
                i = 10
        return self.wrapped.created_time(name)

    def modified_time(self, name):
        if False:
            i = 10
            return i + 15
        return self.wrapped.modified_time(name)

class DummyExternalStorageError(Exception):
    pass

class DummyExternalStorageFile(File):

    def open(self, mode=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.closed:
            self.seek(0)
        else:
            raise ValueError('The file cannot be reopened.')

    def size(self):
        if False:
            while True:
                i = 10
        try:
            return super().size
        except Exception as e:
            raise DummyExternalStorageError(str(e))