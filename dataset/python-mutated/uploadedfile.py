"""
Classes representing uploaded files.
"""
import os
from io import BytesIO
from django.conf import settings
from django.core.files import temp as tempfile
from django.core.files.base import File
from django.core.files.utils import validate_file_name
__all__ = ('UploadedFile', 'TemporaryUploadedFile', 'InMemoryUploadedFile', 'SimpleUploadedFile')

class UploadedFile(File):
    """
    An abstract uploaded file (``TemporaryUploadedFile`` and
    ``InMemoryUploadedFile`` are the built-in concrete subclasses).

    An ``UploadedFile`` object behaves somewhat like a file object and
    represents some file data that the user submitted with a form.
    """

    def __init__(self, file=None, name=None, content_type=None, size=None, charset=None, content_type_extra=None):
        if False:
            while True:
                i = 10
        super().__init__(file, name)
        self.size = size
        self.content_type = content_type
        self.charset = charset
        self.content_type_extra = content_type_extra

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s: %s (%s)>' % (self.__class__.__name__, self.name, self.content_type)

    def _get_name(self):
        if False:
            print('Hello World!')
        return self._name

    def _set_name(self, name):
        if False:
            i = 10
            return i + 15
        if name is not None:
            name = os.path.basename(name)
            if len(name) > 255:
                (name, ext) = os.path.splitext(name)
                ext = ext[:255]
                name = name[:255 - len(ext)] + ext
            name = validate_file_name(name)
        self._name = name
    name = property(_get_name, _set_name)

class TemporaryUploadedFile(UploadedFile):
    """
    A file uploaded to a temporary location (i.e. stream-to-disk).
    """

    def __init__(self, name, content_type, size, charset, content_type_extra=None):
        if False:
            while True:
                i = 10
        (_, ext) = os.path.splitext(name)
        file = tempfile.NamedTemporaryFile(suffix='.upload' + ext, dir=settings.FILE_UPLOAD_TEMP_DIR)
        super().__init__(file, name, content_type, size, charset, content_type_extra)

    def temporary_file_path(self):
        if False:
            i = 10
            return i + 15
        'Return the full path of this file.'
        return self.file.name

    def close(self):
        if False:
            return 10
        try:
            return self.file.close()
        except FileNotFoundError:
            pass

class InMemoryUploadedFile(UploadedFile):
    """
    A file uploaded into memory (i.e. stream-to-memory).
    """

    def __init__(self, file, field_name, name, content_type, size, charset, content_type_extra=None):
        if False:
            print('Hello World!')
        super().__init__(file, name, content_type, size, charset, content_type_extra)
        self.field_name = field_name

    def open(self, mode=None):
        if False:
            return 10
        self.file.seek(0)
        return self

    def chunks(self, chunk_size=None):
        if False:
            for i in range(10):
                print('nop')
        self.file.seek(0)
        yield self.read()

    def multiple_chunks(self, chunk_size=None):
        if False:
            print('Hello World!')
        return False

class SimpleUploadedFile(InMemoryUploadedFile):
    """
    A simple representation of a file, which just has content, size, and a name.
    """

    def __init__(self, name, content, content_type='text/plain'):
        if False:
            i = 10
            return i + 15
        content = content or b''
        super().__init__(BytesIO(content), None, name, content_type, len(content), None, None)

    @classmethod
    def from_dict(cls, file_dict):
        if False:
            print('Hello World!')
        '\n        Create a SimpleUploadedFile object from a dictionary with keys:\n           - filename\n           - content-type\n           - content\n        '
        return cls(file_dict['filename'], file_dict['content'], file_dict.get('content-type', 'text/plain'))