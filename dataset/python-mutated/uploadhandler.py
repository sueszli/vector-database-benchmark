"""
Base file upload handler classes, and the built-in concrete subclasses
"""
import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
__all__ = ['UploadFileException', 'StopUpload', 'SkipFile', 'FileUploadHandler', 'TemporaryFileUploadHandler', 'MemoryFileUploadHandler', 'load_handler', 'StopFutureHandlers']

class UploadFileException(Exception):
    """
    Any error having to do with uploading files.
    """
    pass

class StopUpload(UploadFileException):
    """
    This exception is raised when an upload must abort.
    """

    def __init__(self, connection_reset=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        If ``connection_reset`` is ``True``, Django knows will halt the upload\n        without consuming the rest of the upload. This will cause the browser to\n        show a "connection reset" error.\n        '
        self.connection_reset = connection_reset

    def __str__(self):
        if False:
            print('Hello World!')
        if self.connection_reset:
            return 'StopUpload: Halt current upload.'
        else:
            return 'StopUpload: Consume request data, then halt.'

class SkipFile(UploadFileException):
    """
    This exception is raised by an upload handler that wants to skip a given file.
    """
    pass

class StopFutureHandlers(UploadFileException):
    """
    Upload handlers that have handled a file and do not want future handlers to
    run should raise this exception instead of returning None.
    """
    pass

class FileUploadHandler:
    """
    Base class for streaming upload handlers.
    """
    chunk_size = 64 * 2 ** 10

    def __init__(self, request=None):
        if False:
            for i in range(10):
                print('nop')
        self.file_name = None
        self.content_type = None
        self.content_length = None
        self.charset = None
        self.content_type_extra = None
        self.request = request

    def handle_raw_input(self, input_data, META, content_length, boundary, encoding=None):
        if False:
            i = 10
            return i + 15
        "\n        Handle the raw input from the client.\n\n        Parameters:\n\n            :input_data:\n                An object that supports reading via .read().\n            :META:\n                ``request.META``.\n            :content_length:\n                The (integer) value of the Content-Length header from the\n                client.\n            :boundary: The boundary from the Content-Type header. Be sure to\n                prepend two '--'.\n        "
        pass

    def new_file(self, field_name, file_name, content_type, content_length, charset=None, content_type_extra=None):
        if False:
            while True:
                i = 10
        "\n        Signal that a new file has been started.\n\n        Warning: As with any data from the client, you should not trust\n        content_length (and sometimes won't even get it).\n        "
        self.field_name = field_name
        self.file_name = file_name
        self.content_type = content_type
        self.content_length = content_length
        self.charset = charset
        self.content_type_extra = content_type_extra

    def receive_data_chunk(self, raw_data, start):
        if False:
            i = 10
            return i + 15
        '\n        Receive data from the streamed upload parser. ``start`` is the position\n        in the file of the chunk.\n        '
        raise NotImplementedError('subclasses of FileUploadHandler must provide a receive_data_chunk() method')

    def file_complete(self, file_size):
        if False:
            print('Hello World!')
        '\n        Signal that a file has completed. File size corresponds to the actual\n        size accumulated by all the chunks.\n\n        Subclasses should return a valid ``UploadedFile`` object.\n        '
        raise NotImplementedError('subclasses of FileUploadHandler must provide a file_complete() method')

    def upload_complete(self):
        if False:
            return 10
        '\n        Signal that the upload is complete. Subclasses should perform cleanup\n        that is necessary for this handler.\n        '
        pass

    def upload_interrupted(self):
        if False:
            print('Hello World!')
        '\n        Signal that the upload was interrupted. Subclasses should perform\n        cleanup that is necessary for this handler.\n        '
        pass

class TemporaryFileUploadHandler(FileUploadHandler):
    """
    Upload handler that streams data into a temporary file.
    """

    def new_file(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Create the file object to append to as data is coming in.\n        '
        super().new_file(*args, **kwargs)
        self.file = TemporaryUploadedFile(self.file_name, self.content_type, 0, self.charset, self.content_type_extra)

    def receive_data_chunk(self, raw_data, start):
        if False:
            for i in range(10):
                print('nop')
        self.file.write(raw_data)

    def file_complete(self, file_size):
        if False:
            while True:
                i = 10
        self.file.seek(0)
        self.file.size = file_size
        return self.file

    def upload_interrupted(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'file'):
            temp_location = self.file.temporary_file_path()
            try:
                self.file.close()
                os.remove(temp_location)
            except FileNotFoundError:
                pass

class MemoryFileUploadHandler(FileUploadHandler):
    """
    File upload handler to stream uploads into memory (used for small files).
    """

    def handle_raw_input(self, input_data, META, content_length, boundary, encoding=None):
        if False:
            print('Hello World!')
        '\n        Use the content_length to signal whether or not this handler should be\n        used.\n        '
        self.activated = content_length <= settings.FILE_UPLOAD_MAX_MEMORY_SIZE

    def new_file(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().new_file(*args, **kwargs)
        if self.activated:
            self.file = BytesIO()
            raise StopFutureHandlers()

    def receive_data_chunk(self, raw_data, start):
        if False:
            print('Hello World!')
        'Add the data to the BytesIO file.'
        if self.activated:
            self.file.write(raw_data)
        else:
            return raw_data

    def file_complete(self, file_size):
        if False:
            while True:
                i = 10
        'Return a file object if this handler is activated.'
        if not self.activated:
            return
        self.file.seek(0)
        return InMemoryUploadedFile(file=self.file, field_name=self.field_name, name=self.file_name, content_type=self.content_type, size=file_size, charset=self.charset, content_type_extra=self.content_type_extra)

def load_handler(path, *args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Given a path to a handler, return an instance of that handler.\n\n    E.g.::\n        >>> from django.http import HttpRequest\n        >>> request = HttpRequest()\n        >>> load_handler(\n        ...     'django.core.files.uploadhandler.TemporaryFileUploadHandler',\n        ...     request,\n        ... )\n        <TemporaryFileUploadHandler object at 0x...>\n    "
    return import_string(path)(*args, **kwargs)