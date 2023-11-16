import io
import os.path
from robot.errors import DataError
from .error import get_error_message
from .robottypes import is_pathlike

def file_writer(path=None, encoding='UTF-8', newline=None, usage=None):
    if False:
        print('Hello World!')
    if not path:
        return io.StringIO(newline=newline)
    if is_pathlike(path):
        path = str(path)
    create_destination_directory(path, usage)
    try:
        return io.open(path, 'w', encoding=encoding, newline=newline)
    except EnvironmentError:
        usage = '%s file' % usage if usage else 'file'
        raise DataError("Opening %s '%s' failed: %s" % (usage, path, get_error_message()))

def binary_file_writer(path=None):
    if False:
        return 10
    if path:
        if is_pathlike(path):
            path = str(path)
        return io.open(path, 'wb')
    f = io.BytesIO()
    getvalue = f.getvalue
    f.getvalue = lambda encoding='UTF-8': getvalue().decode(encoding)
    return f

def create_destination_directory(path, usage=None):
    if False:
        for i in range(10):
            print('nop')
    if is_pathlike(path):
        path = str(path)
    directory = os.path.dirname(path)
    if directory and (not os.path.exists(directory)):
        try:
            os.makedirs(directory, exist_ok=True)
        except EnvironmentError:
            usage = '%s directory' % usage if usage else 'directory'
            raise DataError("Creating %s '%s' failed: %s" % (usage, directory, get_error_message()))