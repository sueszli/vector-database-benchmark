from io import BytesIO
from os import fsdecode
import re
from .robottypes import is_bytes, is_pathlike, is_string
try:
    from xml.etree import cElementTree as ET
except ImportError:
    try:
        from xml.etree import ElementTree as ET
    except ImportError:
        raise ImportError('No valid ElementTree XML parser module found')

class ETSource:

    def __init__(self, source):
        if False:
            for i in range(10):
                print('nop')
        self._source = source
        self._opened = None

    def __enter__(self):
        if False:
            return 10
        self._opened = self._open_if_necessary(self._source)
        return self._opened or self._source

    def _open_if_necessary(self, source):
        if False:
            i = 10
            return i + 15
        if self._is_path(source) or self._is_already_open(source):
            return None
        if is_bytes(source):
            return BytesIO(source)
        encoding = self._find_encoding(source)
        return BytesIO(source.encode(encoding))

    def _is_path(self, source):
        if False:
            print('Hello World!')
        if is_pathlike(source):
            return True
        elif is_string(source):
            prefix = '<'
        elif is_bytes(source):
            prefix = b'<'
        else:
            return False
        return not source.lstrip().startswith(prefix)

    def _is_already_open(self, source):
        if False:
            i = 10
            return i + 15
        return not (is_string(source) or is_bytes(source))

    def _find_encoding(self, source):
        if False:
            while True:
                i = 10
        match = re.match('\\s*<\\?xml .*encoding=([\'\\"])(.*?)\\1.*\\?>', source)
        return match.group(2) if match else 'UTF-8'

    def __exit__(self, exc_type, exc_value, exc_trace):
        if False:
            print('Hello World!')
        if self._opened:
            self._opened.close()

    def __str__(self):
        if False:
            return 10
        source = self._source
        if self._is_path(source):
            return self._path_to_string(source)
        if hasattr(source, 'name'):
            return self._path_to_string(source.name)
        return '<in-memory file>'

    def _path_to_string(self, path):
        if False:
            while True:
                i = 10
        if is_pathlike(path):
            return str(path)
        if is_bytes(path):
            return fsdecode(path)
        return path