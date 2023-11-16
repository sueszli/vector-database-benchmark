class _ForceMultiPartDict(dict):
    """
    A dictionary that always evaluates as True.
    Allows us to force requests to use multipart encoding, even when no
    file parameters are passed.
    """

    def __bool__(self):
        if False:
            return 10
        return True

    def __nonzero__(self):
        if False:
            i = 10
            return i + 15
        return True

class BaseEncoder:
    media_type = None

    def encode(self, options, content):
        if False:
            return 10
        raise NotImplementedError()

class JSONEncoder:
    media_type = 'application/json'

    def encode(self, options, content):
        if False:
            for i in range(10):
                print('nop')
        options['json'] = content

class URLEncodedEncoder:
    media_type = 'application/x-www-form-urlencoded'

    def encode(self, options, content):
        if False:
            for i in range(10):
                print('nop')
        options['data'] = content

class MultiPartEncoder:
    media_type = 'multipart/form-data'

    def encode(self, options, content):
        if False:
            i = 10
            return i + 15
        data = {}
        files = _ForceMultiPartDict()
        for (key, value) in content.items():
            if self.is_file(value):
                files[key] = value
            else:
                data[key] = value
        options['data'] = data
        options['files'] = files

    def is_file(self, item):
        if False:
            while True:
                i = 10
        if hasattr(item, '__iter__') and (not isinstance(item, (str, list, tuple, dict))):
            return True
        return False