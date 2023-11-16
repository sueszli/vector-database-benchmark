"""Utilities for the Response class."""
from falcon.util import uri
from falcon.util.misc import isascii
from falcon.util.misc import secure_filename

def header_property(name, doc, transform=None):
    if False:
        for i in range(10):
            print('nop')
    'Create a header getter/setter.\n\n    Args:\n        name: Header name, e.g., "Content-Type"\n        doc: Docstring for the property\n        transform: Transformation function to use when setting the\n            property. The value will be passed to the function, and\n            the function should return the transformed value to use\n            as the value of the header (default ``None``).\n\n    '
    normalized_name = name.lower()

    def fget(self):
        if False:
            return 10
        try:
            return self._headers[normalized_name]
        except KeyError:
            return None
    if transform is None:

        def fset(self, value):
            if False:
                print('Hello World!')
            if value is None:
                try:
                    del self._headers[normalized_name]
                except KeyError:
                    pass
            else:
                self._headers[normalized_name] = str(value)
    else:

        def fset(self, value):
            if False:
                i = 10
                return i + 15
            if value is None:
                try:
                    del self._headers[normalized_name]
                except KeyError:
                    pass
            else:
                self._headers[normalized_name] = transform(value)

    def fdel(self):
        if False:
            print('Hello World!')
        del self._headers[normalized_name]
    return property(fget, fset, fdel, doc)

def format_range(value):
    if False:
        print('Hello World!')
    'Format a range header tuple per the HTTP spec.\n\n    Args:\n        value: ``tuple`` passed to `req.range`\n    '
    if len(value) == 4:
        result = '%s %s-%s/%s' % (value[3], value[0], value[1], value[2])
    else:
        result = 'bytes %s-%s/%s' % (value[0], value[1], value[2])
    return result

def format_content_disposition(value, disposition_type='attachment'):
    if False:
        print('Hello World!')
    'Format a Content-Disposition header given a filename.'
    if isascii(value):
        return '%s; filename="%s"' % (disposition_type, value)
    return "%s; filename=%s; filename*=UTF-8''%s" % (disposition_type, secure_filename(value), uri.encode_value(value))

def format_etag_header(value):
    if False:
        for i in range(10):
            print('nop')
    'Format an ETag header, wrap it with " " in case of need.'
    if value[-1] != '"':
        value = '"' + value + '"'
    return value

def format_header_value_list(iterable):
    if False:
        for i in range(10):
            print('nop')
    'Join an iterable of strings with commas.'
    return ', '.join(iterable)

def is_ascii_encodable(s):
    if False:
        i = 10
        return i + 15
    'Check if argument encodes to ascii without error.'
    try:
        s.encode('ascii')
    except UnicodeEncodeError:
        return False
    except AttributeError:
        return False
    return True