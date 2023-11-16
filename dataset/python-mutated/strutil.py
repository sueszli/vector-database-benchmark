from __future__ import absolute_import
import six

def unescape(s):
    if False:
        print('Hello World!')
    '\n    Action execution escapes escaped chars in result (i.e. \n is stored as \\n).\n    This function unescapes those chars.\n    '
    if isinstance(s, six.string_types):
        s = s.replace('\\n', '\n')
        s = s.replace('\\r', '\r')
        s = s.replace('\\"', '"')
    return s

def dedupe_newlines(s):
    if False:
        print('Hello World!')
    "yaml.safe_dump converts single newlines to double.\n\n    Since we're printing this output and not loading it, we should\n    deduplicate them.\n    "
    if isinstance(s, six.string_types):
        s = s.replace('\n\n', '\n')
    return s

def strip_carriage_returns(s):
    if False:
        while True:
            i = 10
    if isinstance(s, six.string_types):
        s = s.replace('\\r', '')
        s = s.replace('\r', '')
    return s