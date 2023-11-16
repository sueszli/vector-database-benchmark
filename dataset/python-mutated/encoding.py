"""A module for dealing with unknown string and environment encodings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import six

def Encode(string, encoding=None):
    if False:
        i = 10
        return i + 15
    'Encode the text string to a byte string.\n\n  Args:\n    string: str, The text string to encode.\n    encoding: The suggested encoding if known.\n\n  Returns:\n    str, The binary string.\n  '
    if string is None:
        return None
    if not six.PY2:
        return string
    if isinstance(string, six.binary_type):
        return string
    encoding = encoding or _GetEncoding()
    return string.encode(encoding)

def Decode(data, encoding=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns string with non-ascii characters decoded to UNICODE.\n\n  UTF-8, the suggested encoding, and the usual suspects will be attempted in\n  order.\n\n  Args:\n    data: A string or object that has str() and unicode() methods that may\n      contain an encoding incompatible with the standard output encoding.\n    encoding: The suggested encoding if known.\n\n  Returns:\n    A text string representing the decoded byte string.\n  '
    if data is None:
        return None
    if isinstance(data, six.text_type) or isinstance(data, six.binary_type):
        string = data
    else:
        try:
            string = six.text_type(data)
        except (TypeError, UnicodeError):
            string = str(data)
    if isinstance(string, six.text_type):
        return string
    try:
        return string.decode('ascii')
    except UnicodeError:
        pass
    if encoding:
        try:
            return string.decode(encoding)
        except UnicodeError:
            pass
    try:
        return string.decode('utf8')
    except UnicodeError:
        pass
    try:
        return string.decode(sys.getfilesystemencoding())
    except UnicodeError:
        pass
    try:
        return string.decode(sys.getdefaultencoding())
    except UnicodeError:
        pass
    return string.decode('iso-8859-1')

def GetEncodedValue(env, name, default=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns the decoded value of the env var name.\n\n  Args:\n    env: {str: str}, The env dict.\n    name: str, The env var name.\n    default: The value to return if name is not in env.\n\n  Returns:\n    The decoded value of the env var name.\n  '
    name = Encode(name)
    value = env.get(name)
    if value is None:
        return default
    return Decode(value)

def SetEncodedValue(env, name, value, encoding=None):
    if False:
        while True:
            i = 10
    'Sets the value of name in env to an encoded value.\n\n  Args:\n    env: {str: str}, The env dict.\n    name: str, The env var name.\n    value: str or unicode, The value for name. If None then name is removed from\n      env.\n    encoding: str, The encoding to use or None to try to infer it.\n  '
    name = Encode(name, encoding=encoding)
    if value is None:
        env.pop(name, None)
        return
    env[name] = Encode(value, encoding=encoding)

def EncodeEnv(env, encoding=None):
    if False:
        while True:
            i = 10
    'Encodes all the key value pairs in env in preparation for subprocess.\n\n  Args:\n    env: {str: str}, The environment you are going to pass to subprocess.\n    encoding: str, The encoding to use or None to use the default.\n\n  Returns:\n    {bytes: bytes}, The environment to pass to subprocess.\n  '
    encoding = encoding or _GetEncoding()
    return {Encode(k, encoding=encoding): Encode(v, encoding=encoding) for (k, v) in six.iteritems(env)}

def _GetEncoding():
    if False:
        for i in range(10):
            print('nop')
    'Gets the default encoding to use.'
    return sys.getfilesystemencoding() or sys.getdefaultencoding()