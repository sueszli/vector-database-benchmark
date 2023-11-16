"""
Provides some utility functions related to text processing.
"""
from __future__ import absolute_import, print_function
import codecs
import os
import sys
import six
BEHAVE_UNICODE_ERRORS = os.environ.get('BEHAVE_UNICODE_ERRORS', 'replace')

def make_indentation(indent_size, part=u' '):
    if False:
        return 10
    'Creates an indentation prefix string of the given size.'
    return indent_size * part

def indent(text, prefix):
    if False:
        i = 10
        return i + 15
    'Indent text or a number of text lines (with newline).\n\n    :param lines:  Text lines to indent (as string or list of strings).\n    :param prefix: Line prefix to use (as string).\n    :return: Indented text (as unicode string).\n    '
    lines = text
    newline = u''
    if isinstance(text, six.string_types):
        lines = text.splitlines(True)
    elif lines and (not lines[0].endswith('\n')):
        newline = u'\n'
    return newline.join([prefix + six.text_type(line) for line in lines])

def compute_words_maxsize(words):
    if False:
        print('Hello World!')
    'Compute the maximum word size from a list of words (or strings).\n\n    :param words: List of words (or strings) to use.\n    :return: Maximum size of all words.\n    '
    max_size = 0
    for word in words:
        if len(word) > max_size:
            max_size = len(word)
    return max_size

def is_ascii_encoding(encoding):
    if False:
        while True:
            i = 10
    'Checks if a given encoding is ASCII.'
    try:
        return codecs.lookup(encoding).name == 'ascii'
    except LookupError:
        return False

def select_best_encoding(outstream=None):
    if False:
        for i in range(10):
            print('nop')
    'Select the *best* encoding for an output stream/file.\n    Uses:\n    * ``outstream.encoding`` (if available)\n    * ``sys.getdefaultencoding()`` (otherwise)\n\n    Note: If encoding=ascii, uses encoding=UTF-8\n\n    :param outstream:  Output stream to select encoding for (or: stdout)\n    :return: Unicode encoding name (as string) to use (for output stream).\n    '
    outstream = outstream or sys.stdout
    encoding = getattr(outstream, 'encoding', None) or sys.getdefaultencoding()
    if is_ascii_encoding(encoding):
        return 'utf-8'
    return encoding

def text(value, encoding=None, errors=None):
    if False:
        print('Hello World!')
    'Convert into a unicode string.\n\n    :param value:  Value to convert into a unicode string (bytes, str, object).\n    :return: Unicode string\n\n    SYNDROMES:\n      * Convert object to unicode: Has only __str__() method (Python2)\n      * Windows: exception-traceback and encoding=unicode-escape are BAD\n      * exception-traceback w/ weird encoding or bytes\n\n    ALTERNATIVES:\n      * Use traceback2 for Python2: Provides unicode tracebacks\n    '
    if encoding is None:
        encoding = select_best_encoding()
    if errors is None:
        errors = BEHAVE_UNICODE_ERRORS
    if isinstance(value, six.text_type):
        return value
    elif isinstance(value, six.binary_type):
        try:
            return six.text_type(value, encoding, errors)
        except UnicodeError:
            return six.u(value)
    else:
        try:
            if six.PY2:
                try:
                    text2 = six.text_type(value)
                except UnicodeError as e:
                    data = str(value)
                    text2 = six.text_type(data, 'unicode-escape', 'replace')
            else:
                text2 = six.text_type(value)
        except UnicodeError as e:
            text2 = six.text_type(e)
        return text2

def to_texts(args, encoding=None, errors=None):
    if False:
        while True:
            i = 10
    'Process a list of string-like objects into list of unicode values.\n    Optionally converts binary text into unicode for each item.\n    \n    :return: List of text/unicode values.\n    '
    if encoding is None:
        encoding = select_best_encoding()
    return [text(arg, encoding, errors) for arg in args]

def ensure_stream_with_encoder(stream, encoding=None):
    if False:
        for i in range(10):
            print('nop')
    if not encoding:
        encoding = select_best_encoding(stream)
    if six.PY3:
        return stream
    elif hasattr(stream, 'stream'):
        return stream
    else:
        assert six.PY2
        stream = codecs.getwriter(encoding)(stream)
        return stream