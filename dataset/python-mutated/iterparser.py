"""
This module includes a fast iterator-based XML parser.
"""
import contextlib
import io
import sys
from astropy.utils import data
__all__ = ['get_xml_iterator', 'get_xml_encoding', 'xml_readlines']

@contextlib.contextmanager
def _convert_to_fd_or_read_function(fd):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a function suitable for streaming input, or a file object.\n\n    This function is only useful if passing off to C code where:\n\n       - If it's a real file object, we want to use it as a real\n         C file object to avoid the Python overhead.\n\n       - If it's not a real file object, it's much handier to just\n         have a Python function to call.\n\n    This is somewhat quirky behavior, of course, which is why it is\n    private.  For a more useful version of similar behavior, see\n    `astropy.utils.misc.get_readable_fileobj`.\n\n    Parameters\n    ----------\n    fd : object\n        May be:\n\n            - a file object.  If the file is uncompressed, this raw\n              file object is returned verbatim.  Otherwise, the read\n              method is returned.\n\n            - a function that reads from a stream, in which case it is\n              returned verbatim.\n\n            - a file path, in which case it is opened.  Again, like a\n              file object, if it's uncompressed, a raw file object is\n              returned, otherwise its read method.\n\n            - an object with a :meth:`read` method, in which case that\n              method is returned.\n\n    Returns\n    -------\n    fd : context-dependent\n        See above.\n    "
    if callable(fd):
        yield fd
        return
    with data.get_readable_fileobj(fd, encoding='binary') as new_fd:
        if sys.platform.startswith('win'):
            yield new_fd.read
        elif isinstance(new_fd, io.FileIO):
            yield new_fd
        else:
            yield new_fd.read

def _fast_iterparse(fd, buffersize=2 ** 10):
    if False:
        for i in range(10):
            print('nop')
    from xml.parsers import expat
    if not callable(fd):
        read = fd.read
    else:
        read = fd
    queue = []
    text = []

    def start(name, attr):
        if False:
            return 10
        queue.append((True, name, attr, (parser.CurrentLineNumber, parser.CurrentColumnNumber)))
        del text[:]

    def end(name):
        if False:
            i = 10
            return i + 15
        queue.append((False, name, ''.join(text).strip(), (parser.CurrentLineNumber, parser.CurrentColumnNumber)))
    parser = expat.ParserCreate()
    parser.specified_attributes = True
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = text.append
    Parse = parser.Parse
    data = read(buffersize)
    while data:
        Parse(data, False)
        yield from queue
        del queue[:]
        data = read(buffersize)
    Parse('', True)
    yield from queue
_slow_iterparse = _fast_iterparse
try:
    from . import _iterparser
    _fast_iterparse = _iterparser.IterParser
except ImportError:
    pass

@contextlib.contextmanager
def get_xml_iterator(source, _debug_python_based_parser=False):
    if False:
        return 10
    "\n    Returns an iterator over the elements of an XML file.\n\n    The iterator doesn't ever build a tree, so it is much more memory\n    and time efficient than the alternative in ``cElementTree``.\n\n    Parameters\n    ----------\n    source : path-like, readable file-like, or callable\n        Handle that contains the data or function that reads it.\n        If a function or callable object, it must directly read from a stream.\n        Non-callable objects must define a ``read`` method.\n\n    Returns\n    -------\n    parts : iterator\n\n        The iterator returns 4-tuples (*start*, *tag*, *data*, *pos*):\n\n            - *start*: when `True` is a start element event, otherwise\n              an end element event.\n\n            - *tag*: The name of the element\n\n            - *data*: Depends on the value of *event*:\n\n                - if *start* == `True`, data is a dictionary of\n                  attributes\n\n                - if *start* == `False`, data is a string containing\n                  the text content of the element\n\n            - *pos*: Tuple (*line*, *col*) indicating the source of the\n              event.\n    "
    with _convert_to_fd_or_read_function(source) as fd:
        if _debug_python_based_parser:
            context = _slow_iterparse(fd)
        else:
            context = _fast_iterparse(fd)
        yield iter(context)

def get_xml_encoding(source):
    if False:
        return 10
    '\n    Determine the encoding of an XML file by reading its header.\n\n    Parameters\n    ----------\n    source : path-like, readable file-like, or callable\n        Handle that contains the data or function that reads it.\n        If a function or callable object, it must directly read from a stream.\n        Non-callable objects must define a ``read`` method.\n\n    Returns\n    -------\n    encoding : str\n    '
    with get_xml_iterator(source) as iterator:
        (start, tag, data, pos) = next(iterator)
        if not start or tag != 'xml':
            raise OSError('Invalid XML file')
    return data.get('encoding') or 'utf-8'

def xml_readlines(source):
    if False:
        i = 10
        return i + 15
    '\n    Get the lines from a given XML file.  Correctly determines the\n    encoding and always returns unicode.\n\n    Parameters\n    ----------\n    source : path-like, readable file-like, or callable\n        Handle that contains the data or function that reads it.\n        If a function or callable object, it must directly read from a stream.\n        Non-callable objects must define a ``read`` method.\n\n    Returns\n    -------\n    lines : list of unicode\n    '
    encoding = get_xml_encoding(source)
    with data.get_readable_fileobj(source, encoding=encoding) as input:
        input.seek(0)
        xml_lines = input.readlines()
    return xml_lines