"""Implements the majority of smart_open's top-level API.

The main functions are:

  * ``parse_uri()``
  * ``open()``

"""
import collections
import io
import locale
import logging
import os
import os.path as P
import pathlib
import urllib.parse
import warnings
import smart_open.local_file as so_file
import smart_open.compression as so_compression
from smart_open import doctools
from smart_open import transport
from smart_open.compression import register_compressor
from smart_open.utils import check_kwargs as _check_kwargs
from smart_open.utils import inspect_kwargs as _inspect_kwargs
logger = logging.getLogger(__name__)
DEFAULT_ENCODING = locale.getpreferredencoding(do_setlocale=False)

def _sniff_scheme(uri_as_string):
    if False:
        i = 10
        return i + 15
    'Returns the scheme of the URL only, as a string.'
    if os.name == 'nt' and '://' not in uri_as_string:
        uri_as_string = 'file://' + uri_as_string
    return urllib.parse.urlsplit(uri_as_string).scheme

def parse_uri(uri_as_string):
    if False:
        return 10
    '\n    Parse the given URI from a string.\n\n    Parameters\n    ----------\n    uri_as_string: str\n        The URI to parse.\n\n    Returns\n    -------\n    collections.namedtuple\n        The parsed URI.\n\n    Notes\n    -----\n    smart_open/doctools.py magic goes here\n    '
    scheme = _sniff_scheme(uri_as_string)
    submodule = transport.get_transport(scheme)
    as_dict = submodule.parse_uri(uri_as_string)
    Uri = collections.namedtuple('Uri', sorted(as_dict.keys()))
    return Uri(**as_dict)
_parse_uri = parse_uri
_builtin_open = open

def open(uri, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None, compression=so_compression.INFER_FROM_EXTENSION, transport_params=None):
    if False:
        while True:
            i = 10
    'Open the URI object, returning a file-like object.\n\n    The URI is usually a string in a variety of formats.\n    For a full list of examples, see the :func:`parse_uri` function.\n\n    The URI may also be one of:\n\n    - an instance of the pathlib.Path class\n    - a stream (anything that implements io.IOBase-like functionality)\n\n    Parameters\n    ----------\n    uri: str or object\n        The object to open.\n    mode: str, optional\n        Mimicks built-in open parameter of the same name.\n    buffering: int, optional\n        Mimicks built-in open parameter of the same name.\n    encoding: str, optional\n        Mimicks built-in open parameter of the same name.\n    errors: str, optional\n        Mimicks built-in open parameter of the same name.\n    newline: str, optional\n        Mimicks built-in open parameter of the same name.\n    closefd: boolean, optional\n        Mimicks built-in open parameter of the same name.  Ignored.\n    opener: object, optional\n        Mimicks built-in open parameter of the same name.  Ignored.\n    compression: str, optional (see smart_open.compression.get_supported_compression_types)\n        Explicitly specify the compression/decompression behavior.\n    transport_params: dict, optional\n        Additional parameters for the transport layer (see notes below).\n\n    Returns\n    -------\n    A file-like object.\n\n    Notes\n    -----\n    smart_open has several implementations for its transport layer (e.g. S3, HTTP).\n    Each transport layer has a different set of keyword arguments for overriding\n    default behavior.  If you specify a keyword argument that is *not* supported\n    by the transport layer being used, smart_open will ignore that argument and\n    log a warning message.\n\n    smart_open/doctools.py magic goes here\n\n    See Also\n    --------\n    - `Standard library reference <https://docs.python.org/3.7/library/functions.html#open>`__\n    - `smart_open README.rst\n      <https://github.com/RaRe-Technologies/smart_open/blob/master/README.rst>`__\n\n    '
    logger.debug('%r', locals())
    if not isinstance(mode, str):
        raise TypeError('mode should be a string')
    if compression not in so_compression.get_supported_compression_types():
        raise ValueError(f'invalid compression type: {compression}')
    if transport_params is None:
        transport_params = {}
    fobj = _shortcut_open(uri, mode, compression=compression, buffering=buffering, encoding=encoding, errors=errors, newline=newline)
    if fobj is not None:
        return fobj
    if encoding is not None and 'b' in mode:
        mode = mode.replace('b', '')
    if isinstance(uri, pathlib.Path):
        uri = str(uri)
    explicit_encoding = encoding
    encoding = explicit_encoding if explicit_encoding else DEFAULT_ENCODING
    try:
        binary_mode = _get_binary_mode(mode)
    except ValueError as ve:
        raise NotImplementedError(ve.args[0])
    binary = _open_binary_stream(uri, binary_mode, transport_params)
    decompressed = so_compression.compression_wrapper(binary, binary_mode, compression)
    if 'b' not in mode or explicit_encoding is not None:
        decoded = _encoding_wrapper(decompressed, mode, encoding=encoding, errors=errors, newline=newline)
    else:
        decoded = decompressed
    if decoded != binary:
        promoted_attrs = ['to_boto3']
        for attr in promoted_attrs:
            try:
                setattr(decoded, attr, getattr(binary, attr))
            except AttributeError:
                pass
    return decoded

def _get_binary_mode(mode_str):
    if False:
        print('Hello World!')
    mode = list(mode_str)
    binmode = []
    if 't' in mode and 'b' in mode:
        raise ValueError("can't have text and binary mode at once")
    counts = [mode.count(x) for x in 'rwa']
    if sum(counts) > 1:
        raise ValueError('must have exactly one of create/read/write/append mode')

    def transfer(char):
        if False:
            for i in range(10):
                print('nop')
        binmode.append(mode.pop(mode.index(char)))
    if 'a' in mode:
        transfer('a')
    elif 'w' in mode:
        transfer('w')
    elif 'r' in mode:
        transfer('r')
    else:
        raise ValueError('Must have exactly one of create/read/write/append mode and at most one plus')
    if 'b' in mode:
        transfer('b')
    elif 't' in mode:
        mode.pop(mode.index('t'))
        binmode.append('b')
    else:
        binmode.append('b')
    if '+' in mode:
        transfer('+')
    if mode:
        raise ValueError('invalid mode: %r' % mode_str)
    return ''.join(binmode)

def _shortcut_open(uri, mode, compression, buffering=-1, encoding=None, errors=None, newline=None):
    if False:
        return 10
    'Try to open the URI using the standard library io.open function.\n\n    This can be much faster than the alternative of opening in binary mode and\n    then decoding.\n\n    This is only possible under the following conditions:\n\n        1. Opening a local file; and\n        2. Compression is disabled\n\n    If it is not possible to use the built-in open for the specified URI, returns None.\n\n    :param str uri: A string indicating what to open.\n    :param str mode: The mode to pass to the open function.\n    :param str compression: The compression type selected.\n    :returns: The opened file\n    :rtype: file\n    '
    if not isinstance(uri, str):
        return None
    scheme = _sniff_scheme(uri)
    if scheme not in (transport.NO_SCHEME, so_file.SCHEME):
        return None
    local_path = so_file.extract_local_path(uri)
    if compression == so_compression.INFER_FROM_EXTENSION:
        (_, extension) = P.splitext(local_path)
        if extension in so_compression.get_supported_extensions():
            return None
    elif compression != so_compression.NO_COMPRESSION:
        return None
    open_kwargs = {}
    if encoding is not None:
        open_kwargs['encoding'] = encoding
        mode = mode.replace('b', '')
    if newline is not None:
        open_kwargs['newline'] = newline
    if errors and 'b' not in mode:
        open_kwargs['errors'] = errors
    return _builtin_open(local_path, mode, buffering=buffering, **open_kwargs)

def _open_binary_stream(uri, mode, transport_params):
    if False:
        for i in range(10):
            print('nop')
    'Open an arbitrary URI in the specified binary mode.\n\n    Not all modes are supported for all protocols.\n\n    :arg uri: The URI to open.  May be a string, or something else.\n    :arg str mode: The mode to open with.  Must be rb, wb or ab.\n    :arg transport_params: Keyword argumens for the transport layer.\n    :returns: A named file object\n    :rtype: file-like object with a .name attribute\n    '
    if mode not in ('rb', 'rb+', 'wb', 'wb+', 'ab', 'ab+'):
        raise NotImplementedError('unsupported mode: %r' % mode)
    if isinstance(uri, int):
        fobj = _builtin_open(uri, mode, closefd=False)
        return fobj
    if not isinstance(uri, str):
        raise TypeError("don't know how to handle uri %s" % repr(uri))
    scheme = _sniff_scheme(uri)
    submodule = transport.get_transport(scheme)
    fobj = submodule.open_uri(uri, mode, transport_params)
    if not hasattr(fobj, 'name'):
        fobj.name = uri
    return fobj

def _encoding_wrapper(fileobj, mode, encoding=None, errors=None, newline=None):
    if False:
        return 10
    'Decode bytes into text, if necessary.\n\n    If mode specifies binary access, does nothing, unless the encoding is\n    specified.  A non-null encoding implies text mode.\n\n    :arg fileobj: must quack like a filehandle object.\n    :arg str mode: is the mode which was originally requested by the user.\n    :arg str encoding: The text encoding to use.  If mode is binary, overrides mode.\n    :arg str errors: The method to use when handling encoding/decoding errors.\n    :returns: a file object\n    '
    logger.debug('encoding_wrapper: %r', locals())
    if 'b' in mode and encoding is None:
        return fileobj
    if encoding is None:
        encoding = DEFAULT_ENCODING
    fileobj = io.TextIOWrapper(fileobj, encoding=encoding, errors=errors, newline=newline, write_through=True)
    return fileobj

class patch_pathlib(object):
    """Replace `Path.open` with `smart_open.open`"""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.old_impl = _patch_pathlib(open)

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        _patch_pathlib(self.old_impl)

def _patch_pathlib(func):
    if False:
        for i in range(10):
            print('nop')
    'Replace `Path.open` with `func`'
    old_impl = pathlib.Path.open
    pathlib.Path.open = func
    return old_impl

def smart_open(uri, mode='rb', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None, ignore_extension=False, **kwargs):
    if False:
        i = 10
        return i + 15
    url = 'https://github.com/RaRe-Technologies/smart_open/blob/develop/MIGRATING_FROM_OLDER_VERSIONS.rst'
    if kwargs:
        raise DeprecationWarning('The following keyword parameters are not supported: %r. See  %s for more information.' % (sorted(kwargs), url))
    message = 'This function is deprecated.  See %s for more information' % url
    warnings.warn(message, category=DeprecationWarning)
    if ignore_extension:
        compression = so_compression.NO_COMPRESSION
    else:
        compression = so_compression.INFER_FROM_EXTENSION
    del kwargs, url, message, ignore_extension
    return open(**locals())
try:
    doctools.tweak_open_docstring(open)
    doctools.tweak_parse_uri_docstring(parse_uri)
except Exception as ex:
    logger.error('Encountered a non-fatal error while building docstrings (see below). help(smart_open) will provide incomplete information as a result. For full help text, see <https://github.com/RaRe-Technologies/smart_open/blob/master/help.txt>.')
    logger.exception(ex)