"""
Mimetypes-related utilities

# TODO: reexport stdlib mimetypes?
"""
import collections
import io
import logging
import re
import zipfile
__all__ = ['guess_mimetype']
_logger = logging.getLogger(__name__)
_ooxml_dirs = {'word/': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'pt/': 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'xl/': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}

def _check_ooxml(data):
    if False:
        return 10
    with io.BytesIO(data) as f, zipfile.ZipFile(f) as z:
        filenames = z.namelist()
        if '[Content_Types].xml' not in filenames:
            return False
        for (dirname, mime) in _ooxml_dirs.iteritems():
            if any((entry.startswith(dirname) for entry in filenames)):
                return mime
        return False
_mime_validator = re.compile('\n    [\\w-]+ # type-name\n    / # subtype separator\n    [\\w-]+ # registration facet or subtype\n    (?:\\.[\\w-]+)* # optional faceted name\n    (?:\\+[\\w-]+)? # optional structured syntax specifier\n', re.VERBOSE)

def _check_open_container_format(data):
    if False:
        while True:
            i = 10
    with io.BytesIO(data) as f, zipfile.ZipFile(f) as z:
        if 'mimetype' not in z.namelist():
            return False
        marcel = z.read('mimetype')
        if len(marcel) < 256 and _mime_validator.match(marcel):
            return marcel
        return False
_xls_pattern = re.compile('\n    \t\x08\x10\x00\x00\x06\x05\x00\n  | ýÿÿÿ(\x10|\x1f| |"|\\#|\\(|\\))\n', re.VERBOSE)
_ppt_pattern = re.compile('\n    \x00n\x1eð\n  | \x0f\x00è\x03\n  | \xa0F\x1dð\n  | ýÿÿÿ(\x0e|\x1c|C)\x00\x00\x00\n', re.VERBOSE)

def _check_olecf(data):
    if False:
        for i in range(10):
            print('nop')
    ' Pre-OOXML Office formats are OLE Compound Files which all use the same\n    file signature ("magic bytes") and should have a subheader at offset 512\n    (0x200).\n\n    Subheaders taken from http://www.garykessler.net/library/file_sigs.html\n    according to which Mac office files *may* have different subheaders. We\'ll\n    ignore that.\n    '
    offset = 512
    if data.startswith('ì¥Á\x00', offset):
        return 'application/msword'
    elif 'Microsoft Excel' in data:
        return 'application/vnd.ms-excel'
    elif _ppt_pattern.match(data, offset):
        return 'application/vnd.ms-powerpoint'
    return False
_Entry = collections.namedtuple('_Entry', ['mimetype', 'signatures', 'discriminants'])
_mime_mappings = (_Entry('application/pdf', ['%PDF'], []), _Entry('image/jpeg', ['ÿØÿà', 'ÿØÿâ', 'ÿØÿã', 'ÿØÿá'], []), _Entry('image/png', ['\x89PNG\r\n\x1a\n'], []), _Entry('image/gif', ['GIF87a', 'GIF89a'], []), _Entry('image/bmp', ['BM'], []), _Entry('application/msword', ['ÐÏ\x11à¡±\x1aá', '\rDOC'], [_check_olecf]), _Entry('application/zip', ['PK\x03\x04'], [_check_ooxml, _check_open_container_format]))

def guess_mimetype(bin_data, default='application/octet-stream'):
    if False:
        return 10
    ' Attempts to guess the mime type of the provided binary data, similar\n    to but significantly more limited than libmagic\n\n    :param str bin_data: binary data to try and guess a mime type for\n    :returns: matched mimetype or ``application/octet-stream`` if none matched\n    '
    for entry in _mime_mappings:
        for signature in entry.signatures:
            if bin_data.startswith(signature):
                for discriminant in entry.discriminants:
                    try:
                        guess = discriminant(bin_data)
                        if guess:
                            return guess
                    except Exception:
                        _logger.getChild('guess_mimetype').warn("Sub-checker '%s' of type '%s' failed", discriminant.__name__, entry.mimetype, exc_info=True)
                return entry.mimetype
    return default
try:
    import magic
except ImportError:
    magic = None
else:
    if hasattr(magic, 'from_buffer'):
        guess_mimetype = lambda bin_data, default=None: magic.from_buffer(bin_data, mime=True)
    elif hasattr(magic, 'open'):
        ms = magic.open(magic.MAGIC_MIME_TYPE)
        ms.load()
        guess_mimetype = lambda bin_data, default=None: ms.buffer(bin_data)