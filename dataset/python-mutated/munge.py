import os.path
import re
from typing import Union
from ckan import model
from ckan.lib.io import decode_path
MAX_FILENAME_EXTENSION_LENGTH = 21
MAX_FILENAME_TOTAL_LENGTH = 100
MIN_FILENAME_TOTAL_LENGTH = 3

def munge_name(name: str) -> str:
    if False:
        i = 10
        return i + 15
    'Munges the package name field in case it is not to spec.'
    name = substitute_ascii_equivalents(name)
    name = re.sub('[ .:/]', '-', name)
    name = re.sub('[^a-zA-Z0-9-_]', '', name).lower()
    name = _munge_to_length(name, model.PACKAGE_NAME_MIN_LENGTH, model.PACKAGE_NAME_MAX_LENGTH)
    return name

def munge_title_to_name(name: str) -> str:
    if False:
        i = 10
        return i + 15
    'Munge a package title into a package name.'
    name = substitute_ascii_equivalents(name)
    name = re.sub('[ .:/]', '-', name)
    name = re.sub('[^a-zA-Z0-9-_]', '', name).lower()
    name = re.sub('-+', '-', name)
    name = name.strip('-')
    max_length = model.PACKAGE_NAME_MAX_LENGTH - 5
    if len(name) > max_length:
        year_match = re.match('.*?[_-]((?:\\d{2,4}[-/])?\\d{2,4})$', name)
        if year_match:
            year = year_match.groups()[0]
            name = '%s-%s' % (name[:max_length - len(year) - 1], year)
        else:
            name = name[:max_length]
    name = _munge_to_length(name, model.PACKAGE_NAME_MIN_LENGTH, model.PACKAGE_NAME_MAX_LENGTH)
    return name

def substitute_ascii_equivalents(text_unicode: str) -> str:
    if False:
        return 10
    '\n    This takes a UNICODE string and replaces Latin-1 characters with something\n    equivalent in 7-bit ASCII. It returns a plain ASCII string. This function\n    makes a best effort to convert Latin-1 characters into ASCII equivalents.\n    It does not just strip out the Latin-1 characters. All characters in the\n    standard 7-bit ASCII range are preserved. In the 8th bit range all the\n    Latin-1 accented letters are converted to unaccented equivalents. Most\n    symbol characters are converted to something meaningful. Anything not\n    converted is deleted.\n    '
    char_mapping = {192: 'A', 193: 'A', 194: 'A', 195: 'A', 196: 'A', 197: 'A', 198: 'Ae', 199: 'C', 200: 'E', 201: 'E', 202: 'E', 203: 'E', 204: 'I', 205: 'I', 206: 'I', 207: 'I', 208: 'Th', 209: 'N', 210: 'O', 211: 'O', 212: 'O', 213: 'O', 214: 'O', 216: 'O', 217: 'U', 218: 'U', 219: 'U', 220: 'U', 221: 'Y', 222: 'th', 223: 'ss', 224: 'a', 225: 'a', 226: 'a', 227: 'a', 228: 'a', 229: 'a', 230: 'ae', 231: 'c', 232: 'e', 233: 'e', 234: 'e', 235: 'e', 236: 'i', 237: 'i', 238: 'i', 239: 'i', 240: 'th', 241: 'n', 242: 'o', 243: 'o', 244: 'o', 245: 'o', 246: 'o', 248: 'o', 249: 'u', 250: 'u', 251: 'u', 252: 'u', 253: 'y', 254: 'th', 255: 'y'}
    r = ''
    for char in text_unicode:
        if ord(char) in char_mapping:
            r += char_mapping[ord(char)]
        elif ord(char) >= 128:
            pass
        else:
            r += str(char)
    return r

def munge_tag(tag: str) -> str:
    if False:
        print('Hello World!')
    tag = substitute_ascii_equivalents(tag)
    tag = tag.lower().strip()
    tag = re.sub('[^a-zA-Z0-9\\- ]', '', tag).replace(' ', '-')
    tag = _munge_to_length(tag, model.MIN_TAG_LENGTH, model.MAX_TAG_LENGTH)
    return tag

def munge_filename_legacy(filename: str) -> str:
    if False:
        print('Hello World!')
    ' Tidies a filename. NB: deprecated\n\n    Unfortunately it mangles any path or filename extension, so is deprecated.\n    It needs to remain unchanged for use by group_dictize() and\n    Upload.update_data_dict() because if this routine changes then group images\n    uploaded previous to the change may not be viewable.\n    '
    filename = substitute_ascii_equivalents(filename)
    filename = filename.strip()
    filename = re.sub('[^a-zA-Z0-9.\\- ]', '', filename).replace(' ', '-')
    filename = _munge_to_length(filename, 3, 100)
    return filename

def munge_filename(filename: Union[str, bytes]) -> str:
    if False:
        return 10
    ' Tidies a filename\n\n    Keeps the filename extension (e.g. .csv).\n    Strips off any path on the front.\n\n    Returns a Unicode string.\n    '
    if not isinstance(filename, str):
        filename = decode_path(filename)
    filename = os.path.split(filename)[1]
    filename = filename.lower().strip()
    filename = substitute_ascii_equivalents(filename)
    filename = re.sub(u'[^a-zA-Z0-9_. -]', '', filename).replace(u' ', u'-')
    filename = re.sub(u'-+', u'-', filename)
    (name, ext) = os.path.splitext(filename)
    ext = ext[:MAX_FILENAME_EXTENSION_LENGTH]
    ext_len = len(ext)
    name = _munge_to_length(name, max(1, MIN_FILENAME_TOTAL_LENGTH - ext_len), MAX_FILENAME_TOTAL_LENGTH - ext_len)
    filename = name + ext
    return filename

def _munge_to_length(string: str, min_length: int, max_length: int) -> str:
    if False:
        return 10
    'Pad/truncates a string'
    if len(string) < min_length:
        string += '_' * (min_length - len(string))
    if len(string) > max_length:
        string = string[:max_length]
    return string