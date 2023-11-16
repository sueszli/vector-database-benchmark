"""Deobfuscator for scripts messed up with MyObfuscate.com"""
import re
import base64
try:
    from urllib import unquote
except ImportError:
    from urllib.parse import unquote
from jsbeautifier.unpackers import UnpackingError
PRIORITY = 1
CAVEAT = '//\n// Unpacker warning: be careful when using myobfuscate.com for your projects:\n// scripts obfuscated by the free online version call back home.\n//\n\n'
SIGNATURE = '["\\x41\\x42\\x43\\x44\\x45\\x46\\x47\\x48\\x49\\x4A\\x4B\\x4C\\x4D\\x4E\\x4F\\x50\\x51\\x52\\x53\\x54\\x55\\x56\\x57\\x58\\x59\\x5A\\x61\\x62\\x63\\x64\\x65\\x66\\x67\\x68\\x69\\x6A\\x6B\\x6C\\x6D\\x6E\\x6F\\x70\\x71\\x72\\x73\\x74\\x75\\x76\\x77\\x78\\x79\\x7A\\x30\\x31\\x32\\x33\\x34\\x35\\x36\\x37\\x38\\x39\\x2B\\x2F\\x3D","","\\x63\\x68\\x61\\x72\\x41\\x74","\\x69\\x6E\\x64\\x65\\x78\\x4F\\x66","\\x66\\x72\\x6F\\x6D\\x43\\x68\\x61\\x72\\x43\\x6F\\x64\\x65","\\x6C\\x65\\x6E\\x67\\x74\\x68"]'

def detect(source):
    if False:
        i = 10
        return i + 15
    'Detects MyObfuscate.com packer.'
    return SIGNATURE in source

def unpack(source):
    if False:
        print('Hello World!')
    'Unpacks js code packed with MyObfuscate.com'
    if not detect(source):
        return source
    payload = unquote(_filter(source))
    match = re.search("^var _escape\\='<script>(.*)<\\/script>'", payload, re.DOTALL)
    polished = match.group(1) if match else source
    return CAVEAT + polished

def _filter(source):
    if False:
        print('Hello World!')
    'Extracts and decode payload (original file) from `source`'
    try:
        varname = re.search('eval\\(\\w+\\(\\w+\\((\\w+)\\)\\)\\);', source).group(1)
        reverse = re.search("var +%s *\\= *'(.*)';" % varname, source).group(1)
    except AttributeError:
        raise UnpackingError('Malformed MyObfuscate data.')
    try:
        return base64.b64decode(reverse[::-1].encode('utf8')).decode('utf8')
    except TypeError:
        raise UnpackingError('MyObfuscate payload is not base64-encoded.')