from __future__ import absolute_import, division
import functools
import re
import posixpath
import sys
from calendar import timegm
from hashlib import md5
from logging import debug, warning, error
import xml.dom.minidom
import xml.etree.ElementTree as ET
from .ExitCodes import EX_OSFILE
try:
    import dateutil.parser
except ImportError:
    sys.stderr.write(u'\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nImportError trying to import dateutil.parser.\nPlease install the python dateutil module:\n$ sudo apt-get install python-dateutil\n  or\n$ sudo yum install python-dateutil\n  or\n$ pip install python-dateutil\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
    sys.stderr.flush()
    sys.exit(EX_OSFILE)
try:
    from urllib import quote
except ImportError:
    from urllib.parse import quote
try:
    unicode = unicode
except NameError:
    unicode = str
__all__ = []
s3path = posixpath
__all__.append('s3path')
try:
    md5()
except ValueError as exc:
    try:
        md5(usedforsecurity=False)
        md5 = functools.partial(md5, usedforsecurity=False)
    except Exception:
        raise exc
__all__.append('md5')
RE_S3_DATESTRING = re.compile('\\.[0-9]*(?:[Z\\-\\+]*?)')
RE_XML_NAMESPACE = re.compile(b'^(<?[^>]+?>\\s*|\\s*)(<\\w+) xmlns=[\'"](https?://[^\'"]+)[\'"]', re.MULTILINE)

def dateS3toPython(date):
    if False:
        print('Hello World!')
    date = RE_S3_DATESTRING.sub('.000', date)
    return dateutil.parser.parse(date, fuzzy=True)
__all__.append('dateS3toPython')

def dateS3toUnix(date):
    if False:
        while True:
            i = 10
    return timegm(dateS3toPython(date).utctimetuple())
__all__.append('dateS3toUnix')

def dateRFC822toPython(date):
    if False:
        print('Hello World!')
    "\n    Convert a string formatted like '2020-06-27T15:56:34Z' into a python datetime\n    "
    return dateutil.parser.parse(date, fuzzy=True)
__all__.append('dateRFC822toPython')

def dateRFC822toUnix(date):
    if False:
        i = 10
        return i + 15
    return timegm(dateRFC822toPython(date).utctimetuple())
__all__.append('dateRFC822toUnix')

def formatDateTime(s3timestamp):
    if False:
        i = 10
        return i + 15
    date_obj = dateutil.parser.parse(s3timestamp, fuzzy=True)
    return date_obj.strftime('%Y-%m-%d %H:%M')
__all__.append('formatDateTime')

def base_unicodise(string, encoding='UTF-8', errors='replace', silent=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert 'string' to Unicode or raise an exception.\n    "
    if type(string) == unicode:
        return string
    if not silent:
        debug('Unicodising %r using %s' % (string, encoding))
    try:
        return unicode(string, encoding, errors)
    except UnicodeDecodeError:
        raise UnicodeDecodeError('Conversion to unicode failed: %r' % string)
__all__.append('base_unicodise')

def base_deunicodise(string, encoding='UTF-8', errors='replace', silent=False):
    if False:
        print('Hello World!')
    "\n    Convert unicode 'string' to <type str>, by default replacing\n    all invalid characters with '?' or raise an exception.\n    "
    if type(string) != unicode:
        return string
    if not silent:
        debug('DeUnicodising %r using %s' % (string, encoding))
    try:
        return string.encode(encoding, errors)
    except UnicodeEncodeError:
        raise UnicodeEncodeError('Conversion from unicode failed: %r' % string)
__all__.append('base_deunicodise')

def decode_from_s3(string, errors='replace'):
    if False:
        return 10
    "\n    Convert S3 UTF-8 'string' to Unicode or raise an exception.\n    "
    return base_unicodise(string, 'UTF-8', errors, True)
__all__.append('decode_from_s3')

def encode_to_s3(string, errors='replace'):
    if False:
        return 10
    "\n    Convert Unicode to S3 UTF-8 'string', by default replacing\n    all invalid characters with '?' or raise an exception.\n    "
    return base_deunicodise(string, 'UTF-8', errors, True)
__all__.append('encode_to_s3')

def s3_quote(param, quote_backslashes=True, unicode_output=False):
    if False:
        i = 10
        return i + 15
    '\n    URI encode every byte. UriEncode() must enforce the following rules:\n    - URI encode every byte except the unreserved characters: \'A\'-\'Z\', \'a\'-\'z\', \'0\'-\'9\', \'-\', \'.\', \'_\', and \'~\'.\n    - The space character is a reserved character and must be encoded as "%20" (and not as "+").\n    - Each URI encoded byte is formed by a \'%\' and the two-digit hexadecimal value of the byte.\n    - Letters in the hexadecimal value must be uppercase, for example "%1A".\n    - Encode the forward slash character, \'/\', everywhere except in the object key name.\n    For example, if the object key name is photos/Jan/sample.jpg, the forward slash in the key name is not encoded.\n    '
    if quote_backslashes:
        safe_chars = '~'
    else:
        safe_chars = '~/'
    param = encode_to_s3(param)
    param = quote(param, safe=safe_chars)
    if unicode_output:
        param = decode_from_s3(param)
    else:
        param = encode_to_s3(param)
    return param
__all__.append('s3_quote')

def base_urlencode_string(string, urlencoding_mode=None, unicode_output=False):
    if False:
        while True:
            i = 10
    string = encode_to_s3(string)
    if urlencoding_mode == 'verbatim':
        return string
    encoded = quote(string, safe='~/')
    debug("String '%s' encoded to '%s'" % (string, encoded))
    if unicode_output:
        return decode_from_s3(encoded)
    else:
        return encode_to_s3(encoded)
__all__.append('base_urlencode_string')

def base_replace_nonprintables(string, with_message=False):
    if False:
        print('Hello World!')
    "\n    replace_nonprintables(string)\n\n    Replaces all non-printable characters 'ch' in 'string'\n    where ord(ch) <= 26 with ^@, ^A, ... ^Z\n    "
    new_string = ''
    modified = 0
    for c in string:
        o = ord(c)
        if o <= 31:
            new_string += '^' + chr(ord('@') + o)
            modified += 1
        elif o == 127:
            new_string += '^?'
            modified += 1
        else:
            new_string += c
    if modified and with_message:
        warning('%d non-printable characters replaced in: %s' % (modified, new_string))
    return new_string
__all__.append('base_replace_nonprintables')

def parseNodes(nodes):
    if False:
        while True:
            i = 10
    retval = []
    for node in nodes:
        retval_item = {}
        for child in node:
            name = decode_from_s3(child.tag)
            if len(child):
                retval_item[name] = parseNodes([child])
            else:
                found_text = node.findtext('.//%s' % child.tag)
                if found_text is not None:
                    retval_item[name] = decode_from_s3(found_text)
                else:
                    retval_item[name] = None
        if retval_item:
            retval.append(retval_item)
    return retval
__all__.append('parseNodes')

def getPrettyFromXml(xmlstr):
    if False:
        for i in range(10):
            print('nop')
    xmlparser = xml.dom.minidom.parseString(xmlstr)
    return xmlparser.toprettyxml()
__all__.append('getPrettyFromXml')

def stripNameSpace(xml):
    if False:
        print('Hello World!')
    '\n    removeNameSpace(xml) -- remove top-level AWS namespace\n    Operate on raw byte(utf-8) xml string. (Not unicode)\n    '
    xmlns_match = RE_XML_NAMESPACE.match(xml)
    if xmlns_match:
        xmlns = xmlns_match.group(3)
        xml = RE_XML_NAMESPACE.sub(b'\\1\\2', xml, 1)
    else:
        xmlns = None
    return (xml, xmlns)
__all__.append('stripNameSpace')

def getTreeFromXml(xml):
    if False:
        return 10
    (xml, xmlns) = stripNameSpace(encode_to_s3(xml))
    try:
        tree = ET.fromstring(xml)
        if xmlns:
            tree.attrib['xmlns'] = xmlns
        return tree
    except Exception as e:
        error('Error parsing xml: %s', e)
        error(xml)
        raise
__all__.append('getTreeFromXml')

def getListFromXml(xml, node):
    if False:
        i = 10
        return i + 15
    tree = getTreeFromXml(xml)
    nodes = tree.findall('.//%s' % node)
    return parseNodes(nodes)
__all__.append('getListFromXml')

def getDictFromTree(tree):
    if False:
        i = 10
        return i + 15
    ret_dict = {}
    for child in tree:
        if len(child):
            content = getDictFromTree(child)
        else:
            content = decode_from_s3(child.text) if child.text is not None else None
        child_tag = decode_from_s3(child.tag)
        if child_tag in ret_dict:
            if not type(ret_dict[child_tag]) == list:
                ret_dict[child_tag] = [ret_dict[child_tag]]
            ret_dict[child_tag].append(content or '')
        else:
            ret_dict[child_tag] = content or ''
    return ret_dict
__all__.append('getDictFromTree')

def getTextFromXml(xml, xpath):
    if False:
        for i in range(10):
            print('nop')
    tree = getTreeFromXml(xml)
    if tree.tag.endswith(xpath):
        return decode_from_s3(tree.text) if tree.text is not None else None
    else:
        result = tree.findtext(xpath)
        return decode_from_s3(result) if result is not None else None
__all__.append('getTextFromXml')

def getRootTagName(xml):
    if False:
        while True:
            i = 10
    tree = getTreeFromXml(xml)
    return decode_from_s3(tree.tag) if tree.tag is not None else None
__all__.append('getRootTagName')

def xmlTextNode(tag_name, text):
    if False:
        return 10
    el = ET.Element(tag_name)
    el.text = decode_from_s3(text)
    return el
__all__.append('xmlTextNode')

def appendXmlTextNode(tag_name, text, parent):
    if False:
        print('Hello World!')
    "\n    Creates a new <tag_name> Node and sets\n    its content to 'text'. Then appends the\n    created Node to 'parent' element if given.\n    Returns the newly created Node.\n    "
    el = xmlTextNode(tag_name, text)
    parent.append(el)
    return el
__all__.append('appendXmlTextNode')