import re
from calibre.constants import preferred_encoding
from calibre_extensions.speedup import clean_xml_chars as _ncxc
from polyglot.builtins import codepoint_to_chr
from polyglot.html_entities import name2codepoint

def native_clean_xml_chars(x):
    if False:
        i = 10
        return i + 15
    if isinstance(x, bytes):
        x = x.decode(preferred_encoding)
    return _ncxc(x)

def ascii_pat(for_binary=False):
    if False:
        while True:
            i = 10
    attr = 'binary' if for_binary else 'text'
    ans = getattr(ascii_pat, attr, None)
    if ans is None:
        chars = set(range(32)) - {9, 10, 13}
        chars.add(127)
        pat = '|'.join(map(codepoint_to_chr, chars))
        if for_binary:
            pat = pat.encode('ascii')
        ans = re.compile(pat)
        setattr(ascii_pat, attr, ans)
    return ans

def clean_ascii_chars(txt, charlist=None):
    if False:
        print('Hello World!')
    '\n    Remove ASCII control chars.\n    This is all control chars except \\t, \\n and \\r\n    '
    is_binary = isinstance(txt, bytes)
    empty = b'' if is_binary else ''
    if not txt:
        return empty
    if charlist is None:
        pat = ascii_pat(is_binary)
    else:
        pat = '|'.join(map(codepoint_to_chr, charlist))
        if is_binary:
            pat = pat.encode('utf-8')
    return pat.sub(empty, txt)

def allowed(x):
    if False:
        for i in range(10):
            print('nop')
    x = ord(x)
    return x != 127 and (31 < x < 55295 or x in (9, 10, 13)) or 57344 < x < 65533 or 65536 < x < 1114111

def py_clean_xml_chars(unicode_string):
    if False:
        return 10
    return ''.join(filter(allowed, unicode_string))
clean_xml_chars = native_clean_xml_chars or py_clean_xml_chars

def test_clean_xml_chars():
    if False:
        while True:
            i = 10
    raw = 'asd\x02a𐐷x\ud801b\udffe\ud802'
    if native_clean_xml_chars(raw) != 'asda𐐷xb':
        raise ValueError('Failed to XML clean: %r' % raw)

def unescape(text, rm=False, rchar=''):
    if False:
        i = 10
        return i + 15

    def fixup(m, rm=rm, rchar=rchar):
        if False:
            return 10
        text = m.group(0)
        if text[:2] == '&#':
            try:
                if text[:3] == '&#x':
                    return codepoint_to_chr(int(text[3:-1], 16))
                else:
                    return codepoint_to_chr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            try:
                text = codepoint_to_chr(name2codepoint[text[1:-1]])
            except KeyError:
                pass
        if rm:
            return rchar
        return text
    return re.sub('&#?\\w+;', fixup, text)