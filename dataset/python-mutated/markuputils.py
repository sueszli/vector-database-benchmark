import re
from .htmlformatters import LinkFormatter, HtmlFormatter
_format_url = LinkFormatter().format_url
_format_html = HtmlFormatter().format
_generic_escapes = (('&', '&amp;'), ('<', '&lt;'), ('>', '&gt;'))
_attribute_escapes = _generic_escapes + (('"', '&quot;'), ('\n', '&#10;'), ('\r', '&#13;'), ('\t', '&#09;'))
_illegal_chars_in_xml = re.compile('[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]')

def html_escape(text, linkify=True):
    if False:
        i = 10
        return i + 15
    text = _escape(text)
    if linkify and '://' in text:
        text = _format_url(text)
    return text

def xml_escape(text):
    if False:
        return 10
    return _illegal_chars_in_xml.sub('', _escape(text))

def html_format(text):
    if False:
        for i in range(10):
            print('nop')
    return _format_html(_escape(text))

def attribute_escape(attr):
    if False:
        while True:
            i = 10
    attr = _escape(attr, _attribute_escapes)
    return _illegal_chars_in_xml.sub('', attr)

def _escape(text, escapes=_generic_escapes):
    if False:
        print('Hello World!')
    for (name, value) in escapes:
        if name in text:
            text = text.replace(name, value)
    return text