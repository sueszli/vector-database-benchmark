import bs4
from bs4 import CData, Comment, Declaration, NavigableString, ProcessingInstruction, SoupStrainer, Tag, __version__

def parse_html(markup):
    if False:
        print('Hello World!')
    from calibre.ebooks.chardet import strip_encoding_declarations, xml_to_unicode, substitute_entites
    from calibre.utils.cleantext import clean_xml_chars
    if isinstance(markup, str):
        markup = strip_encoding_declarations(markup)
        markup = substitute_entites(markup)
    else:
        markup = xml_to_unicode(markup, strip_encoding_pats=True, resolve_entities=True)[0]
    markup = clean_xml_chars(markup)
    from html5_parser.soup import parse
    return parse(markup, return_root=False)

def prettify(soup):
    if False:
        for i in range(10):
            print('nop')
    ans = soup.prettify()
    if isinstance(ans, bytes):
        ans = ans.decode('utf-8')
    return ans

def BeautifulSoup(markup='', *a, **kw):
    if False:
        i = 10
        return i + 15
    return parse_html(markup)

def BeautifulStoneSoup(markup='', *a, **kw):
    if False:
        print('Hello World!')
    return bs4.BeautifulSoup(markup, 'xml')