__license__ = 'GPL 3'
__copyright__ = '2009, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'
'\nTransform OEB content into RB compatible markup.\n'
import re
from calibre import prepare_string_for_xml
from calibre.ebooks.rb import unique_name
from polyglot.builtins import string_or_bytes
TAGS = ['b', 'big', 'blockquote', 'br', 'center', 'code', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'li', 'ol', 'p', 'pre', 'small', 'sub', 'sup', 'ul']
LINK_TAGS = ['a']
IMAGE_TAGS = ['img']
STYLES = [('font-weight', {'bold': 'b', 'bolder': 'b'}), ('font-style', {'italic': 'i'}), ('text-align', {'center': 'center'})]

class RBMLizer:

    def __init__(self, log, name_map={}):
        if False:
            print('Hello World!')
        self.log = log
        self.name_map = name_map
        self.link_hrefs = {}

    def extract_content(self, oeb_book, opts):
        if False:
            for i in range(10):
                print('nop')
        self.log.info('Converting XHTML to RB markup...')
        self.oeb_book = oeb_book
        self.opts = opts
        return self.mlize_spine()

    def mlize_spine(self):
        if False:
            i = 10
            return i + 15
        self.link_hrefs = {}
        output = ['<HTML><HEAD><TITLE></TITLE></HEAD><BODY>']
        output.append(self.get_cover_page())
        output.append('ghji87yhjko0Caliblre-toc-placeholder-for-insertion-later8ujko0987yjk')
        output.append(self.get_text())
        output.append('</BODY></HTML>')
        output = ''.join(output).replace('ghji87yhjko0Caliblre-toc-placeholder-for-insertion-later8ujko0987yjk', self.get_toc())
        output = self.clean_text(output)
        return output

    def get_cover_page(self):
        if False:
            i = 10
            return i + 15
        from calibre.ebooks.oeb.stylizer import Stylizer
        from calibre.ebooks.oeb.base import XHTML
        output = ''
        if 'cover' in self.oeb_book.guide:
            if self.name_map.get(self.oeb_book.guide['cover'].href, None):
                output += '<IMG SRC="%s">' % self.name_map[self.oeb_book.guide['cover'].href]
        if 'titlepage' in self.oeb_book.guide:
            self.log.debug('Generating cover page...')
            href = self.oeb_book.guide['titlepage'].href
            item = self.oeb_book.manifest.hrefs[href]
            if item.spine_position is None:
                stylizer = Stylizer(item.data, item.href, self.oeb_book, self.opts, self.opts.output_profile)
                output += ''.join(self.dump_text(item.data.find(XHTML('body')), stylizer, item))
        return output

    def get_toc(self):
        if False:
            for i in range(10):
                print('nop')
        toc = ['']
        if self.opts.inline_toc:
            self.log.debug('Generating table of contents...')
            toc.append('<H1>%s</H1><UL>\n' % _('Table of Contents:'))
            for item in self.oeb_book.toc:
                if item.href in self.link_hrefs.keys():
                    toc.append(f'<LI><A HREF="#{self.link_hrefs[item.href]}">{item.title}</A></LI>\n')
                else:
                    self.oeb.warn('Ignoring toc item: %s not found in document.' % item)
            toc.append('</UL>')
        return ''.join(toc)

    def get_text(self):
        if False:
            print('Hello World!')
        from calibre.ebooks.oeb.stylizer import Stylizer
        from calibre.ebooks.oeb.base import XHTML
        output = ['']
        for item in self.oeb_book.spine:
            self.log.debug('Converting %s to RocketBook HTML...' % item.href)
            stylizer = Stylizer(item.data, item.href, self.oeb_book, self.opts, self.opts.output_profile)
            output.append(self.add_page_anchor(item))
            output += self.dump_text(item.data.find(XHTML('body')), stylizer, item)
        return ''.join(output)

    def add_page_anchor(self, page):
        if False:
            print('Hello World!')
        return self.get_anchor(page, '')

    def get_anchor(self, page, aid):
        if False:
            i = 10
            return i + 15
        aid = f'{page.href}#{aid}'
        if aid not in self.link_hrefs.keys():
            self.link_hrefs[aid] = 'calibre_link-%s' % len(self.link_hrefs.keys())
        aid = self.link_hrefs[aid]
        return '<A NAME="%s"></A>' % aid

    def clean_text(self, text):
        if False:
            for i in range(10):
                print('nop')
        anchors = set(re.findall('(?<=<A NAME=").+?(?="></A>)', text))
        links = set(re.findall('(?<=<A HREF="#).+?(?=">)', text))
        for unused in anchors.difference(links):
            text = text.replace('<A NAME="%s"></A>' % unused, '')
        return text

    def dump_text(self, elem, stylizer, page, tag_stack=[]):
        if False:
            print('Hello World!')
        from calibre.ebooks.oeb.base import XHTML_NS, barename, namespace
        if not isinstance(elem.tag, string_or_bytes) or namespace(elem.tag) != XHTML_NS:
            p = elem.getparent()
            if p is not None and isinstance(p.tag, string_or_bytes) and (namespace(p.tag) == XHTML_NS) and elem.tail:
                return [elem.tail]
            return ['']
        text = ['']
        style = stylizer.style(elem)
        if style['display'] in ('none', 'oeb-page-head', 'oeb-page-foot') or style['visibility'] == 'hidden':
            if hasattr(elem, 'tail') and elem.tail:
                return [elem.tail]
            return ['']
        tag = barename(elem.tag)
        tag_count = 0
        if tag in IMAGE_TAGS:
            if elem.attrib.get('src', None):
                if page.abshref(elem.attrib['src']) not in self.name_map.keys():
                    self.name_map[page.abshref(elem.attrib['src'])] = unique_name('%s' % len(self.name_map.keys()), self.name_map.keys())
                text.append('<IMG SRC="%s">' % self.name_map[page.abshref(elem.attrib['src'])])
        rb_tag = tag.upper() if tag in TAGS else None
        if rb_tag:
            tag_count += 1
            text.append('<%s>' % rb_tag)
            tag_stack.append(rb_tag)
        if tag in LINK_TAGS:
            href = elem.get('href')
            if href:
                href = page.abshref(href)
                if '://' not in href:
                    if '#' not in href:
                        href += '#'
                    if href not in self.link_hrefs.keys():
                        self.link_hrefs[href] = 'calibre_link-%s' % len(self.link_hrefs.keys())
                    href = self.link_hrefs[href]
                    text.append('<A HREF="#%s">' % href)
                tag_count += 1
                tag_stack.append('A')
        id_name = elem.get('id')
        if id_name:
            text.append(self.get_anchor(page, id_name))
        for s in STYLES:
            style_tag = s[1].get(style[s[0]], None)
            if style_tag:
                style_tag = style_tag.upper()
                tag_count += 1
                text.append('<%s>' % style_tag)
                tag_stack.append(style_tag)
        if hasattr(elem, 'text') and elem.text:
            text.append(prepare_string_for_xml(elem.text))
        for item in elem:
            text += self.dump_text(item, stylizer, page, tag_stack)
        close_tag_list = []
        for i in range(0, tag_count):
            close_tag_list.insert(0, tag_stack.pop())
        text += self.close_tags(close_tag_list)
        if hasattr(elem, 'tail') and elem.tail:
            text.append(prepare_string_for_xml(elem.tail))
        return text

    def close_tags(self, tags):
        if False:
            while True:
                i = 10
        text = ['']
        for i in range(0, len(tags)):
            tag = tags.pop()
            text.append('</%s>' % tag)
        return text