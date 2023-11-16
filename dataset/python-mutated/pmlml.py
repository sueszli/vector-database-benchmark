__license__ = 'GPL 3'
__copyright__ = '2009, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'
'\nTransform OEB content into PML markup\n'
import re
from lxml import etree
from calibre.ebooks.pdb.ereader import image_name
from calibre.utils.xml_parse import safe_xml_fromstring
from calibre.ebooks.pml import unipmlcode
from polyglot.builtins import string_or_bytes
TAG_MAP = {'b': 'B', 'strong': 'B', 'i': 'i', 'small': 'k', 'sub': 'Sb', 'sup': 'Sp', 'big': 'l', 'del': 'o', 'h1': 'x', 'h2': 'X0', 'h3': 'X1', 'h4': 'X2', 'h5': 'X3', 'h6': 'X4', '!--': 'v'}
STYLES = [('font-weight', {'bold': 'B', 'bolder': 'B'}), ('font-style', {'italic': 'i'}), ('text-decoration', {'underline': 'u'}), ('text-align', {'right': 'r', 'center': 'c'})]
BLOCK_TAGS = ['p', 'div']
BLOCK_STYLES = ['block']
LINK_TAGS = ['a']
IMAGE_TAGS = ['img']
SEPARATE_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'img', 'li', 'tr']

class PMLMLizer:

    def __init__(self, log):
        if False:
            i = 10
            return i + 15
        self.log = log
        self.image_hrefs = {}
        self.link_hrefs = {}

    def extract_content(self, oeb_book, opts):
        if False:
            i = 10
            return i + 15
        self.log.info('Converting XHTML to PML markup...')
        self.oeb_book = oeb_book
        self.opts = opts
        self.toc = {}
        self.create_flat_toc(self.oeb_book.toc)
        return self.pmlmlize_spine()

    def create_flat_toc(self, nodes, level=0):
        if False:
            return 10
        for item in nodes:
            (href, mid, id) = item.href.partition('#')
            self.get_anchor_id(href, id)
            if not self.toc.get(href, None):
                self.toc[href] = {}
            self.toc[href][id] = (item.title, level)
            self.create_flat_toc(item.nodes, level + 1)

    def pmlmlize_spine(self):
        if False:
            return 10
        self.image_hrefs = {}
        self.link_hrefs = {}
        output = ['']
        output.append(self.get_cover_page())
        output.append(self.get_text())
        output = ''.join(output)
        output = self.clean_text(output)
        return output

    def get_cover_page(self):
        if False:
            print('Hello World!')
        from calibre.ebooks.oeb.stylizer import Stylizer
        from calibre.ebooks.oeb.base import XHTML
        output = ''
        if 'cover' in self.oeb_book.guide:
            output += '\\m="cover.png"\n'
            self.image_hrefs[self.oeb_book.guide['cover'].href] = 'cover.png'
        if 'titlepage' in self.oeb_book.guide:
            self.log.debug('Generating title page...')
            href = self.oeb_book.guide['titlepage'].href
            item = self.oeb_book.manifest.hrefs[href]
            if item.spine_position is None:
                stylizer = Stylizer(item.data, item.href, self.oeb_book, self.opts, self.opts.output_profile)
                output += ''.join(self.dump_text(item.data.find(XHTML('body')), stylizer, item))
        return output

    def get_text(self):
        if False:
            print('Hello World!')
        from calibre.ebooks.oeb.stylizer import Stylizer
        from calibre.ebooks.oeb.base import XHTML
        text = ['']
        for item in self.oeb_book.spine:
            self.log.debug('Converting %s to PML markup...' % item.href)
            content = etree.tostring(item.data, encoding='unicode')
            content = self.prepare_text(content)
            content = safe_xml_fromstring(content)
            stylizer = Stylizer(content, item.href, self.oeb_book, self.opts, self.opts.output_profile)
            text.append(self.add_page_anchor(item))
            text += self.dump_text(content.find(XHTML('body')), stylizer, item)
        return ''.join(text)

    def add_page_anchor(self, page):
        if False:
            i = 10
            return i + 15
        return self.get_anchor(page, '')

    def get_anchor_id(self, href, aid):
        if False:
            print('Hello World!')
        aid = f'{href}#{aid}'
        if aid not in self.link_hrefs.keys():
            self.link_hrefs[aid] = 'calibre_link-%s' % len(self.link_hrefs.keys())
        aid = self.link_hrefs[aid]
        return aid

    def get_anchor(self, page, aid):
        if False:
            print('Hello World!')
        aid = self.get_anchor_id(page.href, aid)
        return '\\Q="%s"' % aid

    def remove_newlines(self, text):
        if False:
            print('Hello World!')
        text = text.replace('\r\n', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        return text

    def prepare_string_for_pml(self, text):
        if False:
            return 10
        text = self.remove_newlines(text)
        text = text.replace('\\', '\\\\')
        text = text.replace('\\\\c \\\\c', '\\c \n\\c\n')
        return text

    def prepare_text(self, text):
        if False:
            return 10
        text = re.sub('(?<=</p>)\\s*<p[^>]*>[\\xc2\\xa0\\s]*</p>', '\\\\c\\n\\\\c', text)
        return text

    def clean_text(self, text):
        if False:
            print('Hello World!')
        text = re.sub('\\\\p\\s*\\\\p', '', text)
        anchors = set(re.findall('(?<=\\\\Q=").+?(?=")', text))
        links = set(re.findall('(?<=\\\\q="#).+?(?=")', text))
        for unused in anchors.difference(links):
            text = text.replace('\\Q="%s"' % unused, '')
        text = re.sub('(?msu)(?P<t>\\\\(x|X[0-4]))(?P<a>.*?)(?P<c>\\\\C[0-4]\\s*=\\s*"[^"]*")(?P<b>.*?)(?P=t)', '\\g<t>\\g<a>\\g<b>\\g<t>', text)
        text = text.replace('Â', '')
        text = text.replace('\xa0', ' ')
        text = re.sub('[^\x00-\x7f]', lambda x: unipmlcode(x.group()), text)
        text = re.sub('(?m)^[ ]+', '', text)
        text = re.sub('(?m)[ ]+$', '', text)
        text = re.sub('[ ]{2,}', ' ', text)
        text = re.sub('(\\\\c\\s*\\\\c\\s*){2,}', '\\\\c \\n\\\\c\\n', text)
        text = re.sub('\n[ ]+\n', '\n\n', text)
        if self.opts.remove_paragraph_spacing:
            text = re.sub('\n{2,}', '\n', text)
            text = re.sub('(?imu)^(?P<text>.+)$', lambda mo: mo.group('text') if re.search('\\\\[XxCmrctTp]', mo.group('text')) else '        %s' % mo.group('text'), text)
        else:
            text = re.sub('\n{3,}', '\n\n', text)
        return text

    def dump_text(self, elem, stylizer, page, tag_stack=[]):
        if False:
            return 10
        from calibre.ebooks.oeb.base import XHTML_NS, barename, namespace
        if not isinstance(elem.tag, string_or_bytes) or namespace(elem.tag) != XHTML_NS:
            p = elem.getparent()
            if p is not None and isinstance(p.tag, string_or_bytes) and (namespace(p.tag) == XHTML_NS) and elem.tail:
                return [elem.tail]
            return []
        text = []
        tags = []
        style = stylizer.style(elem)
        if style['display'] in ('none', 'oeb-page-head', 'oeb-page-foot') or style['visibility'] == 'hidden':
            if hasattr(elem, 'tail') and elem.tail:
                return [elem.tail]
            return []
        tag = barename(elem.tag)
        if tag in BLOCK_TAGS or style['display'] in BLOCK_STYLES:
            tags.append('block')
        if tag in IMAGE_TAGS:
            if elem.attrib.get('src', None):
                if page.abshref(elem.attrib['src']) not in self.image_hrefs.keys():
                    if len(self.image_hrefs.keys()) == 0:
                        self.image_hrefs[page.abshref(elem.attrib['src'])] = 'cover.png'
                    else:
                        self.image_hrefs[page.abshref(elem.attrib['src'])] = image_name('%s.png' % len(self.image_hrefs.keys()), self.image_hrefs.keys()).strip('\x00')
                text.append('\\m="%s"' % self.image_hrefs[page.abshref(elem.attrib['src'])])
        elif tag == 'hr':
            w = '\\w'
            width = elem.get('width')
            if width:
                if not width.endswith('%'):
                    width += '%'
                w += '="%s"' % width
            else:
                w += '="50%"'
            text.append(w)
        elif tag == 'br':
            text.append('\n\\c \n\\c\n')
        toc_name = elem.attrib.get('name', None)
        toc_id = elem.attrib.get('id', None)
        if (toc_id or toc_name) and tag not in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6') and ('x' not in tag_stack + tags) and ('X0' not in tag_stack + tags) and ('X1' not in tag_stack + tags) and ('X2' not in tag_stack + tags) and ('X3' not in tag_stack + tags) and ('X4' not in tag_stack + tags):
            toc_page = page.href
            if self.toc.get(toc_page, None):
                for toc_x in (toc_name, toc_id):
                    (toc_title, toc_depth) = self.toc[toc_page].get(toc_x, (None, 0))
                    if toc_title:
                        toc_depth = max(min(toc_depth, 4), 0)
                        text.append(f'\\C{toc_depth}="{toc_title}"')
        if style['page-break-before'] == 'always':
            text.append('\\p')
        pml_tag = TAG_MAP.get(tag, None)
        if pml_tag and pml_tag not in tag_stack + tags:
            text.append('\\%s' % pml_tag)
            tags.append(pml_tag)
        if tag in LINK_TAGS and 'q' not in tag_stack + tags:
            href = elem.get('href')
            if href:
                href = page.abshref(href)
                if '://' not in href:
                    if '#' not in href:
                        href += '#'
                    if href not in self.link_hrefs.keys():
                        self.link_hrefs[href] = 'calibre_link-%s' % len(self.link_hrefs.keys())
                    href = '#%s' % self.link_hrefs[href]
                    text.append('\\q="%s"' % href)
                    tags.append('q')
        id_name = elem.get('id')
        name_name = elem.get('name')
        for name_x in (id_name, name_name):
            if name_x:
                text.append(self.get_anchor(page, name_x))
        for s in STYLES:
            style_tag = s[1].get(style[s[0]], None)
            if style_tag and style_tag not in tag_stack + tags:
                text.append('\\%s' % style_tag)
                tags.append(style_tag)
        try:
            mms = int(float(style['margin-left']) * 100 / style.height)
            if mms:
                text.append('\\T="%s%%"' % mms)
        except:
            pass
        try:
            ems = int(round(float(style.marginTop) / style.fontSize - 1))
            if ems >= 1:
                text.append('\n\\c \n\\c\n')
        except:
            pass
        if hasattr(elem, 'text') and elem.text:
            text.append(self.prepare_string_for_pml(elem.text))
        for item in elem:
            text += self.dump_text(item, stylizer, page, tag_stack + tags)
        tags.reverse()
        text += self.close_tags(tags)
        if style['page-break-after'] == 'always':
            text.append('\\p')
        if hasattr(elem, 'tail') and elem.tail:
            text.append(self.prepare_string_for_pml(elem.tail))
        return text

    def close_tags(self, tags):
        if False:
            print('Hello World!')
        text = []
        for tag in tags:
            if tag == 'block':
                text.append('\n\n')
            elif tag in ('c', 'r'):
                text.append('\n\\%s' % tag)
            else:
                text.append('\\%s' % tag)
        return text