__license__ = 'GPL 3'
__copyright__ = '2009, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'
'\nTransform OEB content into plain text\n'
import re
from lxml import etree
from polyglot.builtins import string_or_bytes
BLOCK_TAGS = ['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr']
BLOCK_STYLES = ['block']
HEADING_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
SPACE_TAGS = ['td', 'br']

class TXTMLizer:

    def __init__(self, log):
        if False:
            print('Hello World!')
        self.log = log

    def extract_content(self, oeb_book, opts):
        if False:
            print('Hello World!')
        self.log.info('Converting XHTML to TXT...')
        self.oeb_book = oeb_book
        self.opts = opts
        self.toc_titles = []
        self.toc_ids = []
        self.last_was_heading = False
        self.create_flat_toc(self.oeb_book.toc)
        return self.mlize_spine()

    def mlize_spine(self):
        if False:
            for i in range(10):
                print('nop')
        from calibre.ebooks.oeb.base import XHTML
        from calibre.ebooks.oeb.stylizer import Stylizer
        from calibre.utils.xml_parse import safe_xml_fromstring
        output = ['']
        output.append(self.get_toc())
        for item in self.oeb_book.spine:
            self.log.debug('Converting %s to TXT...' % item.href)
            for x in item.data.iterdescendants(etree.Comment):
                if x.text and '--' in x.text:
                    x.text = x.text.replace('--', '__')
            content = etree.tostring(item.data, encoding='unicode')
            content = self.remove_newlines(content)
            content = safe_xml_fromstring(content)
            stylizer = Stylizer(content, item.href, self.oeb_book, self.opts, self.opts.output_profile)
            output += self.dump_text(content.find(XHTML('body')), stylizer, item)
            output += '\n\n\n\n\n\n'
        output = ''.join(output)
        output = '\n'.join((l.rstrip() for l in output.splitlines()))
        output = self.cleanup_text(output)
        return output

    def remove_newlines(self, text):
        if False:
            return 10
        self.log.debug('\tRemove newlines for processing...')
        text = text.replace('\r\n', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = re.sub('[ ]{2,}', ' ', text)
        return text

    def get_toc(self):
        if False:
            i = 10
            return i + 15
        toc = ['']
        if getattr(self.opts, 'inline_toc', None):
            self.log.debug('Generating table of contents...')
            toc.append('%s\n\n' % _('Table of Contents:'))
            for item in self.toc_titles:
                toc.append('* %s\n\n' % item)
        return ''.join(toc)

    def create_flat_toc(self, nodes):
        if False:
            for i in range(10):
                print('nop')
        "\n        Turns a hierarchical list of TOC href's into a flat list.\n        "
        for item in nodes:
            self.toc_titles.append(item.title)
            self.toc_ids.append(item.href)
            self.create_flat_toc(item.nodes)

    def cleanup_text(self, text):
        if False:
            print('Hello World!')
        self.log.debug('\tClean up text...')
        text = text.replace('\xa0', ' ')
        text = text.replace('\t+', ' ')
        text = text.replace('\x0b+', ' ')
        text = text.replace('\x0c+', ' ')
        text = re.sub('(?<=.)\n(?=.)', ' ', text)
        text = re.sub('[ ]{2,}', ' ', text)
        text = re.sub('\n[ ]+\n', '\n\n', text)
        if self.opts.remove_paragraph_spacing:
            text = re.sub('\n{2,}', '\n', text)
            text = re.sub('(?msu)^(?P<t>[^\\t\\n]+?)$', lambda mo: '%s\n\n' % mo.group('t'), text)
            text = re.sub('(?msu)(?P<b>[^\\n])\\n+(?P<t>[^\\t\\n]+?)(?=\\n)', lambda mo: '{}\n\n\n\n\n\n{}'.format(mo.group('b'), mo.group('t')), text)
        else:
            text = re.sub('\n{7,}', '\n\n\n\n\n\n', text)
        text = re.sub('(?imu)^[ ]+', '', text)
        text = re.sub('(?imu)[ ]+$', '', text)
        text = re.sub('(?u)^[ \\n]+', '', text)
        if self.opts.max_line_length:
            max_length = int(self.opts.max_line_length)
            if max_length < 25 and (not self.opts.force_max_line_length):
                max_length = 25
            short_lines = []
            lines = text.splitlines()
            for line in lines:
                while len(line) > max_length:
                    space = line.rfind(' ', 0, max_length)
                    if space != -1:
                        short_lines.append(line[:space])
                        line = line[space + 1:]
                    elif self.opts.force_max_line_length:
                        short_lines.append(line[:max_length])
                        line = line[max_length:]
                    else:
                        space = line.find(' ', max_length, len(line))
                        if space != -1:
                            short_lines.append(line[:space])
                            line = line[space + 1:]
                        else:
                            short_lines.append(line)
                            line = ''
                short_lines.append(line)
            text = '\n'.join(short_lines)
        return text

    def dump_text(self, elem, stylizer, page):
        if False:
            for i in range(10):
                print('nop')
        '\n        @elem: The element in the etree that we are working on.\n        @stylizer: The style information attached to the element.\n        @page: OEB page used to determine absolute urls.\n        '
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
        tag_id = elem.attrib.get('id', None)
        in_block = False
        in_heading = False
        if tag in HEADING_TAGS or f'{page.href}#{tag_id}' in self.toc_ids:
            in_heading = True
            if not self.last_was_heading:
                text.append('\n\n\n\n\n\n')
        if tag in BLOCK_TAGS or style['display'] in BLOCK_STYLES:
            if self.opts.remove_paragraph_spacing and (not in_heading):
                text.append('\t')
            in_block = True
        if tag in SPACE_TAGS:
            text.append(' ')
        if tag == 'hr':
            text.append('\n\n* * *\n\n')
        try:
            ems = int(round(float(style.marginTop) / style.fontSize - 1))
            if ems >= 1:
                text.append('\n' * ems)
        except:
            pass
        if hasattr(elem, 'text') and elem.text:
            text.append(elem.text)
        for item in elem:
            text += self.dump_text(item, stylizer, page)
        if in_block:
            text.append('\n\n')
        if in_heading:
            text.append('\n')
            self.last_was_heading = True
        else:
            self.last_was_heading = False
        if hasattr(elem, 'tail') and elem.tail:
            text.append(elem.tail)
        return text