"""
HTML-TOC-adding transform.
"""
__license__ = 'GPL v3'
__copyright__ = '2008, Marshall T. Vandegrift <llasram@gmail.com>'
from calibre.ebooks.oeb.base import CSS_MIME, XHTML, XHTML_MIME, XHTML_NS, XML, XPath, element
from calibre.utils.localization import __
__all__ = ['HTMLTOCAdder']
DEFAULT_TITLE = __('Table of Contents')
STYLE_CSS = {'nested': '\n.calibre_toc_header {\n  text-align: center;\n}\n.calibre_toc_block {\n  margin-left: 1.2em;\n  text-indent: -1.2em;\n}\n.calibre_toc_block .calibre_toc_block {\n  margin-left: 2.4em;\n}\n.calibre_toc_block .calibre_toc_block .calibre_toc_block {\n  margin-left: 3.6em;\n}\n', 'centered': '\n.calibre_toc_header {\n  text-align: center;\n}\n.calibre_toc_block {\n  text-align: center;\n}\nbody > .calibre_toc_block {\n  margin-top: 1.2em;\n}\n'}

class HTMLTOCAdder:

    def __init__(self, title=None, style='nested', position='end'):
        if False:
            return 10
        self.title = title
        self.style = style
        self.position = position

    @classmethod
    def config(cls, cfg):
        if False:
            print('Hello World!')
        group = cfg.add_group('htmltoc', _('HTML TOC generation options.'))
        group('toc_title', ['--toc-title'], default=None, help=_('Title for any generated in-line table of contents.'))
        return cfg

    @classmethod
    def generate(cls, opts):
        if False:
            for i in range(10):
                print('nop')
        return cls(title=opts.toc_title)

    def __call__(self, oeb, context):
        if False:
            return 10
        has_toc = getattr(getattr(oeb, 'toc', False), 'nodes', False)
        if 'toc' in oeb.guide:
            from calibre.ebooks.oeb.base import urlnormalize
            href = urlnormalize(oeb.guide['toc'].href)
            if href in oeb.manifest.hrefs:
                item = oeb.manifest.hrefs[href]
                if hasattr(item.data, 'xpath') and XPath('//h:a[@href]')(item.data):
                    if oeb.spine.index(item) < 0:
                        if self.position == 'end':
                            oeb.spine.add(item, linear=False)
                        else:
                            oeb.spine.insert(0, item, linear=True)
                    return
                elif has_toc:
                    oeb.guide.remove('toc')
            else:
                oeb.guide.remove('toc')
        if not has_toc:
            return
        oeb.logger.info('Generating in-line TOC...')
        title = self.title or oeb.translate(DEFAULT_TITLE)
        style = self.style
        if style not in STYLE_CSS:
            oeb.logger.error('Unknown TOC style %r' % style)
            style = 'nested'
        (id, css_href) = oeb.manifest.generate('tocstyle', 'tocstyle.css')
        oeb.manifest.add(id, css_href, CSS_MIME, data=STYLE_CSS[style])
        language = str(oeb.metadata.language[0])
        contents = element(None, XHTML('html'), nsmap={None: XHTML_NS}, attrib={XML('lang'): language})
        head = element(contents, XHTML('head'))
        htitle = element(head, XHTML('title'))
        htitle.text = title
        element(head, XHTML('link'), rel='stylesheet', type=CSS_MIME, href=css_href)
        body = element(contents, XHTML('body'), attrib={'class': 'calibre_toc'})
        h1 = element(body, XHTML('h2'), attrib={'class': 'calibre_toc_header'})
        h1.text = title
        self.add_toc_level(body, oeb.toc)
        (id, href) = oeb.manifest.generate('contents', 'contents.xhtml')
        item = oeb.manifest.add(id, href, XHTML_MIME, data=contents)
        if self.position == 'end':
            oeb.spine.add(item, linear=False)
        else:
            oeb.spine.insert(0, item, linear=True)
        oeb.guide.add('toc', 'Table of Contents', href)

    def add_toc_level(self, elem, toc):
        if False:
            i = 10
            return i + 15
        for node in toc:
            block = element(elem, XHTML('div'), attrib={'class': 'calibre_toc_block'})
            line = element(block, XHTML('a'), attrib={'href': node.href, 'class': 'calibre_toc_line'})
            line.text = node.title
            self.add_toc_level(block, node)