"""
This class converts Markdown to PDF
"""
import typing
import xml.etree.ElementTree as ET
from decimal import Decimal
from lxml.etree import Element
from lxml.etree import HTMLParser
from markdown_it import MarkdownIt
from borb.pdf.canvas.font.font import Font
from borb.pdf.canvas.font.simple_font.font_type_1 import StandardType1Font
from borb.pdf.canvas.layout.emoji.emoji import Emojis
from borb.pdf.canvas.layout.layout_element import LayoutElement
from borb.pdf.document.document import Document
from borb.pdf.page.page_size import PageSize
from borb.toolkit.export.html_to_pdf.html_to_pdf import HTMLToPDF

class MarkdownToPDF:
    """
    This class converts Markdown to PDF
    """

    @staticmethod
    def _replace_github_flavored_emoji(e: Element, parents: typing.List[Element]=[]) -> Element:
        if False:
            for i in range(10):
                print('nop')
        if e.tag == 'span' and 'emoji' in e.get('class', '').split(' '):
            return e
        TAGS_TO_IGNORE: typing.List[str] = ['code', 'pre']
        element_can_be_changed: bool = len([True for x in parents if x.tag in TAGS_TO_IGNORE]) == 0 and e.tag not in TAGS_TO_IGNORE
        text_exists: bool = e.text is not None and len(e.text) > 0
        if text_exists and element_can_be_changed:
            for (k, v) in [(':' + x.lower() + ':', x) for x in dir(Emojis)]:
                if k in e.text:
                    n: int = e.text.find(k)
                    before: str = e.text[0:n]
                    after: str = e.text[n + len(k):]
                    e.text = before
                    span: Element = Element('span')
                    span.set('class', 'emoji emoji_%s' % v)
                    span.text = k
                    span.tail = after
                    e.insert(0, span)
        for x in e:
            MarkdownToPDF._replace_github_flavored_emoji(x, parents + [e])
        tail_exists: bool = e.tail is not None and len(e.tail) > 0
        if tail_exists and element_can_be_changed:
            for (k, v) in [(':' + x.lower() + ':', x) for x in dir(Emojis)]:
                if k in e.tail:
                    n = e.tail.find(k)
                    before = e.tail[0:n]
                    after = e.tail[n + len(k):]
                    e.tail = after
                    span = Element('span')
                    span.set('class', 'emoji emoji_%s' % v)
                    span.text = k
                    span.tail = before
                    parent: Element = parents[-1]
                    index_of_e_in_parent: int = [i for (i, x) in enumerate(parent) if x == e][0]
                    parent.insert(index_of_e_in_parent, span)
        return e

    @staticmethod
    def _set_img_width_and_height(e: Element) -> Element:
        if False:
            while True:
                i = 10
        if e.tag == 'img':
            w: typing.Optional[int] = e.attrib['width'] if 'width' in e.attrib else None
            h: typing.Optional[int] = e.attrib['height'] if 'height' in e.attrib else None
            if w is None or h is None or w > PageSize.A4_PORTRAIT.value[0] or (h > PageSize.A4_PORTRAIT.value[1]):
                w = int(PageSize.A4_PORTRAIT.value[0] * Decimal(0.8))
                h = int(w * 0.618)
                e.attrib['width'] = str(w)
                e.attrib['height'] = str(h)
        for x in e:
            MarkdownToPDF._set_img_width_and_height(x)
        return e

    @staticmethod
    def convert_markdown_to_layout_element(markdown: str, fallback_fonts_regular: typing.List[Font]=[StandardType1Font('Helvetica')], fallback_fonts_bold: typing.List[Font]=[StandardType1Font('Helvetica-Bold')], fallback_fonts_italic: typing.List[Font]=[StandardType1Font('Helvetica-Oblique')], fallback_fonts_bold_italic: typing.List[Font]=[StandardType1Font('Helvetica-Bold-Oblique')]) -> LayoutElement:
        if False:
            i = 10
            return i + 15
        '\n        This function converts a markdown str to a LayoutElement\n        :param markdown:                    the markdown str to be converted\n        :param fallback_fonts_regular:      fallback (regular) fonts to try when the default font is unable to render a character\n        :param fallback_fonts_bold:         fallback (bold) fonts to try when the default font is unable to render a character\n        :param fallback_fonts_italic:       fallback (italic) fonts to try when the default font is unable to render a character\n        :param fallback_fonts_bold_italic:  fallback (bold, italic) fonts to try when the default font is unable to render a character\n        :return:\n        '
        html: str = MarkdownIt().enable('table').render(markdown)
        html_root: ET.Element = ET.fromstring(html, HTMLParser())
        html_root = MarkdownToPDF._replace_github_flavored_emoji(html_root)
        html_root = MarkdownToPDF._set_img_width_and_height(html_root)
        return HTMLToPDF.convert_html_to_layout_element(html_root, fallback_fonts_regular, fallback_fonts_bold, fallback_fonts_italic, fallback_fonts_bold_italic)

    @staticmethod
    def convert_markdown_to_pdf(markdown: str, fallback_fonts_regular: typing.List[Font]=[StandardType1Font('Helvetica')], fallback_fonts_bold: typing.List[Font]=[StandardType1Font('Helvetica-Bold')], fallback_fonts_italic: typing.List[Font]=[StandardType1Font('Helvetica-Oblique')], fallback_fonts_bold_italic: typing.List[Font]=[StandardType1Font('Helvetica-Bold-Oblique')]) -> Document:
        if False:
            print('Hello World!')
        '\n        This function converts a markdown str to a Document\n        :param markdown:                    the markdown str to be converted\n        :param fallback_fonts_regular:      fallback (regular) fonts to try when the default font is unable to render a character\n        :param fallback_fonts_bold:         fallback (bold) fonts to try when the default font is unable to render a character\n        :param fallback_fonts_italic:       fallback (italic) fonts to try when the default font is unable to render a character\n        :param fallback_fonts_bold_italic:  fallback (bold, italic) fonts to try when the default font is unable to render a character\n        :return:\n        '
        html: str = MarkdownIt().enable('table').render(markdown)
        html_root: ET.Element = ET.fromstring(html, HTMLParser())
        html_root = MarkdownToPDF._replace_github_flavored_emoji(html_root)
        html_root = MarkdownToPDF._set_img_width_and_height(html_root)
        return HTMLToPDF.convert_html_to_pdf(html_root, fallback_fonts_regular, fallback_fonts_bold, fallback_fonts_italic, fallback_fonts_bold_italic)