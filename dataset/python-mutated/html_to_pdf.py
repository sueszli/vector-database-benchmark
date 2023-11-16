"""
This class converts HTML to PDF.
"""
import logging
import typing
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path
from lxml.etree import HTMLParser
from lxml.etree import _Comment
from borb.io.read.types import Dictionary
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.font.font import Font
from borb.pdf.canvas.font.simple_font.font_type_1 import StandardType1Font
from borb.pdf.canvas.layout.emoji.emoji import Emojis
from borb.pdf.canvas.layout.horizontal_rule import HorizontalRule
from borb.pdf.canvas.layout.image.image import Image
from borb.pdf.canvas.layout.layout_element import LayoutElement
from borb.pdf.canvas.layout.list.ordered_list import OrderedList
from borb.pdf.canvas.layout.list.unordered_list import UnorderedList
from borb.pdf.canvas.layout.page_layout.block_flow import BlockFlow
from borb.pdf.canvas.layout.page_layout.inline_flow import InlineFlow
from borb.pdf.canvas.layout.page_layout.page_layout import PageLayout
from borb.pdf.canvas.layout.page_layout.single_column_layout_with_overflow import SingleColumnLayoutWithOverflow
from borb.pdf.canvas.layout.table.flexible_column_width_table import FlexibleColumnWidthTable
from borb.pdf.canvas.layout.table.table import Table
from borb.pdf.canvas.layout.text.chunk_of_text import ChunkOfText
from borb.pdf.canvas.layout.text.heterogeneous_paragraph import LineBreakChunk
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.xref.plaintext_xref import PlainTextXREF
logger = logging.getLogger(__name__)

class HTMLToPDF:
    """
    This class converts HTML to PDF
    """

    class Context:
        """
        This class represents the context in which an (HTML) Element is
        being transformed into its LayoutElement representation. This context
        keeps track of whether an Element should be rendered bold, italic, or both,
        the font_size and the background_color.
        """

        def __init__(self):
            if False:
                print('Hello World!')
            self.fallback_fonts_regular: typing.List[Font] = [StandardType1Font('Helvetica')]
            self.fallback_fonts_italic: typing.List[Font] = [StandardType1Font('Helvetica-Oblique')]
            self.fallback_fonts_bold: typing.List[Font] = [StandardType1Font('Helvetica-Bold')]
            self.fallback_fonts_bold_italic: typing.List[Font] = [StandardType1Font('Helvetica-Bold-Oblique')]
            self.fallback_fonts_monospaced: typing.List[Font] = [StandardType1Font('Courier')]
            self.is_bold: bool = False
            self.is_italic: bool = False
            self.is_monospaced: bool = False
            self.default_font_size: Decimal = Decimal(12)
            self.font_size: Decimal = self.default_font_size
            self.font_color: typing.Optional[Color] = HexColor('000000')
            self.background_color: typing.Optional[Color] = None
            self.is_preformatted: bool = False
            self.document: typing.Optional[Document] = None

    @staticmethod
    def _build_chunk_of_text(s: str, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        fonts_to_try: typing.List[Font] = []
        if c.is_bold and c.is_italic:
            fonts_to_try = c.fallback_fonts_bold_italic
        elif c.is_bold:
            fonts_to_try = c.fallback_fonts_bold
        elif c.is_italic:
            fonts_to_try = c.fallback_fonts_italic
        else:
            fonts_to_try = c.fallback_fonts_regular
        if c.is_monospaced:
            fonts_to_try = c.fallback_fonts_monospaced
        for font_to_try in fonts_to_try:
            try:
                cot: ChunkOfText = ChunkOfText(s, font=font_to_try, font_size=c.font_size, font_color=c.font_color or HexColor('000000'), background_color=c.background_color)
                cot._write_text_bytes()
                return cot
            except:
                continue
        return ChunkOfText('□', font='Helvetica', font_size=c.font_size, font_color=c.font_color or HexColor('000000'), background_color=c.background_color)

    @staticmethod
    def _process_a_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            return 10
        assert e.tag == 'a'
        e.tag = 'span'
        prev_tail: typing.Optional[str] = e.tail
        e.tail = ''
        prev_font_color = c.font_color
        c.font_color = HexColor('#0645ad')
        out_value_001: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value_001 is not None
        c.font_color = prev_font_color
        e.tag = 'a'
        tmp: ET.Element = ET.Element('span')
        tmp.text = prev_tail
        out_value_002: typing.Optional[LayoutElement] = HTMLToPDF._process_element(tmp, c)
        assert out_value_002 is not None
        e.tail = prev_tail
        return InlineFlow().extend([out_value_001, out_value_002])

    @staticmethod
    def _process_abbr_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'abbr'
        e.tag = 'span'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'abbr'
        return out_value

    @staticmethod
    def _process_acronym_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'acronym'
        e.tag = 'span'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'acronym'
        return out_value

    @staticmethod
    def _process_address_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'address'
        e.tag = 'p'
        prev_is_italic: bool = c.is_italic
        c.is_italic = True
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        c.is_italic = prev_is_italic
        e.tag = 'address'
        return BlockFlow().extend([LineBreakChunk(), out_value, LineBreakChunk()])

    @staticmethod
    def _process_article_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            for i in range(10):
                print('nop')
        assert e.tag == 'article'
        e.tag = 'div'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'article'
        return out_value

    @staticmethod
    def _process_aside_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'aside'
        e.tag = 'div'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'aside'
        return out_value

    @staticmethod
    def _process_b_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'b'
        e.tag = 'strong'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'b'
        return out_value

    @staticmethod
    def _process_big_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'big'
        e.tag = 'span'
        prev_font_size: Decimal = c.font_size
        c.font_size *= Decimal(1.2)
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        c.font_size = prev_font_size
        e.tag = 'big'
        return out_value

    @staticmethod
    def _process_block_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            return 10
        block_layout_element: BlockFlow = BlockFlow()
        if e.text is not None and len(e.text) > 0:
            tmp_span: ET.Element = ET.Element('span')
            tmp_span.text = e.text
            tmp_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(tmp_span, c)
            assert tmp_value is not None
            block_layout_element.add(tmp_value)
        for child_element in e:
            if isinstance(child_element, _Comment):
                continue
            tmp_value = HTMLToPDF._process_element(child_element, c)
            assert tmp_value is not None
            block_layout_element.add(tmp_value)
        if e.tail is not None and len(e.tail) > 0:
            tmp_span = ET.Element('span')
            tmp_span.text = e.tail
            tmp_value = HTMLToPDF._process_element(tmp_span, c)
            assert tmp_value is not None
            block_layout_element.add(tmp_value)
        return block_layout_element

    @staticmethod
    def _process_blockquote_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'blockquote'
        prev_background_color: typing.Optional[Color] = c.background_color
        c.background_color = HexColor('F5F5F5')
        blockquote_element: typing.Optional[LayoutElement] = HTMLToPDF._process_block_element(e, c)
        assert blockquote_element is not None
        blockquote_element._background_color = c.background_color
        blockquote_element._padding_left = c.font_size * 2
        blockquote_element._border_left = True
        blockquote_element._border_width = Decimal(2)
        blockquote_element._border_color = HexColor('DCDCDC')
        c.background_color = prev_background_color
        return blockquote_element

    @staticmethod
    def _process_body_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            for i in range(10):
                print('nop')
        assert e.tag == 'body'
        return HTMLToPDF._process_block_element(e, c)

    @staticmethod
    def _process_br_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            for i in range(10):
                print('nop')
        assert e.tag == 'br'
        e.tag = 'span'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'br'
        return InlineFlow().extend([LineBreakChunk(), out_value])

    @staticmethod
    def _process_cite_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            for i in range(10):
                print('nop')
        assert e.tag == 'cite'
        e.tag = 'em'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'cite'
        return out_value

    @staticmethod
    def _process_code_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'code'
        e.tag = 'span'
        prev_tail: typing.Optional[str] = e.tail
        e.tail = ''
        prev_background_color = c.background_color
        prev_is_monospaced: bool = c.is_monospaced
        c.background_color = HexColor('#f5f5f5')
        c.is_monospaced = True
        out_value_001: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value_001 is not None
        c.background_color = prev_background_color
        c.is_monospaced = prev_is_monospaced
        e.tag = 'code'
        tmp: ET.Element = ET.Element('span')
        tmp.text = prev_tail
        out_value_002: typing.Optional[LayoutElement] = HTMLToPDF._process_element(tmp, c)
        assert out_value_002 is not None
        e.tail = prev_tail
        return InlineFlow().extend([out_value_001, out_value_002])

    @staticmethod
    def _process_dd_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            return 10
        assert e.tag == 'dd'
        e.tag = 'p'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        out_value._padding_left = Decimal(40)
        e.tag = 'dd'
        return out_value

    @staticmethod
    def _process_div_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'div'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_block_element(e, c)
        assert out_value is not None
        return out_value

    @staticmethod
    def _process_dl_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            for i in range(10):
                print('nop')
        assert e.tag == 'dl'
        e.tag = 'div'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'dl'
        return out_value

    @staticmethod
    def _process_dt_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            i = 10
            return i + 15
        assert e.tag == 'dt'
        e.tag = 'p'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'dt'
        return out_value

    @staticmethod
    def _process_element(e: ET.Element, c: Context) -> typing.Optional[LayoutElement]:
        if False:
            return 10
        ELEMENT_CREATOR_METHODS: typing.Dict[str, typing.Callable[[ET.Element, HTMLToPDF.Context], typing.Optional[LayoutElement]]] = {'a': HTMLToPDF._process_a_element, 'abbr': HTMLToPDF._process_abbr_element, 'acronym': HTMLToPDF._process_acronym_element, 'address': HTMLToPDF._process_address_element, 'article': HTMLToPDF._process_article_element, 'aside': HTMLToPDF._process_aside_element, 'b': HTMLToPDF._process_b_element, 'bdo': HTMLToPDF._process_unsupported_element, 'big': HTMLToPDF._process_big_element, 'blockquote': HTMLToPDF._process_blockquote_element, 'body': HTMLToPDF._process_body_element, 'br': HTMLToPDF._process_br_element, 'button': HTMLToPDF._process_unsupported_element, 'canvas': HTMLToPDF._process_unsupported_element, 'cite': HTMLToPDF._process_cite_element, 'code': HTMLToPDF._process_code_element, 'dd': HTMLToPDF._process_dd_element, 'dfn': HTMLToPDF._process_unsupported_element, 'div': HTMLToPDF._process_div_element, 'dl': HTMLToPDF._process_dl_element, 'dt': HTMLToPDF._process_dt_element, 'em': HTMLToPDF._process_em_element, 'fieldset': HTMLToPDF._process_unsupported_element, 'figcaption': HTMLToPDF._process_unsupported_element, 'figure': HTMLToPDF._process_figure_element, 'footer': HTMLToPDF._process_footer_element, 'form': HTMLToPDF._process_unsupported_element, 'h1': HTMLToPDF._process_hx_element, 'h2': HTMLToPDF._process_hx_element, 'h3': HTMLToPDF._process_hx_element, 'h4': HTMLToPDF._process_hx_element, 'h5': HTMLToPDF._process_hx_element, 'h6': HTMLToPDF._process_hx_element, 'head': HTMLToPDF._process_head_element, 'header': HTMLToPDF._process_header_element, 'hr': HTMLToPDF._process_hr_element, 'html': HTMLToPDF._process_html_element, 'i': HTMLToPDF._process_i_element, 'img': HTMLToPDF._process_img_element, 'kbd': HTMLToPDF._process_unsupported_element, 'li': HTMLToPDF._process_li_element, 'main': HTMLToPDF._process_main_element, 'map': HTMLToPDF._process_unsupported_element, 'mark': HTMLToPDF._process_mark_element, 'meta': HTMLToPDF._process_meta_element, 'nav': HTMLToPDF._process_unsupported_element, 'noscript': HTMLToPDF._process_noscript_element, 'ol': HTMLToPDF._process_ol_element, 'p': HTMLToPDF._process_p_element, 'pre': HTMLToPDF._process_pre_element, 'q': HTMLToPDF._process_q_element, 'samp': HTMLToPDF._process_samp_element, 'section': HTMLToPDF._process_section_element, 'small': HTMLToPDF._process_small_element, 'span': HTMLToPDF._process_inline_element, 'strong': HTMLToPDF._process_strong_element, 'table': HTMLToPDF._process_table_element, 'title': HTMLToPDF._process_title_element, 'tfoot': HTMLToPDF._process_unsupported_element, 'ul': HTMLToPDF._process_ul_element, 'video': HTMLToPDF._process_video_element}
        if ELEMENT_CREATOR_METHODS.get(e.tag, None) is None:
            logger.warning('<%s> unsupported' % e.tag)
            return None
        return ELEMENT_CREATOR_METHODS[e.tag](e, c)

    @staticmethod
    def _process_em_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'em'
        e.tag = 'span'
        prev_tail: typing.Optional[str] = e.tail
        e.tail = ''
        prev_is_italic = c.is_italic
        c.is_italic = True
        out_value_001: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value_001 is not None
        c.is_italic = prev_is_italic
        e.tag = 'em'
        tmp: ET.Element = ET.Element('span')
        tmp.text = prev_tail
        out_value_002: typing.Optional[LayoutElement] = HTMLToPDF._process_element(tmp, c)
        assert out_value_002 is not None
        e.tail = prev_tail
        return InlineFlow().extend([out_value_001, out_value_002])

    @staticmethod
    def _process_figure_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'figure'
        figcaption: typing.Optional[ET.Element] = next(iter([x for x in e if x.tag == 'figcaption']), None)
        img: typing.Optional[ET.Element] = next(iter([x for x in e if x.tag == 'img']), None)
        number_of_rows: int = 0
        if figcaption is not None:
            number_of_rows += 1
        if img is not None:
            number_of_rows += 1
        table: Table = FlexibleColumnWidthTable(number_of_columns=1, number_of_rows=number_of_rows)
        if img is not None:
            tmp_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(img, c)
            assert tmp_value is not None
            table.add(tmp_value)
        if figcaption is not None:
            tmp_value = HTMLToPDF._process_element(figcaption, c)
            assert tmp_value is not None
            table.add(tmp_value)
        table.set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        table.no_borders()
        return table

    @staticmethod
    def _process_footer_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            i = 10
            return i + 15
        assert e.tag == 'footer'
        e.tag = 'div'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'footer'
        return out_value

    @staticmethod
    def _process_head_element(e: ET.Element, c: Context) -> None:
        if False:
            return 10
        assert e.tag == 'head'
        if 'title' in [x.tag for x in e]:
            HTMLToPDF._process_element([x for x in e if x.tag == 'title'][0], c)
        for meta_element in [x for x in e if x.tag == 'meta']:
            HTMLToPDF._process_element(meta_element, c)

    @staticmethod
    def _process_header_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'header'
        e.tag = 'div'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'header'
        return out_value

    @staticmethod
    def _process_hr_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'hr'
        tmp: ET.Element = ET.Element('span')
        tmp.text = e.tail
        out_value_002: typing.Optional[LayoutElement] = HTMLToPDF._process_element(tmp, c)
        assert out_value_002 is not None
        return BlockFlow().extend([HorizontalRule(), out_value_002])

    @staticmethod
    def _process_html_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        if 'head' in [x.tag for x in e]:
            HTMLToPDF._process_element([x for x in e if x.tag == 'head'][0], c)
        if 'body' in [x.tag for x in e]:
            out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element([x for x in e if x.tag == 'body'][0], c)
            assert out_value is not None
            return out_value
        return BlockFlow()

    @staticmethod
    def _process_hx_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            i = 10
            return i + 15
        assert e.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        FONT_SIZE_MODIFIER: typing.Dict[str, Decimal] = {'h1': Decimal(2), 'h2': Decimal(1.5), 'h3': Decimal(1.17), 'h4': Decimal(1), 'h5': Decimal(0.83), 'h6': Decimal(0.67)}
        prev_font_size: Decimal = c.font_size
        prev_is_bold: bool = c.is_bold
        c.font_size = c.default_font_size * FONT_SIZE_MODIFIER[e.tag]
        c.is_bold = True
        header_layout_element: LayoutElement = BlockFlow().add(HTMLToPDF._process_inline_element(e, c))
        header_layout_element._padding_bottom = c.font_size
        header_layout_element._padding_top = c.font_size
        c.font_size = prev_font_size
        c.is_bold = prev_is_bold
        return header_layout_element

    @staticmethod
    def _process_i_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'i'
        e.tag = 'em'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'i'
        return out_value

    @staticmethod
    def _process_img_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            i = 10
            return i + 15
        assert e.tag == 'img'
        src: str = e.attrib['src']
        w: typing.Optional[Decimal] = Decimal(e.attrib['width']) if 'width' in e.attrib else None
        h: typing.Optional[Decimal] = Decimal(e.attrib['height']) if 'height' in e.attrib else None
        out_value_001: LayoutElement = Image(src, width=w, height=h)
        tmp: ET.Element = ET.Element('span')
        tmp.text = e.tail
        out_value_002: typing.Optional[LayoutElement] = HTMLToPDF._process_element(tmp, c)
        assert out_value_002 is not None
        return InlineFlow().extend([out_value_001, out_value_002])

    @staticmethod
    def _process_inline_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            for i in range(10):
                print('nop')
        chunks: typing.List[LayoutElement] = []
        if e.tag == 'span' and 'EMOJI' in e.attrib.get('class', '').upper().split(' '):
            emoji: str = [x for x in e.attrib.get('class', '').upper().split(' ') if x.startswith('EMOJI_')][0][6:]
            chunks.append(Emojis[emoji].value)
            e.text = ''
        if e.text is not None and len(e.text) > 0:
            if c.is_preformatted:
                ws: typing.List[str] = [x for x in e.text.split('\n')]
                for w in ws:
                    chunks.append(HTMLToPDF._build_chunk_of_text(w, c))
                    chunks.append(LineBreakChunk())
            else:
                ws = [x for x in e.text.split(' ')]
                ws = [x + ' ' for x in ws[0:len(ws) - 1]] + [ws[-1]]
                chunks.extend([HTMLToPDF._build_chunk_of_text(w, c) for w in ws])
        for child_element in e:
            if isinstance(child_element, _Comment):
                continue
            tmp_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(child_element, c)
            assert tmp_value is not None
            chunks.append(tmp_value)
        if e.tail is not None and len(e.tail) > 0:
            ws = [x for x in e.tail.split(' ')]
            ws = [x + ' ' for x in ws[0:len(ws) - 1]] + [ws[-1]]
            chunks.extend([HTMLToPDF._build_chunk_of_text(w, c) for w in ws])
        if len(chunks) == 0:
            chunks.append(HTMLToPDF._build_chunk_of_text(' ', c))
        return InlineFlow().extend(chunks)

    @staticmethod
    def _process_li_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'li'
        return HTMLToPDF._process_block_element(e, c)

    @staticmethod
    def _process_main_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            for i in range(10):
                print('nop')
        assert e.tag == 'main'
        return HTMLToPDF._process_block_element(e, c)

    @staticmethod
    def _process_mark_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'mark'
        e.tag = 'span'
        prev_tail: typing.Optional[str] = e.tail
        e.tail = ''
        prev_background_color = c.background_color
        c.background_color = HexColor('#ffff00')
        out_value_001: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value_001 is not None
        c.background_color = prev_background_color
        e.tag = 'mark'
        tmp: ET.Element = ET.Element('span')
        tmp.text = prev_tail
        out_value_002: typing.Optional[LayoutElement] = HTMLToPDF._process_element(tmp, c)
        assert out_value_002 is not None
        e.tail = prev_tail
        return InlineFlow().extend([out_value_001, out_value_002])

    @staticmethod
    def _process_meta_element(e: ET.Element, c: Context) -> None:
        if False:
            return 10
        assert e.tag == 'meta'
        document: typing.Optional[Document] = c.document
        if document is None:
            return
        name: str = e.attrib.get('name', '')
        content: str = e.attrib.get('content', '')
        if 'XRef' not in document:
            document[Name('XRef')] = PlainTextXREF()
        if 'Trailer' not in document['XRef']:
            document['XRef'][Name('Trailer')] = Dictionary()
        if 'Info' not in document['XRef']['Trailer']:
            document['XRef']['Trailer'][Name('Info')] = Dictionary()
        if name == 'keywords':
            document['XRef']['Trailer']['Info'][Name('Keywords')] = String(content)
        if name == 'description':
            document['XRef']['Trailer']['Info'][Name('Subject')] = String(content)
        if name == 'author':
            document['XRef']['Trailer']['Info'][Name('Author')] = String(content)

    @staticmethod
    def _process_noscript_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            return 10
        assert e.tag == 'noscript'
        e.tag = 'p'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'noscript'
        return out_value

    @staticmethod
    def _process_ol_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'ol'
        ol_layout_element: OrderedList = OrderedList()
        for child_element in e:
            if child_element.tag == ET.Comment:
                continue
            tmp_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(child_element, c)
            assert tmp_value is not None
            ol_layout_element.add(tmp_value)
        return BlockFlow().add(ol_layout_element)

    @staticmethod
    def _process_p_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            i = 10
            return i + 15
        assert e.tag == 'p'
        return BlockFlow().add(HTMLToPDF._process_inline_element(e, c))

    @staticmethod
    def _process_pre_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            i = 10
            return i + 15
        assert e.tag == 'pre'
        e.tag = 'div'
        prev_is_preformatted: bool = c.is_preformatted
        c.is_preformatted = True
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        c.is_preformatted = prev_is_preformatted
        e.tag = 'pre'
        return out_value

    @staticmethod
    def _process_q_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            i = 10
            return i + 15
        assert e.tag == 'q'
        e.tag = 'span'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'q'
        return InlineFlow().extend([HTMLToPDF._build_chunk_of_text('"', c), out_value, HTMLToPDF._build_chunk_of_text('"', c)])

    @staticmethod
    def _process_samp_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'samp'
        e.tag = 'span'
        prev_is_monospaced: bool = c.is_monospaced
        c.is_monospaced = True
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        c.is_monospaced = prev_is_monospaced
        e.tag = 'samp'
        return out_value

    @staticmethod
    def _process_section_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            return 10
        assert e.tag == 'section'
        e.tag = 'div'
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        e.tag = 'section'
        return out_value

    @staticmethod
    def _process_small_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            return 10
        assert e.tag == 'small'
        e.tag = 'span'
        prev_font_size: Decimal = c.font_size
        c.font_size /= Decimal(1.2)
        out_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value is not None
        c.font_size = prev_font_size
        e.tag = 'small'
        return out_value

    @staticmethod
    def _process_strong_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            while True:
                i = 10
        assert e.tag == 'strong'
        e.tag = 'span'
        prev_tail: typing.Optional[str] = e.tail
        e.tail = ''
        prev_is_bold = c.is_bold
        c.is_bold = True
        out_value_001: typing.Optional[LayoutElement] = HTMLToPDF._process_element(e, c)
        assert out_value_001 is not None
        c.is_bold = prev_is_bold
        e.tag = 'strong'
        tmp: ET.Element = ET.Element('span')
        tmp.text = prev_tail
        out_value_002: typing.Optional[LayoutElement] = HTMLToPDF._process_element(tmp, c)
        assert out_value_002 is not None
        e.tail = prev_tail
        return InlineFlow().extend([out_value_001, out_value_002])

    @staticmethod
    def _process_table_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            i = 10
            return i + 15
        assert e.tag == 'table'
        if 'tbody' in [x.tag for x in e]:
            return HTMLToPDF._process_table_element_001(e, c)
        else:
            return HTMLToPDF._process_table_element_002(e, c)

    @staticmethod
    def _process_table_element_001(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        thead: typing.Optional[ET.Element] = next(iter([x for x in e if x.tag == 'thead']), None)
        tbody: ET.Element = next(iter([x for x in e if x.tag == 'tbody']))
        nrows_000: int = len([x for x in tbody if x.tag == 'tr']) + (1 if thead is not None else 0)
        ncols_000: int = sum([int(y.attrib.get('colspan', '1')) for y in [x for x in tbody if x.tag == 'tr'][0] if y.tag in ['td', 'th']])
        table: Table = FlexibleColumnWidthTable(number_of_rows=nrows_000, number_of_columns=ncols_000)
        if thead is not None:
            for tr in [x for x in thead if x.tag == 'tr']:
                for th in [y for y in tr if y.tag == 'th']:
                    table.add(HTMLToPDF._process_inline_element(th, c))
        for tr in [x for x in tbody if x.tag == 'tr']:
            for td in [y for y in tr if y.tag == 'td']:
                table.add(HTMLToPDF._process_inline_element(td, c))
        table.set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        table._margin_top = c.default_font_size
        table._margin_bottom = c.default_font_size
        return table

    @staticmethod
    def _process_table_element_002(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        nrows_001: int = len([x for x in e if x.tag == 'tr'])
        ncols_001: int = len([y for y in [x for x in e if x.tag == 'tr'][0] if y.tag == 'td'])
        table: Table = FlexibleColumnWidthTable(number_of_rows=nrows_001, number_of_columns=ncols_001)
        for tr in [x for x in e if x.tag == 'tr']:
            for td in [y for y in tr if y.tag == 'td']:
                tmp_value: typing.Optional[LayoutElement] = HTMLToPDF._process_inline_element(td, c)
                assert tmp_value is not None
                table.add(tmp_value)
        table.set_padding_on_all_cells(Decimal(2), Decimal(2), Decimal(2), Decimal(2))
        table._margin_top = c.default_font_size
        table._margin_bottom = c.default_font_size
        return table

    @staticmethod
    def _process_title_element(e: ET.Element, c: Context) -> None:
        if False:
            print('Hello World!')
        assert e.tag == 'title'
        document: typing.Optional[Document] = c.document
        if document is None:
            return
        if 'XRef' not in document:
            document[Name('XRef')] = PlainTextXREF()
        if 'Trailer' not in document['XRef']:
            document['XRef'][Name('Trailer')] = Dictionary()
        if 'Info' not in document['XRef']['Trailer']:
            document['XRef']['Trailer'][Name('Info')] = Dictionary()
        document['XRef']['Trailer']['Info'][Name('Title')] = String(e.text or '')

    @staticmethod
    def _process_ul_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            return 10
        assert e.tag == 'ul'
        ul_layout_element: UnorderedList = UnorderedList()
        for child_element in e:
            if child_element.tag == ET.Comment:
                continue
            tmp_value: typing.Optional[LayoutElement] = HTMLToPDF._process_element(child_element, c)
            assert tmp_value is not None
            ul_layout_element.add(tmp_value)
        return ul_layout_element

    @staticmethod
    def _process_unsupported_element(e: ET.Element, c: Context) -> typing.Optional[LayoutElement]:
        if False:
            i = 10
            return i + 15
        logger.warning('<%s> unsupported' % e.tag)
        return None

    @staticmethod
    def _process_video_element(e: ET.Element, c: Context) -> LayoutElement:
        if False:
            print('Hello World!')
        assert e.tag == 'video'
        w: typing.Optional[Decimal] = Decimal(e.attrib['width']) if 'width' in e.attrib else None
        h: typing.Optional[Decimal] = Decimal(e.attrib['height']) if 'height' in e.attrib else None
        src: str = e.attrib['src']
        try:
            import cv2
            import tempfile
            video_capture = cv2.VideoCapture(src)
            (success, nd_array_image) = video_capture.read()
            tmp_file: Path = Path(tempfile.NamedTemporaryFile(suffix='.jpg').name)
            cv2.imwrite(str(tmp_file), nd_array_image)
            return Image(tmp_file, width=w, height=h)
        except:
            logger.warning('Unable to retrieve frame for <video> element, defaulting to empty Image.')
            from PIL import Image as pImage
            return Image(pImage.new(mode='RGB', size=(int(w), int(h)), color=(113, 121, 126)), width=w, height=h)

    @staticmethod
    def convert_html_to_layout_element(html: typing.Union[str, ET.Element], fallback_fonts_regular: typing.List[Font]=[StandardType1Font('Helvetica')], fallback_fonts_bold: typing.List[Font]=[StandardType1Font('Helvetica-Bold')], fallback_fonts_italic: typing.List[Font]=[StandardType1Font('Helvetica-Oblique')], fallback_fonts_bold_italic: typing.List[Font]=[StandardType1Font('Helvetica-Bold-Oblique')], fallback_fonts_monospaced: typing.List[Font]=[StandardType1Font('Courier')]) -> LayoutElement:
        if False:
            return 10
        '\n        This function converts a html str to a LayoutElement\n        :param html:                        the html str (or ET.Element) to be converted\n        :param fallback_fonts_regular:      fallback (regular) fonts to try when the default font is unable to render a character\n        :param fallback_fonts_bold:         fallback (bold) fonts to try when the default font is unable to render a character\n        :param fallback_fonts_italic:       fallback (italic) fonts to try when the default font is unable to render a character\n        :param fallback_fonts_bold_italic:  fallback (bold, italic) fonts to try when the default font is unable to render a character\n        :return:\n        '
        root_element: typing.Optional[ET.Element] = None
        if isinstance(html, str):
            root_element = ET.fromstring(html, HTMLParser())
        else:
            root_element = html
        assert root_element is not None
        c: HTMLToPDF.Context = HTMLToPDF.Context()
        c.fallback_fonts_regular = fallback_fonts_regular
        c.fallback_fonts_bold = fallback_fonts_bold
        c.fallback_fonts_italic = fallback_fonts_italic
        c.fallback_fonts_bold_italic = fallback_fonts_bold_italic
        c.fallback_fonts_monospaced = fallback_fonts_monospaced
        return HTMLToPDF._process_html_element(root_element, c)

    @staticmethod
    def convert_html_to_pdf(html: typing.Union[str, ET.Element], fallback_fonts_regular: typing.List[Font]=[StandardType1Font('Helvetica')], fallback_fonts_bold: typing.List[Font]=[StandardType1Font('Helvetica-Bold')], fallback_fonts_italic: typing.List[Font]=[StandardType1Font('Helvetica-Oblique')], fallback_fonts_bold_italic: typing.List[Font]=[StandardType1Font('Helvetica-Bold-Oblique')], fallback_fonts_monospaced: typing.List[Font]=[StandardType1Font('Courier')]) -> Document:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function converts HTML to PDF\n        '
        doc: Document = Document()
        page: Page = Page()
        doc.add_page(page)
        layout: PageLayout = SingleColumnLayoutWithOverflow(page)
        root_element: typing.Optional[ET.Element] = None
        if isinstance(html, str):
            root_element = ET.fromstring(html, HTMLParser())
        else:
            root_element = html
        assert root_element is not None
        c: HTMLToPDF.Context = HTMLToPDF.Context()
        c.document = doc
        c.fallback_fonts_regular = fallback_fonts_regular
        c.fallback_fonts_bold = fallback_fonts_bold
        c.fallback_fonts_italic = fallback_fonts_italic
        c.fallback_fonts_bold_italic = fallback_fonts_bold_italic
        c.fallback_fonts_monospaced = fallback_fonts_monospaced
        layout.add(HTMLToPDF._process_html_element(root_element, c))
        return doc