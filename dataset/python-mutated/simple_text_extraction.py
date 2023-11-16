"""
    This implementation of EventListener extracts all text from a PDF Document
"""
import io
import typing
from decimal import Decimal
from functools import cmp_to_key
from borb.pdf.canvas.canvas import Canvas
from borb.pdf.canvas.canvas_stream_processor import CanvasStreamProcessor
from borb.pdf.canvas.event.begin_page_event import BeginPageEvent
from borb.pdf.canvas.event.chunk_of_text_render_event import ChunkOfTextRenderEvent
from borb.pdf.canvas.event.chunk_of_text_render_event import LeftToRightComparator
from borb.pdf.canvas.event.end_page_event import EndPageEvent
from borb.pdf.canvas.event.event_listener import Event
from borb.pdf.canvas.event.event_listener import EventListener
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page

class SimpleTextExtraction(EventListener):
    """
    This implementation of EventListener extracts all text from a PDF Document
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._text_render_info_per_page: typing.Dict[int, typing.List[ChunkOfTextRenderEvent]] = {}
        self._text_per_page: typing.Dict[int, str] = {}
        self._current_page: int = -1

    def _begin_page(self, page: Page):
        if False:
            while True:
                i = 10
        self._current_page += 1

    def _end_page(self, page: Page):
        if False:
            return 10
        tris = self._text_render_info_per_page[self._current_page] if self._current_page in self._text_render_info_per_page else []
        tris = [x for x in tris if x.get_text() is not None]
        tris = [x for x in tris if len(x.get_text().replace(' ', '')) != 0]
        if len(tris) == 0:
            return
        tris = sorted(tris, key=cmp_to_key(LeftToRightComparator.cmp))
        last_baseline_bottom = tris[0].get_baseline().y
        last_baseline_right = tris[0].get_baseline().x
        text = ''
        for t in tris:
            if abs(t.get_baseline().y - last_baseline_bottom) > 10 and len(text) > 0:
                if text.endswith(' '):
                    text = text[0:-1]
                text += '\n'
                text += t.get_text()
                last_baseline_right = t.get_baseline().x + t.get_baseline().width
                last_baseline_bottom = t.get_baseline().y
                continue
            if t.get_text().startswith(' ') or text.endswith(' '):
                text += t.get_text()
                last_baseline_right = t.get_baseline().x + t.get_baseline().width
                continue
            delta = abs(last_baseline_right - t.get_baseline().x)
            space_width = round(t.get_space_character_width_estimate_in_user_space(), 1)
            text += ' ' if space_width * Decimal(0.9) < delta else ''
            text += t.get_text()
            last_baseline_right = t.get_baseline().x + t.get_baseline().width
            continue
        self._text_per_page[self._current_page] = text

    def _event_occurred(self, event: Event) -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(event, ChunkOfTextRenderEvent):
            self._render_text(event)
        if isinstance(event, BeginPageEvent):
            self._begin_page(event.get_page())
        if isinstance(event, EndPageEvent):
            self._end_page(event.get_page())

    def _render_text(self, text_render_info: ChunkOfTextRenderEvent):
        if False:
            for i in range(10):
                print('nop')
        if self._current_page not in self._text_render_info_per_page:
            self._text_render_info_per_page[self._current_page] = []
        self._text_render_info_per_page[self._current_page].append(text_render_info)

    def get_text(self) -> typing.Dict[int, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns all text on a given page\n        '
        return self._text_per_page

    @staticmethod
    def get_text_from_pdf(pdf: Document) -> typing.Dict[int, str]:
        if False:
            print('Hello World!')
        '\n        This function returns the text for a given PDF (per page)\n        :param pdf:     the PDF to be analyzed\n        :return:        the text per page (represented by typing.Dict[int, str])\n        '
        text_per_page: typing.Dict[int, str] = {}
        number_of_pages: int = int(pdf.get_document_info().get_number_of_pages() or 0)
        for page_nr in range(0, number_of_pages):
            page: Page = pdf.get_page(page_nr)
            page_source: io.BytesIO = io.BytesIO(page['Contents']['DecodedBytes'])
            l: 'SimpleTextExtraction' = SimpleTextExtraction()
            l._event_occurred(BeginPageEvent(page))
            CanvasStreamProcessor(page, Canvas(), []).read(page_source, [l])
            l._event_occurred(EndPageEvent(page))
            text_per_page[page_nr] = l.get_text()[0]
        return text_per_page