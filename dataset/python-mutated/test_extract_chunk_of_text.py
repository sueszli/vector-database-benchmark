import random
import re
import typing
import unittest
from decimal import Decimal
from borb.pdf import ChunkOfText
from borb.pdf import ConnectedShape
from borb.pdf import HeterogeneousParagraph
from borb.pdf import HexColor
from borb.pdf import LineArtFactory
from borb.pdf import Lipsum
from borb.pdf import PageLayout
from borb.pdf.canvas.event.event_listener import Event
from borb.pdf.canvas.event.event_listener import EventListener
from borb.pdf.canvas.layout.layout_element import Alignment
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from borb.toolkit.text.font_name_filter import FontNameFilter
from borb.toolkit.text.simple_text_extraction import SimpleTextExtraction
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestExtractChunkOfText(TestCase):

    def test_create_dummy_pdf(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout: PageLayout = SingleColumnLayout(page)
        random.seed(0)
        layout.add(HeterogeneousParagraph([ChunkOfText(x + ' ') for x in Lipsum.generate_lipsum_text(5).split(' ')]))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_extract_chunks_of_text(self):
        if False:
            print('Hello World!')

        class PrivateEventListener(EventListener):

            def __init__(self):
                if False:
                    return 10
                self._chunks_of_text: typing.List[ChunkOfText] = []

            def _event_occurred(self, event: Event) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(event, ChunkOfText):
                    self._chunks_of_text += [event]

            def get_chunks_of_text(self) -> typing.List[ChunkOfText]:
                if False:
                    print('Hello World!')
                return self._chunks_of_text
        doc: typing.Optional[Document] = None
        l: PrivateEventListener = PrivateEventListener()
        with open(self.get_first_output_file(), 'rb') as fh:
            doc = PDF.loads(fh, [l])
        assert doc is not None
        for c in l.get_chunks_of_text():
            ConnectedShape(LineArtFactory.rectangle(c.get_previous_layout_box()), fill_color=None, stroke_color=HexColor('ff0000')).paint(doc.get_page(0), c.get_previous_layout_box())
        with open(self.get_second_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, doc)
        self.check_pdf_using_validator(self.get_second_output_file())