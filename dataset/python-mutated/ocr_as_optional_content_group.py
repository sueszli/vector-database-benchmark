"""
    This class adds performs OCR and adds recognized text in an optional content group on the PDF.
    This enables the user to have a searchable PDF, whilst being able to turn on/off OCR features.
"""
from __future__ import annotations
import datetime
import typing
import zlib
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING
from borb.datastructure.disjoint_set import disjointset
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.pdf.canvas.event.event_listener import Event
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.text.chunk_of_text import ChunkOfText
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.toolkit.ocr.ocr_image_render_event_listener import OCREvent
from borb.toolkit.ocr.ocr_image_render_event_listener import OCRImageRenderEventListener
EndDocumentEvent = type(None)
if TYPE_CHECKING:
    pass

class OCRAsOptionalContentGroup(OCRImageRenderEventListener):
    """
    This class adds performs OCR and adds recognized text in an optional content group on the PDF.
    This enables the user to have a searchable PDF, whilst being able to turn on/off OCR features.
    """

    def __init__(self, tesseract_data_dir: Path, minimal_confidence: Decimal=Decimal(0.75)):
        if False:
            return 10
        super(OCRAsOptionalContentGroup, self).__init__(tesseract_data_dir, minimal_confidence)
        self._ocr_events: typing.List[OCREvent] = []

    def _add_ocr_optional_content_group(self, document: Document) -> None:
        if False:
            while True:
                i = 10
        if 'OCProperties' not in document['XRef']['Trailer']['Root']:
            document['XRef']['Trailer']['Root'][Name('OCProperties')] = Dictionary()
            document['XRef']['Trailer']['Root']['OCProperties'][Name('OCGs')] = List()
            document['XRef']['Trailer']['Root']['OCProperties'][Name('D')] = Dictionary()
        ocg_dict: Dictionary = Dictionary()
        ocg_dict[Name('Type')] = Name('OCG')
        ocg_dict[Name('Name')] = String('OCR by borb')
        ocg_dict[Name('Intent')] = Name('View')
        document['XRef']['Trailer']['Root']['OCProperties'][Name('OCGs')].append(ocg_dict)
        now = datetime.datetime.now()
        ocr_layer_internal_name: str = 'ocr%d%d%d' % (now.year, now.month, now.day)
        number_of_pages: typing.Optional[Decimal] = document.get_document_info().get_number_of_pages()
        assert number_of_pages is not None
        for page_nr in range(0, int(number_of_pages)):
            page: Page = document.get_page(page_nr)
            if 'Resources' not in page:
                page[Name('Resources')] = Dictionary
            if 'Properties' not in page['Resources']:
                page['Resources'][Name('Properties')] = Dictionary()
            page['Resources']['Properties'][Name(ocr_layer_internal_name)] = ocg_dict
            ocr_events_per_page: typing.List[OCREvent] = [x for x in self._ocr_events if x.get_page() == page]
            if len(ocr_events_per_page) == 0:
                continue
            ds: disjointset = disjointset()
            for e in ocr_events_per_page:
                ds.add(e)
            for e1 in ocr_events_per_page:
                for e2 in ocr_events_per_page:
                    if e1 == e2:
                        continue
                    if self._overlaps_vertically(e1.get_bounding_box(), e2.get_bounding_box()):
                        ds.union(e1, e2)
            for es in ds.sets():
                avg_y: Decimal = Decimal(sum([x.get_bounding_box().get_y() for x in es]) / len(es))
                for e in es:
                    e.get_bounding_box().y = avg_y
            page['Contents'][Name('DecodedBytes')] += ('\n/OC /%s BDC\n' % ocr_layer_internal_name).encode('latin1')
            for e in ocr_events_per_page:
                ChunkOfText(e.get_text(), e.get_font(), e.get_font_size(), e.get_font_color()).paint(page, e.get_bounding_box())
            page['Contents'][Name('DecodedBytes')] += '\nEMC'.encode('latin1')
            page['Contents'][Name('Bytes')] = zlib.compress(page['Contents']['DecodedBytes'], 9)
            page['Contents'][Name('Length')] = bDecimal(len(page['Contents'][Name('Bytes')]))

    def _end_document(self):
        if False:
            while True:
                i = 10
        if len(self._ocr_events) == 0:
            return
        document: Document = self._ocr_events[0].get_page().get_document()
        self._add_ocr_optional_content_group(document)

    def _event_occurred(self, event: Event) -> None:
        if False:
            return 10
        super(OCRAsOptionalContentGroup, self)._event_occurred(event)
        if event.__class__.__name__ == 'EndDocumentEvent':
            self._end_document()

    def _ocr_text_occurred(self, event: OCREvent):
        if False:
            for i in range(10):
                print('nop')
        self._ocr_events.append(event)

    def _overlaps_vertically(self, r0: Rectangle, r1: Rectangle) -> bool:
        if False:
            while True:
                i = 10
        '\n        This function returns True iff two Rectangle objects overlap vertically, False otherwise.\n        '
        return int(r0.get_y()) <= int(r1.get_y()) <= int(r0.get_y() + r0.get_height()) or int(r1.get_y()) <= int(r0.get_y()) <= int(r1.get_y() + r1.get_height())