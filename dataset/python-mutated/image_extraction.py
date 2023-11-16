"""
This implementation of EventListener extracts all Image objects on a Page
"""
import io
import typing
from PIL import Image as PILImage
from borb.pdf.canvas.canvas import Canvas
from borb.pdf.canvas.canvas_stream_processor import CanvasStreamProcessor
from borb.pdf.canvas.event.begin_page_event import BeginPageEvent
from borb.pdf.canvas.event.end_page_event import EndPageEvent
from borb.pdf.canvas.event.event_listener import Event
from borb.pdf.canvas.event.event_listener import EventListener
from borb.pdf.canvas.event.image_render_event import ImageRenderEvent
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page

class ImageExtraction(EventListener):
    """
    This implementation of EventListener extracts all Image objects on a Page
    """

    def __init__(self):
        if False:
            return 10
        '\n        Constructs a new SimpleImageExtraction\n        '
        self._image_render_info_per_page = {}
        self._current_page: int = -1

    def _begin_page(self, page: Page):
        if False:
            return 10
        self._current_page += 1

    def _event_occurred(self, event: 'Event') -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(event, BeginPageEvent):
            self._begin_page(event.get_page())
        if isinstance(event, ImageRenderEvent):
            self._render_image(event)

    def _render_image(self, image_render_event: 'ImageRenderEvent'):
        if False:
            for i in range(10):
                print('nop')
        if self._current_page not in self._image_render_info_per_page:
            self._image_render_info_per_page[self._current_page] = []
        self._image_render_info_per_page[self._current_page].append(image_render_event.get_image())

    def get_images(self) -> typing.Dict[int, typing.List[PILImage.Image]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns a typing.List[Image] on a given page\n        '
        return self._image_render_info_per_page

    @staticmethod
    def get_images_from_pdf(pdf: Document) -> typing.Dict[int, typing.List[PILImage.Image]]:
        if False:
            print('Hello World!')
        '\n        This function returns the images used in a given PDF\n        :param pdf:     the PDF to be analysed\n        :return:        the images (typing.List[PILImage.Image]) in the PDF\n        '
        images_of_each_page: typing.Dict[int, typing.List[PILImage.Image]] = {}
        number_of_pages: int = int(pdf.get_document_info().get_number_of_pages() or 0)
        for page_nr in range(0, number_of_pages):
            page: Page = pdf.get_page(page_nr)
            page_source: io.BytesIO = io.BytesIO(page['Contents']['DecodedBytes'])
            cse: 'ImageExtraction' = ImageExtraction()
            cse._event_occurred(BeginPageEvent(page))
            CanvasStreamProcessor(page, Canvas(), []).read(page_source, [cse])
            cse._event_occurred(EndPageEvent(page))
            images_of_each_page[page_nr] = cse.get_images()[0]
        return images_of_each_page