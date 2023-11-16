"""
This implementation of EventListener resizes all Image objects on a Page
to fit their actual dimensions (ensuring they are not bigger than they need to be)
"""
import typing
from PIL import Image as PILImage
from borb.io.read.pdf_object import PDFObject
from borb.io.read.types import Name
from borb.pdf.canvas.event.begin_page_event import BeginPageEvent
from borb.pdf.canvas.event.event_listener import Event
from borb.pdf.canvas.event.event_listener import EventListener
from borb.pdf.canvas.event.image_render_event import ImageRenderEvent
from borb.pdf.page.page import Page

class ImageFormatOptimization(EventListener):
    """
    This implementation of EventListener resizes all Image objects on a Page
    to fit their actual dimensions (ensuring they are not bigger than they need to be)
    """

    def __init__(self):
        if False:
            return 10
        super(ImageFormatOptimization, self).__init__()
        self._current_page: typing.Optional[Page] = None

    def _event_occurred(self, event: 'Event') -> None:
        if False:
            while True:
                i = 10
        if isinstance(event, BeginPageEvent):
            self._current_page = event.get_page()
        if isinstance(event, ImageRenderEvent):
            self._render_image(event)

    def _render_image(self, image_render_event: 'ImageRenderEvent'):
        if False:
            for i in range(10):
                print('nop')
        source_image: PILImage = image_render_event.get_image()
        w0: int = int(image_render_event.get_width())
        h0: int = int(image_render_event.get_height())
        w1: int = source_image.width
        h1: int = source_image.height
        assert self._current_page is not None
        resource_name: typing.Optional[Name] = next(iter([k for (k, v) in self._current_page['Resources']['XObject'].items() if id(v) == id(source_image)]), None)
        assert resource_name is not None
        if w0 * h0 < w1 * h1:
            resized_image: PILImage = source_image.resize((w0, h0))
            PDFObject.add_pdf_object_methods(resized_image)
            self._current_page['Resources']['XObject'][resource_name] = resized_image
            resized_image.set_parent(self._current_page['Resources']['XObject'][resource_name])