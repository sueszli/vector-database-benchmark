"""
    This implementation of Event is triggered right after the Canvas has been processed.
"""
from borb.pdf.canvas.event.event_listener import Event
from borb.pdf.page.page import Page

class EndPageEvent(Event):
    """
    This implementation of Event is triggered right after the Canvas has been processed.
    """

    def __init__(self, page: Page):
        if False:
            i = 10
            return i + 15
        self._page: Page = page

    def get_page(self) -> Page:
        if False:
            i = 10
            return i + 15
        '\n        This function returns the Page that triggered this BeginPageEvent\n        '
        return self._page