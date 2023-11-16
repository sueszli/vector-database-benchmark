"""
    This implementation of Event is triggered right before the Canvas is being processed.
"""
from borb.pdf.canvas.event.event_listener import Event
from borb.pdf.page.page import Page

class BeginPageEvent(Event):
    """
    This implementation of Event is triggered right before the Canvas is being processed.
    """

    def __init__(self, page: Page):
        if False:
            while True:
                i = 10
        self._page: Page = page

    def get_page(self) -> Page:
        if False:
            print('Hello World!')
        '\n        This function returns the Page that triggered this BeginPageEvent\n        '
        return self._page