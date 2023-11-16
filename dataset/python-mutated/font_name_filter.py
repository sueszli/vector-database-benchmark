"""
    This implementation of EventListener acts as a filter for other EventListener implementations.
    It only allows ChunkOfTextRenderEvent to pass if their corresponding font (name) matches.
"""
import typing
from borb.pdf.canvas.event.chunk_of_text_render_event import ChunkOfTextRenderEvent
from borb.pdf.canvas.event.event_listener import Event
from borb.pdf.canvas.event.event_listener import EventListener

class FontNameFilter(EventListener):
    """
    This implementation of EventListener acts as a filter for other EventListener implementations.
    It only allows ChunkOfTextRenderEvent to pass if their corresponding font (name) matches.
    """

    def __init__(self, font_name: str):
        if False:
            i = 10
            return i + 15
        self._font_name = font_name
        self._listeners: typing.List[EventListener] = []

    def _event_occurred(self, event: 'Event') -> None:
        if False:
            return 10
        if isinstance(event, ChunkOfTextRenderEvent):
            font_name: typing.Optional[str] = event.get_font().get_font_name()
            if font_name == self._font_name:
                for l in self._listeners:
                    l._event_occurred(event)
            return
        for l in self._listeners:
            l._event_occurred(event)

    def add_listener(self, listener: 'EventListener') -> 'FontNameFilter':
        if False:
            return 10
        '\n        This methods add an EventListener to this (meta)-EventListener\n        '
        self._listeners.append(listener)
        return self