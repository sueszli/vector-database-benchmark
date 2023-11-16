"""
This implementation of WriteBaseTransformer is responsible for determining which characters (typing.Set[str])
are used by which Font. This is particularly useful when performing Font subsetting.
"""
import typing
from borb.pdf.canvas.event.chunk_of_text_render_event import ChunkOfTextRenderEvent
from borb.pdf.canvas.event.event_listener import Event
from borb.pdf.canvas.event.event_listener import EventListener
from borb.pdf.canvas.font.font import Font

class CharacterSetListener(EventListener):
    """
    This implementation of WriteBaseTransformer is responsible for determining which characters (typing.Set[str])
    are used by which Font. This is particularly useful when performing Font subsetting.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(CharacterSetListener, self).__init__()
        self._character_set_per_font: typing.Dict[Font, typing.Set[str]] = {}

    def _event_occurred(self, event: Event) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(event, ChunkOfTextRenderEvent):
            f: Font = event.get_font()
            if f in self._character_set_per_font:
                s: typing.Set[str] = self._character_set_per_font[f]
                for c in event.get_text():
                    s.add(c)
                self._character_set_per_font[f] = s
            else:
                self._character_set_per_font[f] = set([x for x in event.get_text()])

    def get_character_set_per_font(self) -> typing.Dict[Font, typing.Set[str]]:
        if False:
            return 10
        '\n        This function returns the character set (typing.Set[str]) used by each Font\n        :return:    the character set used by each Font\n        '
        return self._character_set_per_font