"""This module implements a base class for popups"""
from collections import defaultdict
from abc import abstractmethod
from asciimatics.event import KeyboardEvent, MouseEvent
from asciimatics.exceptions import InvalidFields
from asciimatics.screen import Screen
from asciimatics.widgets.frame import Frame

class _TempPopup(Frame):
    """
    An internal Frame for creating a temporary pop-up for a Widget in another Frame.
    """

    def __init__(self, screen, parent, x, y, w, h):
        if False:
            return 10
        '\n        :param screen: The Screen being used for this pop-up.\n        :param parent: The widget that spawned this pop-up.\n        :param x: The X coordinate for the desired pop-up.\n        :param y: The Y coordinate for the desired pop-up.\n        :param w: The width of the desired pop-up.\n        :param h: The height of the desired pop-up.\n        '
        super().__init__(screen, h, w, x=x, y=y, has_border=True, can_scroll=False, is_modal=True)
        self.palette = defaultdict(lambda : parent.frame.palette['focus_field'])
        self.palette['selected_field'] = parent.frame.palette['selected_field']
        self.palette['selected_focus_field'] = parent.frame.palette['selected_focus_field']
        self.palette['invalid'] = parent.frame.palette['invalid']
        self._parent = parent

    def process_event(self, event):
        if False:
            return 10
        cancelled = False
        if event is not None:
            if isinstance(event, KeyboardEvent):
                if event.key_code in [Screen.ctrl('M'), Screen.ctrl('J'), ord(' ')]:
                    event = None
                elif event.key_code == Screen.KEY_ESCAPE:
                    event = None
                    cancelled = True
            elif isinstance(event, MouseEvent) and event.buttons != 0:
                if self._outside_frame(event):
                    event = None
        if event is None:
            try:
                self.close(cancelled)
            except InvalidFields:
                pass
        return super().process_event(event)

    def close(self, cancelled=False):
        if False:
            return 10
        '\n        Close this temporary pop-up.\n\n        :param cancelled: Whether the pop-up was cancelled (e.g. by pressing Esc).\n        '
        self._on_close(cancelled)
        self._scene.remove_effect(self)

    @abstractmethod
    def _on_close(self, cancelled):
        if False:
            print('Hello World!')
        '\n        Method to handle any communication back to the parent widget on closure of this pop-up.\n\n        :param cancelled: Whether the pop-up was cancelled (e.g. by pressing Esc).\n\n        This method can raise an InvalidFields exception to indicate that the current selection is\n        invalid and so the pop-up cannot be dismissed.\n        '