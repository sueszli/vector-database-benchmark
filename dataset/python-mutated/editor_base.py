import pyxel
from .settings import TOOL_BUCKET, TOOL_CIRC, TOOL_CIRCB, TOOL_PENCIL, TOOL_RECT, TOOL_RECTB, TOOL_SELECT
from .widgets import Widget

class EditorBase(Widget):
    """
    Variables:
        help_message_var

    Events:
        undo (data)
        redo (data)
        drop (filename)
    """

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        super().__init__(parent, 0, 0, 0, 0, is_visible=False)
        self._history_list = []
        self._history_index = 0
        self.copy_var('help_message_var', parent)

    @property
    def can_undo(self):
        if False:
            i = 10
            return i + 15
        return self._history_index > 0

    @property
    def can_redo(self):
        if False:
            for i in range(10):
                print('nop')
        return self._history_index < len(self._history_list)

    def undo(self):
        if False:
            while True:
                i = 10
        if not self.can_undo:
            return
        self._history_index -= 1
        self.trigger_event('undo', self._history_list[self._history_index])

    def redo(self):
        if False:
            i = 10
            return i + 15
        if not self.can_redo:
            return
        self.trigger_event('redo', self._history_list[self._history_index])
        self._history_index += 1

    def add_history(self, data):
        if False:
            for i in range(10):
                print('nop')
        self._history_list = self._history_list[:self._history_index]
        self._history_list.append(data)
        self._history_index += 1

    def reset_history(self):
        if False:
            for i in range(10):
                print('nop')
        self._history_list = []
        self._history_index = 0

    def add_number_picker_help(self, number_picker):
        if False:
            i = 10
            return i + 15
        number_picker.dec_button.add_event_listener('mouse_hover', self.__on_number_picker_dec_mouse_hover)
        number_picker.inc_button.add_event_listener('mouse_hover', self.__on_number_picker_inc_mouse_hover)

    def __on_number_picker_dec_mouse_hover(self, x, y):
        if False:
            return 10
        self.help_message_var = '-10:SHIFT+CLICK'

    def __on_number_picker_inc_mouse_hover(self, x, y):
        if False:
            i = 10
            return i + 15
        self.help_message_var = '+10:SHIFT+CLICK'

    def check_tool_button_shortcuts(self):
        if False:
            for i in range(10):
                print('nop')
        if pyxel.btn(pyxel.KEY_CTRL) or pyxel.btn(pyxel.KEY_ALT) or pyxel.btn(pyxel.KEY_GUI):
            return
        if pyxel.btnp(pyxel.KEY_S):
            self.tool_var = TOOL_SELECT
        elif pyxel.btnp(pyxel.KEY_P):
            self.tool_var = TOOL_PENCIL
        elif pyxel.btnp(pyxel.KEY_R):
            self.tool_var = TOOL_RECT if pyxel.btn(pyxel.KEY_SHIFT) else TOOL_RECTB
        elif pyxel.btnp(pyxel.KEY_C):
            self.tool_var = TOOL_CIRC if pyxel.btn(pyxel.KEY_SHIFT) else TOOL_CIRCB
        elif pyxel.btnp(pyxel.KEY_B):
            self.tool_var = TOOL_BUCKET

    def add_tool_button_help(self, tool_button):
        if False:
            print('Hello World!')
        tool_button.add_event_listener('mouse_hover', self.__on_tool_button_mouse_hover)

    def __on_tool_button_mouse_hover(self, x, y):
        if False:
            i = 10
            return i + 15
        value = self._tool_button.check_value(x, y)
        if value == TOOL_SELECT:
            s = 'SELECT:S'
        elif value == TOOL_PENCIL:
            s = 'PENCIL:P'
        elif value == TOOL_RECTB:
            s = 'RECTANGLE:R'
        elif value == TOOL_RECT:
            s = 'FILLED-RECT:SHIFT+R'
        elif value == TOOL_CIRCB:
            s = 'CIRCLE:C'
        elif value == TOOL_CIRC:
            s = 'FILLED-CIRC:SHIFT+C'
        elif value == TOOL_BUCKET:
            s = 'BUCKET:B'
        else:
            s = ''
        self.help_message_var = s