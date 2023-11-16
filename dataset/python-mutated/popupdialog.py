"""This module implements a Pop up dialog message box"""
from inspect import isfunction
from functools import partial
from wcwidth import wcswidth
from asciimatics.widgets.button import Button
from asciimatics.widgets.frame import Frame
from asciimatics.widgets.layout import Layout
from asciimatics.widgets.textbox import TextBox
from asciimatics.widgets.utilities import _split_text

class PopUpDialog(Frame):
    """
    A fixed implementation Frame that provides a standard message box dialog.
    """

    def __init__(self, screen, text, buttons, on_close=None, has_shadow=False, theme='warning'):
        if False:
            return 10
        '\n        :param screen: The Screen that owns this dialog.\n        :param text: The message text to display.\n        :param buttons: A list of button names to display. This may be an empty list.\n        :param on_close: Optional function to invoke on exit.\n        :param has_shadow: optional flag to specify if dialog should have a shadow when drawn.\n        :param theme: optional colour theme for this pop-up.  Defaults to the warning colours.\n\n        The `on_close` method (if specified) will be called with one integer parameter that\n        corresponds to the index of the button passed in the array of available `buttons`.\n\n        Note that `on_close` must be a static method to work across screen resizing.  Either it\n        is static (and so the dialog will be cloned) or it is not (and the dialog will disappear\n        when the screen is resized).\n        '
        self._text = text
        self._buttons = buttons
        self._on_close = on_close
        string_len = wcswidth if screen.unicode_aware else len
        width = max((string_len(x) for x in text.split('\n')))
        width = max(width + 2, sum((string_len(x) + 4 for x in buttons)) + len(buttons) + 5)
        width = min(width, screen.width * 2 // 3)
        delta_h = 4 if len(buttons) > 0 else 2
        self._message = _split_text(text, width - 2, screen.height - delta_h, screen.unicode_aware)
        height = len(self._message) + delta_h
        self._data = {'message': self._message}
        super().__init__(screen, height, width, self._data, has_shadow=has_shadow, is_modal=True)
        layout = Layout([width - 2], fill_frame=True)
        self.add_layout(layout)
        text_box = TextBox(len(self._message), name='message')
        text_box.disabled = True
        layout.add_widget(text_box)
        layout2 = Layout([1 for _ in buttons])
        self.add_layout(layout2)
        for (i, button) in enumerate(buttons):
            func = partial(self._destroy, i)
            layout2.add_widget(Button(button, func), i)
        self.fix()
        self.set_theme(theme)

    def _destroy(self, selected):
        if False:
            print('Hello World!')
        self._scene.remove_effect(self)
        if self._on_close:
            self._on_close(selected)

    def clone(self, screen, scene):
        if False:
            print('Hello World!')
        '\n        Create a clone of this Dialog into a new Screen.\n\n        :param screen: The new Screen object to clone into.\n        :param scene: The new Scene object to clone into.\n        '
        if self._on_close is None or isfunction(self._on_close):
            scene.add_effect(PopUpDialog(screen, self._text, self._buttons, self._on_close))