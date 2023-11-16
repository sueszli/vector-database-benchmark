from typing import ClassVar

class LayoutWidget:
    """
    All top-level layout widgets and all widgets that may be set in an
    overlay must comply with this API.
    """
    title = ''
    keyctx: ClassVar[str] = ''

    def key_responder(self):
        if False:
            while True:
                i = 10
        '\n        Returns the object responding to key input. Usually self, but may be\n        a wrapped object.\n        '
        return self

    def focus_changed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The view focus has changed. Layout objects should implement the API\n        rather than directly subscribing to events.\n        '

    def view_changed(self):
        if False:
            while True:
                i = 10
        '\n        The view list has changed.\n        '

    def layout_popping(self):
        if False:
            while True:
                i = 10
        '\n        We are just about to pop a window off the stack, or exit an overlay.\n        '

    def layout_pushed(self, prev):
        if False:
            for i in range(10):
                print('nop')
        '\n        We have just pushed a window onto the stack.\n        '