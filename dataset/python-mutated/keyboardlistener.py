import kivy
kivy.require('1.0.8')
from kivy.core.window import Window
from kivy.uix.widget import Widget

class MyKeyboardListener(Widget):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(MyKeyboardListener, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self, 'text')
        if self._keyboard.widget:
            pass
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _keyboard_closed(self):
        if False:
            while True:
                i = 10
        print('My keyboard have been closed!')
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if False:
            return 10
        print('The key', keycode, 'have been pressed')
        print(' - text is %r' % text)
        print(' - modifiers are %r' % modifiers)
        if keycode[1] == 'escape':
            keyboard.release()
        return True
if __name__ == '__main__':
    from kivy.base import runTouchApp
    runTouchApp(MyKeyboardListener())