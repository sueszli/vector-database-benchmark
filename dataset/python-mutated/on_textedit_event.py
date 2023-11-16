"""
on_textedit event sample.
"""
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.uix.textinput import TextInput
from kivy.base import EventLoop

class TextInputIME(TextInput):
    testtext = StringProperty()

    def __init__(self, **kwargs):
        if False:
            return 10
        super(TextInputIME, self).__init__(**kwargs)
        EventLoop.window.bind(on_textedit=self._on_textedit)

    def _on_textedit(self, window, text):
        if False:
            i = 10
            return i + 15
        self.testtext = text

class MainWidget(Widget):
    text = StringProperty()

    def __init__(self, **kwargs):
        if False:
            return 10
        super(MainWidget, self).__init__(**kwargs)
        self.text = ''

    def confim(self):
        if False:
            return 10
        self.text = self.ids['text_box'].text

    def changeFont(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            LabelBase.register(DEFAULT_FONT, self.ids['text_font'].text)
        except Exception:
            self.ids['text_font'].text = "can't load font."

class TextEditTestApp(App):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(TextEditTestApp, self).__init__(**kwargs)

    def build(self):
        if False:
            return 10
        return MainWidget()
if __name__ == '__main__':
    Builder.load_string('\n<MainWidget>:\n    BoxLayout:\n        orientation: \'vertical\'\n        size: root.size\n        BoxLayout:\n            Label:\n                size_hint_x: 3\n                text: "Multi language font file path"\n            TextInput:\n                id: text_font\n                size_hint_x: 5\n            Button:\n                size_hint_x: 2\n                text: "Change Font"\n                on_press: root.changeFont()\n        BoxLayout:\n            Label:\n                size_hint_x: 3\n                text: "Text editing by IME"\n            Label:\n                size_hint_x: 7\n                text:text_box.testtext\n                canvas.before:\n                    Color:\n                        rgb: 0.5765 ,0.5765 ,0.5843\n                    Rectangle:\n                        pos: self.pos\n                        size: self.size\n        BoxLayout:\n            Label:\n                size_hint_x: 3\n                text: "Enter text ->"\n            TextInputIME:\n                id: text_box\n                size_hint_x: 7\n                focus: True\n        BoxLayout:\n            Button:\n                size_hint_x: 3\n                text: "Confirm text property"\n                on_press: root.confim()\n            Label:\n                size_hint_x: 7\n                text: root.text\n                canvas.before:\n                    Color:\n                        rgb: 0.5765 ,0.5765 ,0.5843\n                    Rectangle:\n                        pos: self.pos\n                        size: self.size\n    ')
    TextEditTestApp().run()