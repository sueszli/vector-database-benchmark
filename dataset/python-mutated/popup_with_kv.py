"""
Example to show a Popup usage with the content from kv lang.
"""
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.app import App
from kivy.lang import Builder
Builder.load_string("\n<CustomPopup>:\n    size_hint: .5, .5\n    auto_dismiss: False\n    title: 'Hello world'\n    Button:\n        text: 'Click me to dismiss'\n        on_press: root.dismiss()\n\n")

class CustomPopup(Popup):
    pass

class TestApp(App):

    def build(self):
        if False:
            i = 10
            return i + 15
        b = Button(on_press=self.show_popup, text='Show Popup')
        return b

    def show_popup(self, b):
        if False:
            while True:
                i = 10
        p = CustomPopup()
        p.open()
TestApp().run()