"""
Demonstrates using kv language to create some simple buttons and a
label, with each button modifying the label text.
"""
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
Builder.load_string("\n<MainWidget>:\n    BoxLayout:\n        orientation: 'vertical'\n        Button:\n            text: 'some string '\n            on_press: the_right_pane.text += self.text\n        Button:\n            text: 'one two three four '\n            on_press: the_right_pane.text += self.text\n        Button:\n            text: 'follow the yellow brick road '\n            on_press: the_right_pane.text += self.text\n        Button:\n            text: 'five six seven eight '\n            on_press: the_right_pane.text += self.text\n        Button:\n            text: 'CLEAR LABEL'\n            on_press: the_right_pane.text = ''\n    Label:\n        id: the_right_pane\n        text: ''\n        text_size: self.size\n        halign: 'center'\n        valign: 'middle'\n")

class MainWidget(BoxLayout):
    pass

class ExampleApp(App):

    def build(self):
        if False:
            print('Hello World!')
        return MainWidget()
ExampleApp().run()