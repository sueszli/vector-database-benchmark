from kivy.uix.gridlayout import GridLayout
from kivy.app import App
from kivy.lang import Builder
Builder.load_string("\n<Demo>:\n    cols: 1\n\n    BoxLayout:\n        orientation: 'vertical'\n        Button:\n            size_hint_x: 0.4\n            pos_hint: {'x': 0}\n            text: 'pos_hint: x=0'\n\n        Button:\n            size_hint_x: 0.2\n            pos_hint: {'center_x': 0.5}\n            text: 'pos_hint: center_x=0.5'\n\n        Button:\n            size_hint_x: 0.4\n            pos_hint: {'right': 1}\n            text: 'pos_hint: right=1'\n\n    BoxLayout:\n        Button:\n            size_hint_y: 0.4\n            pos_hint: {'y': 0}\n            text: 'pos_hint: y=0'\n\n        Button:\n            size_hint_y: 0.2\n            pos_hint: {'center_y': .5}\n            text: 'pos_hint: center_y=0.5'\n\n        Button:\n            size_hint_y: 0.4\n            pos_hint: {'top': 1}\n            text: 'pos_hint: top=1'\n")

class Demo(GridLayout):
    pass

class DemoApp(App):

    def build(self):
        if False:
            while True:
                i = 10
        return Demo()
if __name__ == '__main__':
    DemoApp().run()