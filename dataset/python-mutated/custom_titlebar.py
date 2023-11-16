from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.logger import Logger
kv = '\n#:import rgba kivy.utils.rgba\n<TitleBar>:\n    id:title_bar\n    size_hint: 1,0.1\n    pos_hint : {\'top\':0.5}\n    BoxLayout:\n        orientation:"vertical"\n        BoxLayout:\n            Button:\n                text: "Click-able"\n                draggable:False\n            Button:\n                text: "non Click-able"\n            Button:\n                text: "non Click-able"\n        BoxLayout:\n            draggable:False\n            Button:\n                text: "Click-able"\n            Button:\n                text: "click-able"\n            Button:\n                text: "Click-able"\n\nFloatLayout:\n'

class TitleBar(BoxLayout):
    pass

class CustomTitleBar(App):

    def build(self):
        if False:
            i = 10
            return i + 15
        root = Builder.load_string(kv)
        Window.custom_titlebar = True
        title_bar = TitleBar()
        root.add_widget(title_bar)
        if Window.set_custom_titlebar(title_bar):
            Logger.info('Window: setting custom titlebar successful')
        else:
            Logger.info('Window: setting custom titlebar Not allowed on this system ')
        self.title = 'MyApp'
        return root
if __name__ == '__main__':
    CustomTitleBar().run()