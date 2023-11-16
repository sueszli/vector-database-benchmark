from kivy.app import App
from kivy.lang import Builder
kv = "\n#:import window kivy.core.window.Window\n\nBoxLayout:\n    orientation: 'vertical'\n    Label:\n        text: f'Window opacity: {window.opacity}'\n        font_size: '25sp'\n    Slider:\n        size_hint_y: 4\n        min: 0.0\n        max: 1.0\n        value: window.opacity\n        on_value: window.opacity = args[1]\n"

class WindowOpacityApp(App):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        return Builder.load_string(kv)
if __name__ == '__main__':
    WindowOpacityApp().run()