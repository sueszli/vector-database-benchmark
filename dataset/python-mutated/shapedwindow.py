from kivy.config import Config
Config.set('graphics', 'shaped', 1)
from kivy.resources import resource_find
default_shape = Config.get('kivy', 'window_shape')
alpha_shape = resource_find('data/logo/kivy-icon-512.png')
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import BooleanProperty, StringProperty, ListProperty
Builder.load_string("\n#:import win kivy.core.window.Window\n\n<Root>:\n    orientation: 'vertical'\n    BoxLayout:\n        Button:\n            text: 'default_shape'\n            on_release: app.shape_image = app.default_shape\n        Button:\n            text: 'alpha_shape'\n            on_release: app.shape_image = app.alpha_shape\n\n    BoxLayout:\n        ToggleButton:\n            group: 'mode'\n            text: 'default'\n            state: 'down'\n            on_release: win.shape_mode = 'default'\n        ToggleButton:\n            group: 'mode'\n            text: 'binalpha'\n            on_release: win.shape_mode = 'binalpha'\n        ToggleButton:\n            group: 'mode'\n            text: 'reversebinalpha'\n            on_release: win.shape_mode = 'reversebinalpha'\n        ToggleButton:\n            group: 'mode'\n            text: 'colorkey'\n            on_release: win.shape_mode = 'colorkey'\n\n    BoxLayout:\n        ToggleButton:\n            group: 'cutoff'\n            text: 'cutoff True'\n            state: 'down'\n            on_release: win.shape_cutoff = True\n        ToggleButton:\n            group: 'cutoff'\n            text: 'cutoff False'\n            on_release: win.shape_cutoff = False\n\n    BoxLayout:\n        ToggleButton:\n            group: 'colorkey'\n            text: '1, 1, 1, 1'\n            state: 'down'\n            on_release: win.shape_color_key = [1, 1, 1, 1]\n        ToggleButton:\n            group: 'colorkey'\n            text: '0, 0, 0, 1'\n            on_release: win.shape_color_key = [0, 0, 0, 1]\n")

class Root(BoxLayout):
    pass

class ShapedWindow(App):
    shape_image = StringProperty('', force_dispatch=True)

    def on_shape_image(self, instance, value):
        if False:
            print('Hello World!')
        if 'kivy-icon' in value:
            Window.size = (512, 512)
            Window.shape_image = self.alpha_shape
        else:
            Window.size = (800, 600)
            Window.shape_image = self.default_shape

    def build(self):
        if False:
            return 10
        self.default_shape = default_shape
        self.alpha_shape = alpha_shape
        return Root()
if __name__ == '__main__':
    ShapedWindow().run()