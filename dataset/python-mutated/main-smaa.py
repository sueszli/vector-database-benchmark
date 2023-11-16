import sys
from glob import glob
from os.path import join, dirname
from kivy.uix.scatter import Scatter
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.app import App
from kivy.graphics.svg import Svg
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
smaa_ui = "\n#:kivy 1.8.0\n\nBoxLayout:\n    orientation: 'horizontal'\n    pos_hint: {'top': 1}\n    size_hint_y: None\n    height: '48dp'\n    padding: '2dp'\n    spacing: '2dp'\n    Label:\n        text: 'Quality:'\n    ToggleButton:\n        text: 'Low'\n        group: 'smaa-quality'\n        on_release: app.smaa.quality = 'low'\n    ToggleButton:\n        text: 'Medium'\n        group: 'smaa-quality'\n        on_release: app.smaa.quality = 'medium'\n    ToggleButton:\n        text: 'High'\n        group: 'smaa-quality'\n        on_release: app.smaa.quality = 'high'\n    ToggleButton:\n        text: 'Ultra'\n        group: 'smaa-quality'\n        state: 'down'\n        on_release: app.smaa.quality = 'ultra'\n\n    Label:\n        text: 'Debug:'\n    ToggleButton:\n        text: 'None'\n        group: 'smaa-debug'\n        state: 'down'\n        on_release: app.smaa.debug = ''\n    ToggleButton:\n        text: 'Source'\n        group: 'smaa-debug'\n        on_release: app.smaa.debug = 'source'\n    ToggleButton:\n        text: 'Edges'\n        group: 'smaa-debug'\n        on_release: app.smaa.debug = 'edges'\n    ToggleButton:\n        text: 'Blend'\n        group: 'smaa-debug'\n        on_release: app.smaa.debug = 'blend'\n\n"

class SvgWidget(Scatter):

    def __init__(self, filename):
        if False:
            i = 10
            return i + 15
        super(SvgWidget, self).__init__()
        with self.canvas:
            svg = Svg(filename)
        self.size = (svg.width, svg.height)

class SvgApp(App):

    def build(self):
        if False:
            print('Hello World!')
        from kivy.garden.smaa import SMAA
        Window.bind(on_keyboard=self._on_keyboard_handler)
        self.smaa = SMAA()
        self.effects = [self.smaa, Widget()]
        self.effect_index = 0
        self.label = Label(text='SMAA', top=Window.height)
        self.effect = effect = self.effects[0]
        self.root = FloatLayout()
        self.root.add_widget(effect)
        if 0:
            from kivy.graphics import Color, Rectangle
            wid = Widget(size=Window.size)
            with wid.canvas:
                Color(1, 1, 1, 1)
                Rectangle(size=Window.size)
            effect.add_widget(wid)
        if 1:
            filenames = sys.argv[1:]
            if not filenames:
                filenames = glob(join(dirname(__file__), '*.svg'))
            for filename in filenames:
                svg = SvgWidget(filename)
                effect.add_widget(svg)
            effect.add_widget(self.label)
            svg.scale = 5.0
            svg.center = Window.center
        if 0:
            wid = Scatter(size=Window.size)
            from kivy.graphics import Color, Triangle, Rectangle
            with wid.canvas:
                Color(0, 0, 0, 1)
                Rectangle(size=Window.size)
                Color(1, 1, 1, 1)
                (w, h) = Window.size
                (cx, cy) = (w / 2.0, h / 2.0)
                Triangle(points=[cx - w * 0.25, cy - h * 0.25, cx, cy + h * 0.25, cx + w * 0.25, cy - h * 0.25])
            effect.add_widget(wid)
        if 0:
            from kivy.uix.button import Button
            from kivy.uix.slider import Slider
            effect.add_widget(Button(text='Hello World'))
            effect.add_widget(Slider(pos=(200, 200)))
        control_ui = Builder.load_string(smaa_ui)
        self.root.add_widget(control_ui)

    def _on_keyboard_handler(self, instance, key, *args):
        if False:
            while True:
                i = 10
        if key == 32:
            self.effect_index = (self.effect_index + 1) % 2
            childrens = self.effect.children[:]
            self.effect.clear_widgets()
            self.root.remove_widget(self.effect)
            self.effect = self.effects[self.effect_index]
            self.root.add_widget(self.effect)
            for child in reversed(childrens):
                self.effect.add_widget(child)
            self.label.text = self.effect.__class__.__name__
            Window.title = self.label.text
if __name__ == '__main__':
    SvgApp().run()