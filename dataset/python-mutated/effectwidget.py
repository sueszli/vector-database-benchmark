"""
Example usage of the effectwidget.

Currently highly experimental.
"""
from kivy.app import App
from kivy.uix.effectwidget import EffectWidget
from kivy.uix.spinner import Spinner
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.effectwidget import MonochromeEffect, InvertEffect, ChannelMixEffect, ScanlinesEffect, FXAAEffect, PixelateEffect, HorizontalBlurEffect, VerticalBlurEffect

class ComparisonWidget(EffectWidget):
    pass

class EffectSpinner(Spinner):
    pass

class SpinnerRow(BoxLayout):
    effectwidget = ObjectProperty()

    def update_effectwidget(self, *args):
        if False:
            return 10
        effects = []
        for child in self.children[::-1]:
            text = child.text
            if text == 'none':
                pass
            if text == 'fxaa':
                effects.append(FXAAEffect())
            if text == 'monochrome':
                effects.append(MonochromeEffect())
            if text == 'invert':
                effects.append(InvertEffect())
            if text == 'mix':
                effects.append(ChannelMixEffect())
            if text == 'blur_h':
                effects.append(HorizontalBlurEffect())
            if text == 'blur_v':
                effects.append(VerticalBlurEffect())
            if text == 'postprocessing':
                effects.append(ScanlinesEffect())
            if text == 'pixelate':
                effects.append(PixelateEffect())
        if self.effectwidget:
            self.effectwidget.effects = effects
example = Builder.load_string("\n#:import Vector kivy.vector.Vector\nBoxLayout:\n    orientation: 'vertical'\n    FloatLayout:\n        ComparisonWidget:\n            pos_hint: {'x': 0, 'y': 0}\n            size_hint: 0.5, 1\n            id: effect1\n        ComparisonWidget:\n            pos_hint: {'x': pos_slider.value, 'y': 0}\n            size_hint: 0.5, 1\n            id: effect2\n            background_color: (rs.value, gs.value, bs.value, als.value)\n    SpinnerRow:\n        effectwidget: effect1\n        text: 'left effects'\n    SpinnerRow:\n        effectwidget: effect2\n        text: 'right effects'\n    BoxLayout:\n        size_hint_y: None\n        height: sp(40)\n        Label:\n            text: 'control overlap:'\n        Slider:\n            min: 0\n            max: 0.5\n            value: 0.5\n            id: pos_slider\n    BoxLayout:\n        size_hint_y: None\n        height: sp(40)\n        Label:\n            text: 'right bg r,g,b,a'\n        Slider:\n            min: 0\n            max: 1\n            value: 0\n            id: rs\n        Slider:\n            min: 0\n            max: 1\n            value: 0\n            id: gs\n        Slider:\n            min: 0\n            max: 1\n            value: 0\n            id: bs\n        Slider:\n            min: 0\n            max: 1\n            value: 0\n            id: als\n\n\n<ComparisonWidget>:\n    Widget:\n        canvas:\n            Color:\n                rgba: 1, 0, 0, 1\n            Ellipse:\n                pos: Vector(self.pos) + 0.5*Vector(self.size)\n                size: 0.4*Vector(self.size)\n            Color:\n                rgba: 0, 1, 0.3, 1\n            Ellipse:\n                pos: Vector(self.pos) + 0.1*Vector(self.size)\n                size: 0.6*Vector(self.size)\n            Color:\n                rgba: 0.5, 0.3, 0.8, 1\n            Ellipse:\n                pos: Vector(self.pos) + Vector([0, 0.6])*Vector(self.size)\n                size: 0.4*Vector(self.size)\n            Color:\n                rgba: 1, 0.8, 0.1, 1\n            Ellipse:\n                pos: Vector(self.pos) + Vector([0.5, 0])*Vector(self.size)\n                size: 0.4*Vector(self.size)\n            Color:\n                rgba: 0, 0, 0.8, 1\n            Line:\n                points:\n                    [self.x, self.y,\n                    self.x + self.width, self.y + 0.3*self.height,\n                    self.x + 0.2*self.width, self.y + 0.1*self.height,\n                    self.x + 0.85*self.width, self.y + 0.72*self.height,\n                    self.x + 0.31*self.width, self.y + 0.6*self.height,\n                    self.x, self.top]\n                width: 1\n            Color:\n                rgba: 0, 0.9, 0.1, 1\n            Line:\n                points:\n                    [self.x + self.width, self.y + self.height,\n                    self.x + 0.35*self.width, self.y + 0.6*self.height,\n                    self.x + 0.7*self.width, self.y + 0.15*self.height,\n                    self.x + 0.2*self.width, self.y + 0.22*self.height,\n                    self.x + 0.3*self.width, self.y + 0.92*self.height]\n                width: 2\n\n<SpinnerRow>:\n    orientation: 'horizontal'\n    size_hint_y: None\n    height: dp(40)\n    text: ''\n    Label:\n        text: root.text\n    EffectSpinner:\n        on_text: root.update_effectwidget()\n    EffectSpinner:\n        on_text: root.update_effectwidget()\n    EffectSpinner:\n        on_text: root.update_effectwidget()\n\n<EffectSpinner>:\n    text: 'none'\n    values:\n        ['none', 'fxaa', 'monochrome',\n        'invert', 'mix',\n        'blur_h', 'blur_v',\n        'postprocessing', 'pixelate',]\n")

class EffectApp(App):

    def build(self):
        if False:
            i = 10
            return i + 15
        return example
EffectApp().run()