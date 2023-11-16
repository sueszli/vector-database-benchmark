"""
Circle Example
==============

This example exercises circle (ellipse) drawing. You should see sliders at the
top of the screen with the Kivy logo below it. The sliders control the
angle start and stop and the height and width scales. There is a button
to reset the sliders. The logo used for the circle's background image is
from the kivy/data directory. The entire example is coded in the
kv language description.
"""
from kivy.app import App
from kivy.lang import Builder
kv = "\nBoxLayout:\n    orientation: 'vertical'\n    BoxLayout:\n        size_hint_y: None\n        height: sp(100)\n        BoxLayout:\n            orientation: 'vertical'\n            Slider:\n                id: e1\n                min: -360.\n                max: 360.\n            Label:\n                text: 'angle_start = {}'.format(e1.value)\n        BoxLayout:\n            orientation: 'vertical'\n            Slider:\n                id: e2\n                min: -360.\n                max: 360.\n                value: 360\n            Label:\n                text: 'angle_end = {}'.format(e2.value)\n\n    BoxLayout:\n        size_hint_y: None\n        height: sp(100)\n        BoxLayout:\n            orientation: 'vertical'\n            Slider:\n                id: wm\n                min: 0\n                max: 2\n                value: 1\n            Label:\n                text: 'Width mult. = {}'.format(wm.value)\n        BoxLayout:\n            orientation: 'vertical'\n            Slider:\n                id: hm\n                min: 0\n                max: 2\n                value: 1\n            Label:\n                text: 'Height mult. = {}'.format(hm.value)\n        Button:\n            text: 'Reset ratios'\n            on_press: wm.value = 1; hm.value = 1\n\n    FloatLayout:\n        canvas:\n            Color:\n                rgb: 1, 1, 1\n            Ellipse:\n                pos: 100, 100\n                size: 200 * wm.value, 201 * hm.value\n                source: 'data/logo/kivy-icon-512.png'\n                angle_start: e1.value\n                angle_end: e2.value\n\n"

class CircleApp(App):

    def build(self):
        if False:
            return 10
        return Builder.load_string(kv)
CircleApp().run()