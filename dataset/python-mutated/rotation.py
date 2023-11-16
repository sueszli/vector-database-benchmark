"""
Rotation Example
================

This example rotates a button using PushMatrix and PopMatrix. You should see
a static button with the words 'hello world' rotated at a 45 degree angle.
"""
from kivy.app import App
from kivy.lang import Builder
kv = "\nFloatLayout:\n\n    Button:\n        text: 'hello world'\n        size_hint: None, None\n        pos_hint: {'center_x': .5, 'center_y': .5}\n        canvas.before:\n            PushMatrix\n            Rotate:\n                angle: 45\n                origin: self.center\n        canvas.after:\n            PopMatrix\n"

class RotationApp(App):

    def build(self):
        if False:
            i = 10
            return i + 15
        return Builder.load_string(kv)
RotationApp().run()