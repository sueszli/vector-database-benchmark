"""
This is an example of creating your own effect by writing a glsl string.
"""
from kivy.base import runTouchApp
from kivy.lang import Builder
from kivy.uix.effectwidget import EffectWidget, EffectBase
effect_string = '\nvec4 effect(vec4 color, sampler2D texture, vec2 tex_coords, vec2 coords)\n{\n    // Note that time is a uniform variable that is automatically\n    // provided to all effects.\n    float red = color.x * abs(sin(time*2.0));\n    float green = color.y;  // No change\n    float blue = color.z * (1.0 - abs(sin(time*2.0)));\n    return vec4(red, green, blue, color.w);\n}\n'

class DemoEffect(EffectWidget):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.effect_reference = EffectBase(glsl=effect_string)
        super(DemoEffect, self).__init__(*args, **kwargs)
widget = Builder.load_string("\nDemoEffect:\n    effects: [self.effect_reference] if checkbox.active else []\n    orientation: 'vertical'\n    Button:\n        text: 'Some text so you can see what happens.'\n    BoxLayout:\n        size_hint_y: None\n        height: dp(50)\n        Label:\n            text: 'Enable effect?'\n        CheckBox:\n            id: checkbox\n            active: True\n")
runTouchApp(widget)