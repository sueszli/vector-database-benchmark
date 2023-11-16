"""
This example demonstrates creating and using an AdvancedEffectBase. In
this case, we use it to efficiently pass the touch coordinates into the shader.
"""
from kivy.base import runTouchApp
from kivy.properties import ListProperty
from kivy.lang import Builder
from kivy.uix.effectwidget import EffectWidget, AdvancedEffectBase
effect_string = '\nuniform vec2 touch;\n\nvec4 effect(vec4 color, sampler2D texture, vec2 tex_coords, vec2 coords)\n{\n    vec2 distance = 0.025*(coords - touch);\n    float dist_mag = (distance.x*distance.x + distance.y*distance.y);\n    vec3 multiplier = vec3(abs(sin(dist_mag - time)));\n    return vec4(multiplier * color.xyz, 1.0);\n}\n'

class TouchEffect(AdvancedEffectBase):
    touch = ListProperty([0.0, 0.0])

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(TouchEffect, self).__init__(*args, **kwargs)
        self.glsl = effect_string
        self.uniforms = {'touch': [0.0, 0.0]}

    def on_touch(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.uniforms['touch'] = [float(i) for i in self.touch]

class TouchWidget(EffectWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(TouchWidget, self).__init__(*args, **kwargs)
        self.effect = TouchEffect()
        self.effects = [self.effect]

    def on_touch_down(self, touch):
        if False:
            for i in range(10):
                print('nop')
        super(TouchWidget, self).on_touch_down(touch)
        self.on_touch_move(touch)

    def on_touch_move(self, touch):
        if False:
            return 10
        self.effect.touch = touch.pos
root = Builder.load_string('\nTouchWidget:\n    Button:\n        text: \'Some text!\'\n    Image:\n        source: \'data/logo/kivy-icon-512.png\'\n        fit_mode: "fill"\n')
runTouchApp(root)