"""
Rotated Shader
=============

This shader example is a modified version of plasma.py that shows how to
rotate areas of fragment shaders bounded by vertex_instructions.
"""
from kivy.app import App
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import StringProperty
from kivy.uix.widget import Widget
import kivy.core.window
shared_code = '\n$HEADER$\n\nuniform float time;\n\nvec4 tex(void)\n{\n   return frag_color * texture2D(texture0, tex_coord0);\n}\n\nfloat plasmaFunc(float n1, float n2, float n3, float n4)\n{\n   vec4 fPos = frag_modelview_mat * gl_FragCoord;\n   return abs(sin(\n                  sin(sin(fPos.x / n1) + time) +\n                  sin(fPos.y / n2 + time) +\n                  n4 * sin((fPos.x + fPos.y) / n3)));\n}\n\n'
plasma_shader = shared_code + '\nvoid main(void)\n{\n   float green = plasmaFunc(40., 30., 100., 3.5);\n   gl_FragColor = vec4(1.0, green, 1.0, 1.0) * tex();\n}\n\n'
plasma_shader2 = shared_code + '\nvoid main(void)\n{\n   float red = plasmaFunc(30., 20., 10., .5);\n   gl_FragColor = vec4(red, 1.0, 1.0, 1.0) * tex();\n}\n\n'

class ShaderWidget(Widget):
    fs = StringProperty(None)

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.canvas = RenderContext(use_parent_projection=True, use_parent_modelview=True, use_parent_frag_modelview=True)
        super(ShaderWidget, self).__init__(**kwargs)
        Clock.schedule_interval(self.update_glsl, 1 / 60.0)

    def update_glsl(self, *largs):
        if False:
            i = 10
            return i + 15
        self.canvas['time'] = Clock.get_boottime()

    def on_fs(self, instance, value):
        if False:
            print('Hello World!')
        shader = self.canvas.shader
        old_value = shader.fs
        shader.fs = value
        if not shader.success:
            shader.fs = old_value
            raise Exception('failed')

class RotatedApp(App):

    def build(self):
        if False:
            print('Hello World!')
        main_widget = Factory.MainWidget()
        main_widget.fs = plasma_shader
        main_widget.mini.fs = plasma_shader2
        return main_widget
if __name__ == '__main__':
    RotatedApp().run()