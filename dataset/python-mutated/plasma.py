"""
Plasma Shader
=============

This shader example have been taken from
http://www.iquilezles.org/apps/shadertoy/ with some adaptation.

This might become a Kivy widget when experimentation will be done.
"""
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.graphics import RenderContext
from kivy.properties import StringProperty
plasma_shader = '\n$HEADER$\n\nuniform vec2 resolution;\nuniform float time;\n\nvoid main(void)\n{\n   vec4 frag_coord = frag_modelview_mat * gl_FragCoord;\n   float x = frag_coord.x;\n   float y = frag_coord.y;\n   float mov0 = x+y+cos(sin(time)*2.)*100.+sin(x/100.)*1000.;\n   float mov1 = y / resolution.y / 0.2 + time;\n   float mov2 = x / resolution.x / 0.2;\n   float c1 = abs(sin(mov1+time)/2.+mov2/2.-mov1-mov2+time);\n   float c2 = abs(sin(c1+sin(mov0/1000.+time)\n              +sin(y/40.+time)+sin((x+y)/100.)*3.));\n   float c3 = abs(sin(c2+cos(mov1+mov2+c2)+cos(mov2)+sin(x/1000.)));\n   gl_FragColor = vec4( c1,c2,c3,1.0);\n}\n'

class ShaderWidget(FloatLayout):
    fs = StringProperty(None)

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.canvas = RenderContext()
        super(ShaderWidget, self).__init__(**kwargs)
        Clock.schedule_interval(self.update_glsl, 1 / 60.0)

    def on_fs(self, instance, value):
        if False:
            i = 10
            return i + 15
        shader = self.canvas.shader
        old_value = shader.fs
        shader.fs = value
        if not shader.success:
            shader.fs = old_value
            raise Exception('failed')

    def update_glsl(self, *largs):
        if False:
            i = 10
            return i + 15
        self.canvas['time'] = Clock.get_boottime()
        self.canvas['resolution'] = list(map(float, self.size))
        win_rc = Window.render_context
        self.canvas['projection_mat'] = win_rc['projection_mat']
        self.canvas['modelview_mat'] = win_rc['modelview_mat']
        self.canvas['frag_modelview_mat'] = win_rc['frag_modelview_mat']

class PlasmaApp(App):

    def build(self):
        if False:
            return 10
        return ShaderWidget(fs=plasma_shader)
if __name__ == '__main__':
    PlasmaApp().run()