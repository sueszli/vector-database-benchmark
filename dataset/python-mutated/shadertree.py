"""
Tree shader
===========

This example is an experimentation to show how we can use shader for a tree
subset. Here, we made a ShaderTreeWidget, different than the ShaderWidget
in the plasma.py example.

The ShaderTree widget create a Framebuffer, render his children on it, and
render the Framebuffer with a specific Shader.
With this way, you can apply cool effect on your widgets :)

"""
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.properties import StringProperty, ObjectProperty
from kivy.graphics import RenderContext, Fbo, Color, ClearColor, ClearBuffers, Rectangle
import itertools
header = '\n$HEADER$\n\nuniform vec2 resolution;\nuniform float time;\n'
shader_pulse = header + '\nvoid main(void)\n{\n    vec2 halfres = resolution.xy/2.0;\n    vec2 cPos = vec4(frag_modelview_mat * gl_FragCoord).xy;\n\n    cPos.x -= 0.5*halfres.x*sin(time/2.0)+0.3*halfres.x*cos(time)+halfres.x;\n    cPos.y -= 0.4*halfres.y*sin(time/5.0)+0.3*halfres.y*cos(time)+halfres.y;\n    float cLength = length(cPos);\n\n    vec2 uv = tex_coord0+(cPos/cLength)*sin(cLength/30.0-time*10.0)/25.0;\n    vec3 col = texture2D(texture0,uv).xyz*50.0/cLength;\n\n    gl_FragColor = vec4(col,1.0);\n}\n'
shader_postprocessing = header + '\nuniform vec2 uvsize;\nuniform vec2 uvpos;\nvoid main(void)\n{\n    vec2 q = tex_coord0 * vec2(1, -1);\n    vec2 uv = 0.5 + (q-0.5);//*(0.9);// + 0.1*sin(0.2*time));\n\n    vec3 oricol = texture2D(texture0,vec2(q.x,1.0-q.y)).xyz;\n    vec3 col;\n\n    col.r = texture2D(texture0,vec2(uv.x+0.003,-uv.y)).x;\n    col.g = texture2D(texture0,vec2(uv.x+0.000,-uv.y)).y;\n    col.b = texture2D(texture0,vec2(uv.x-0.003,-uv.y)).z;\n\n    col = clamp(col*0.5+0.5*col*col*1.2,0.0,1.0);\n\n    //col *= 0.5 + 0.5*16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y);\n\n    col *= vec3(0.8,1.0,0.7);\n\n    col *= 0.9+0.1*sin(10.0*time+uv.y*1000.0);\n\n    col *= 0.97+0.03*sin(110.0*time);\n\n    float comp = smoothstep( 0.2, 0.7, sin(time) );\n    //col = mix( col, oricol, clamp(-2.0+2.0*q.x+3.0*comp,0.0,1.0) );\n\n    gl_FragColor = vec4(col,1.0);\n}\n'
shader_monochrome = header + '\nvoid main() {\n    vec4 rgb = texture2D(texture0, tex_coord0);\n    float c = (rgb.x + rgb.y + rgb.z) * 0.3333;\n    gl_FragColor = vec4(c, c, c, 1.0);\n}\n'

class ShaderWidget(FloatLayout):
    fs = StringProperty(None)
    texture = ObjectProperty(None)

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.canvas = RenderContext(use_parent_projection=True, use_parent_modelview=True, use_parent_frag_modelview=True)
        with self.canvas:
            self.fbo = Fbo(size=self.size)
            self.fbo_color = Color(1, 1, 1, 1)
            self.fbo_rect = Rectangle()
        with self.fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()
        super(ShaderWidget, self).__init__(**kwargs)
        Clock.schedule_interval(self.update_glsl, 0)

    def update_glsl(self, *largs):
        if False:
            while True:
                i = 10
        self.canvas['time'] = Clock.get_boottime()
        self.canvas['resolution'] = [float(v) for v in self.size]

    def on_fs(self, instance, value):
        if False:
            for i in range(10):
                print('nop')
        shader = self.canvas.shader
        old_value = shader.fs
        shader.fs = value
        if not shader.success:
            shader.fs = old_value
            raise Exception('failed')

    def add_widget(self, *args, **kwargs):
        if False:
            return 10
        c = self.canvas
        self.canvas = self.fbo
        super(ShaderWidget, self).add_widget(*args, **kwargs)
        self.canvas = c

    def remove_widget(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        c = self.canvas
        self.canvas = self.fbo
        super(ShaderWidget, self).remove_widget(*args, **kwargs)
        self.canvas = c

    def on_size(self, instance, value):
        if False:
            while True:
                i = 10
        self.fbo.size = value
        self.texture = self.fbo.texture
        self.fbo_rect.size = value

    def on_pos(self, instance, value):
        if False:
            print('Hello World!')
        self.fbo_rect.pos = value

    def on_texture(self, instance, value):
        if False:
            print('Hello World!')
        self.fbo_rect.texture = value

class RootWidget(FloatLayout):
    shader_btn = ObjectProperty(None)
    shader_widget = ObjectProperty(None)

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(RootWidget, self).__init__(**kwargs)
        available_shaders = [shader_pulse, shader_postprocessing, shader_monochrome]
        self.shaders = itertools.cycle(available_shaders)
        self.shader_btn.bind(on_release=self.change)

    def change(self, *largs):
        if False:
            print('Hello World!')
        self.shader_widget.fs = next(self.shaders)

class ShaderTreeApp(App):

    def build(self):
        if False:
            print('Hello World!')
        return RootWidget()
if __name__ == '__main__':
    ShaderTreeApp().run()