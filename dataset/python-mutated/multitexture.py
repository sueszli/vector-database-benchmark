"""
Multitexture Example
====================

This example blends two textures: the image mtexture1.png of the letter K
and the image mtexture2.png of an orange circle. You should see an orange
K clipped to a circle. It uses a custom shader, written in glsl
(OpenGL Shading Language), stored in a local string.

Note the image mtexture1.png is a white 'K' on a transparent background, which
makes it hard to see.
"""
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.graphics import RenderContext, Color, Rectangle, BindTexture
fs_multitexture = '\n$HEADER$\n\n// New uniform that will receive texture at index 1\nuniform sampler2D texture1;\n\nvoid main(void) {\n\n    // multiple current color with both texture (0 and 1).\n    // currently, both will use exactly the same texture coordinates.\n    gl_FragColor = frag_color *         texture2D(texture0, tex_coord0) *         texture2D(texture1, tex_coord0);\n}\n'
kv = '\n<MultitextureLayout>:\n\n    Image:\n        source: "mtexture1.png"\n        size_hint: .3,.3\n        id: 1\n        pos: 0,200\n    Image:\n        source: "mtexture2.png"\n        size_hint: .3,.3\n        id: 2\n        pos: 200,200\n\n    MultitextureWidget:\n\n'
Builder.load_string(kv)

class MultitextureWidget(Widget):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.canvas = RenderContext()
        self.canvas.shader.fs = fs_multitexture
        with self.canvas:
            Color(1, 1, 1)
            BindTexture(source='mtexture2.png', index=1)
            Rectangle(size=(150, 150), source='mtexture1.png', pos=(500, 200))
        self.canvas['texture1'] = 1
        super(MultitextureWidget, self).__init__(**kwargs)
        Clock.schedule_interval(self.update_glsl, 0)

    def update_glsl(self, *largs):
        if False:
            print('Hello World!')
        self.canvas['projection_mat'] = Window.render_context['projection_mat']
        self.canvas['modelview_mat'] = Window.render_context['modelview_mat']

class MultitextureLayout(FloatLayout):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self.size = kwargs['size']
        super(MultitextureLayout, self).__init__(**kwargs)

class MultitextureApp(App):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        return MultitextureLayout(size=(600, 600))
if __name__ == '__main__':
    MultitextureApp().run()