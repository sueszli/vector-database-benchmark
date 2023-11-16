import freenect
from time import sleep
from threading import Thread
from collections import deque
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import NumericProperty, StringProperty
from kivy.graphics import RenderContext, Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.boxlayout import BoxLayout
fragment_header = '\n#ifdef GL_ES\n    precision highp float;\n#endif\n\n/* Outputs from the vertex shader */\nvarying vec4 frag_color;\nvarying vec2 tex_coord0;\n\n/* uniform texture samplers */\nuniform sampler2D texture0;\n\n/* custom input */\nuniform float depth_range;\nuniform vec2 size;\n'
hsv_func = '\nvec3 HSVtoRGB(vec3 color) {\n    float f,p,q,t, hueRound;\n    int hueIndex;\n    float hue, saturation, v;\n    vec3 result;\n\n    /* just for clarity */\n    hue = color.r;\n    saturation = color.g;\n    v = color.b;\n\n    hueRound = floor(hue * 6.0);\n    hueIndex = mod(int(hueRound), 6.);\n    f = (hue * 6.0) - hueRound;\n    p = v * (1.0 - saturation);\n    q = v * (1.0 - f*saturation);\n    t = v * (1.0 - (1.0 - f)*saturation);\n\n    switch(hueIndex) {\n        case 0:\n            result = vec3(v,t,p);\n        break;\n        case 1:\n            result = vec3(q,v,p);\n        break;\n        case 2:\n            result = vec3(p,v,t);\n        break;\n        case 3:\n            result = vec3(p,q,v);\n        break;\n        case 4:\n            result = vec3(t,p,v);\n        break;\n        case 5:\n            result = vec3(v,p,q);\n        break;\n    }\n    return result;\n}\n'
rgb_kinect = fragment_header + '\nvoid main (void) {\n    float value = texture2D(texture0, tex_coord0).r;\n    value = mod(value * depth_range, 1.);\n    vec3 col = vec3(0., 0., 0.);\n    if ( value <= 0.33 )\n        col.r = clamp(value, 0., 0.33) * 3.;\n    if ( value <= 0.66 )\n        col.g = clamp(value - 0.33, 0., 0.33) * 3.;\n    col.b = clamp(value - 0.66, 0., 0.33) * 3.;\n    gl_FragColor = vec4(col, 1.);\n}\n'
points_kinect = fragment_header + hsv_func + '\nvoid main (void) {\n    // threshold used to reduce the depth (better result)\n    const int th = 5;\n\n    // size of a square\n    int square = floor(depth_range);\n\n    // number of square on the display\n    vec2 count = size / square;\n\n    // current position of the square\n    vec2 pos = floor(tex_coord0.xy * count) / count;\n\n    // texture step to pass to another square\n    vec2 step = 1 / count;\n\n    // texture step to pass to another pixel\n    vec2 pxstep = 1 / size;\n\n    // center of the square\n    vec2 center = pos + step / 2.;\n\n    // calculate average of every pixels in the square\n    float s = 0, x, y;\n    for (x = 0; x < square; x++) {\n        for (y = 0; y < square; y++) {\n            s += texture2D(texture0, pos + pxstep * vec2(x,y)).r;\n        }\n    }\n    float v = s / (square * square);\n\n    // threshold the value\n    float dr = th / 10.;\n    v = min(v, dr) / dr;\n\n    // calculate the distance between the center of the square and current\n    // pixel; display the pixel only if the distance is inside the circle\n    float vdist = length(abs(tex_coord0 - center) * size / square);\n    float value = 1 - v;\n    if ( vdist < value ) {\n        vec3 col = HSVtoRGB(vec3(value, 1., 1.));\n        gl_FragColor = vec4(col, 1);\n    }\n}\n'
hsv_kinect = fragment_header + hsv_func + '\nvoid main (void) {\n    float value = texture2D(texture0, tex_coord0).r;\n    value = mod(value * depth_range, 1.);\n    vec3 col = HSVtoRGB(vec3(value, 1., 1.));\n    gl_FragColor = vec4(col, 1.);\n}\n'

class KinectDepth(Thread):

    def __init__(self, *largs, **kwargs):
        if False:
            i = 10
            return i + 15
        super(KinectDepth, self).__init__(*largs, **kwargs)
        self.daemon = True
        self.queue = deque()
        self.quit = False
        self.index = 0

    def run(self):
        if False:
            while True:
                i = 10
        q = self.queue
        while not self.quit:
            depths = freenect.sync_get_depth(index=self.index)
            if depths is None:
                sleep(2)
                continue
            q.appendleft(depths)

    def pop(self):
        if False:
            print('Hello World!')
        return self.queue.pop()

class KinectViewer(Widget):
    depth_range = NumericProperty(7.7)
    shader = StringProperty('rgb')
    index = NumericProperty(0)

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.canvas = RenderContext()
        self.canvas.shader.fs = hsv_kinect
        self.kinect = KinectDepth()
        self.kinect.start()
        super(KinectViewer, self).__init__(**kwargs)
        self.texture = Texture.create(size=(640, 480), colorfmt='luminance', bufferfmt='ushort')
        self.texture.flip_vertical()
        with self.canvas:
            Color(1, 1, 1)
            Rectangle(size=Window.size, texture=self.texture)
        Clock.schedule_interval(self.update_transformation, 0)

    def on_index(self, instance, value):
        if False:
            for i in range(10):
                print('nop')
        self.kinect.index = value

    def on_shader(self, instance, value):
        if False:
            return 10
        if value == 'rgb':
            self.canvas.shader.fs = rgb_kinect
        elif value == 'hsv':
            self.canvas.shader.fs = hsv_kinect
        elif value == 'points':
            self.canvas.shader.fs = points_kinect

    def update_transformation(self, *largs):
        if False:
            print('Hello World!')
        self.canvas['projection_mat'] = Window.render_context['projection_mat']
        self.canvas['depth_range'] = self.depth_range
        self.canvas['size'] = list(map(float, self.size))
        try:
            value = self.kinect.pop()
        except:
            return
        f = value[0].astype('ushort') * 32
        self.texture.blit_buffer(f.tostring(), colorfmt='luminance', bufferfmt='ushort')
        self.canvas.ask_update()

class KinectViewerApp(App):

    def build(self):
        if False:
            i = 10
            return i + 15
        root = BoxLayout(orientation='vertical')
        self.viewer = viewer = KinectViewer(index=self.config.getint('kinect', 'index'), shader=self.config.get('shader', 'theme'))
        root.add_widget(viewer)
        toolbar = BoxLayout(size_hint=(1, None), height=50)
        root.add_widget(toolbar)
        slider = Slider(min=1.0, max=32.0, value=1.0)

        def update_depth_range(instance, value):
            if False:
                i = 10
                return i + 15
            viewer.depth_range = value
        slider.bind(value=update_depth_range)
        toolbar.add_widget(slider)
        return root

    def build_config(self, config):
        if False:
            return 10
        config.add_section('kinect')
        config.set('kinect', 'index', '0')
        config.add_section('shader')
        config.set('shader', 'theme', 'rgb')

    def build_settings(self, settings):
        if False:
            for i in range(10):
                print('nop')
        settings.add_json_panel('Kinect Viewer', self.config, data='[\n            { "type": "title", "title": "Kinect" },\n            { "type": "numeric", "title": "Index",\n              "desc": "Kinect index, from 0 to X",\n              "section": "kinect", "key": "index" },\n            { "type": "title", "title": "Shaders" },\n            { "type": "options", "title": "Theme",\n              "desc": "Shader to use for a specific visualization",\n              "section": "shader", "key": "theme",\n              "options": ["rgb", "hsv", "points"]}\n        ]')

    def on_config_change(self, config, section, key, value):
        if False:
            print('Hello World!')
        if config is not self.config:
            return
        token = (section, key)
        if token == ('kinect', 'index'):
            self.viewer.index = int(value)
        elif token == ('shader', 'theme'):
            if value == 'rgb':
                self.viewer.canvas.shader.fs = rgb_kinect
            elif value == 'hsv':
                self.viewer.shader = value
if __name__ == '__main__':
    KinectViewerApp().run()