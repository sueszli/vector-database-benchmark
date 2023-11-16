"""
Demonstrate how to do offscreen rendering.
Possible use cases:

  * GPGPU without CUDA or OpenCL
  * creation of scripted animations
  * remote and Web backends

The method consists of:

  1. Not showing the canvas (show=False).
  2. Rendering to an FBO.
  3. Manually triggering a rendering pass with self.update().
  4. Retrieving the scene with _screenshot().
  5. Closing the app after the first rendering pass (if that's the intended
     scenario).

"""
from vispy import gloo
from vispy import app
from vispy.util.ptime import time
from vispy.gloo.util import _screenshot
app.use_app('glfw')
vertex = '\nattribute vec2 position;\n\nvoid main()\n{\n    gl_Position = vec4(position, 0, 1.0);\n}\n'
fragment = '\nuniform vec2 resolution;\nuniform vec2 center;\nuniform float scale;\nuniform int iter;\n\n// Jet color scheme\nvec4 color_scheme(float x) {\n    vec3 a, b;\n    float c;\n    if (x < 0.34) {\n        a = vec3(0, 0, 0.5);\n        b = vec3(0, 0.8, 0.95);\n        c = (x - 0.0) / (0.34 - 0.0);\n    } else if (x < 0.64) {\n        a = vec3(0, 0.8, 0.95);\n        b = vec3(0.85, 1, 0.04);\n        c = (x - 0.34) / (0.64 - 0.34);\n    } else if (x < 0.89) {\n        a = vec3(0.85, 1, 0.04);\n        b = vec3(0.96, 0.7, 0);\n        c = (x - 0.64) / (0.89 - 0.64);\n    } else {\n        a = vec3(0.96, 0.7, 0);\n        b = vec3(0.5, 0, 0);\n        c = (x - 0.89) / (1.0 - 0.89);\n    }\n    return vec4(mix(a, b, c), 1.0);\n}\n\nvoid main() {\n    vec2 z, c;\n\n    // Recover coordinates from pixel coordinates\n    c.x = (gl_FragCoord.x / resolution.x - 0.5) * scale + center.x;\n    c.y = (gl_FragCoord.y / resolution.y - 0.5) * scale + center.y;\n\n    // Main Mandelbrot computation\n    int i;\n    z = c;\n    for(i = 0; i < iter; i++) {\n        float x = (z.x * z.x - z.y * z.y) + c.x;\n        float y = (z.y * z.x + z.x * z.y) + c.y;\n\n        if((x * x + y * y) > 4.0) break;\n        z.x = x;\n        z.y = y;\n    }\n\n    // Convert iterations to color\n    float color = 1.0 - float(i) / float(iter);\n    gl_FragColor = color_scheme(color);\n\n}\n'

class Canvas(app.Canvas):

    def __init__(self, size=(600, 600)):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, show=False, size=size)
        self._t0 = time()
        self._rendertex = gloo.Texture2D(shape=self.size[::-1] + (4,))
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(self.size[::-1]))
        self.program = gloo.Program(vertex, fragment)
        self.program['position'] = [(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)]
        self.program['scale'] = 3
        self.program['center'] = [-0.5, 0]
        self.program['iter'] = 300
        self.program['resolution'] = self.size
        self.update()

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        with self._fbo:
            gloo.clear('black')
            gloo.set_viewport(0, 0, *self.size)
            self.program.draw()
            self.im = _screenshot((0, 0, self.size[0], self.size[1]))
        self._time = time() - self._t0
        app.quit()
if __name__ == '__main__':
    c = Canvas()
    size = c.size
    app.run()
    render = c.im
    print('Finished in %.1fms.' % (c._time * 1000.0))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100)
    plt.imshow(render, interpolation='none')
    plt.show()