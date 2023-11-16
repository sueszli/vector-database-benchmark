"""
Animate a 2D function
=====================

Illustrate how to plot a 2D function (an image) y=f(x,y) on the GPU.

"""
from vispy import app, gloo
vertex = '\nattribute vec2 a_position;\nvarying vec2 v_position;\nvoid main()\n{\n    gl_Position = vec4(a_position, 0.0, 1.0);\n    v_position = a_position;\n}\n'
fragment = '\n#include "math/constants.glsl"\n//const float M_PI = 3.14159265358979323846;\nuniform float u_time;\nvarying vec2 v_position;\n\n/**********************************************************\nSpecify the parameters here.\n**********************************************************/\nconst float z_offset = 1.;  // (z+z_offset)/z_max should be in [0,1]\nconst float z_max = 2.;\nconst float x_scale = 5.;  // x is between -x_scale and +x_scale\nconst float y_scale = 5.; // y is between -y_scale and +y_scale\nconst float t_scale = 5.; // scale for the time\n/*********************************************************/\n\nfloat f(float x, float y, float t) {\n\n    // x is in [-x_scale, +x_scale]\n    // y is in [-y_scale, +y_scale]\n    // t is in [0, +oo)\n\n    /**********************************************************\n    Write your function below.\n    **********************************************************/\n\n    float k = .25*cos(t);\n    return (cos(x)+k)*(sin(y)-k);\n\n    /*********************************************************/\n\n}\n\nvec4 jet(float x) {\n    vec3 a, b;\n    float c;\n    if (x < 0.34) {\n        a = vec3(0, 0, 0.5);\n        b = vec3(0, 0.8, 0.95);\n        c = (x - 0.0) / (0.34 - 0.0);\n    } else if (x < 0.64) {\n        a = vec3(0, 0.8, 0.95);\n        b = vec3(0.85, 1, 0.04);\n        c = (x - 0.34) / (0.64 - 0.34);\n    } else if (x < 0.89) {\n        a = vec3(0.85, 1, 0.04);\n        b = vec3(0.96, 0.7, 0);\n        c = (x - 0.64) / (0.89 - 0.64);\n    } else {\n        a = vec3(0.96, 0.7, 0);\n        b = vec3(0.5, 0, 0);\n        c = (x - 0.89) / (1.0 - 0.89);\n    }\n    return vec4(mix(a, b, c), 1.0);\n}\n\nvoid main() {\n    vec2 pos = v_position;\n    float z = f(x_scale * pos.x, y_scale * pos.y, t_scale * u_time);\n    gl_FragColor = jet((z + z_offset) / (z_max));\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            return 10
        app.Canvas.__init__(self, position=(300, 100), size=(800, 800), keys='interactive')
        self.program = gloo.Program(vertex, fragment)
        self.program['a_position'] = [(-1.0, -1.0), (-1.0, +1.0), (+1.0, -1.0), (+1.0, +1.0)]
        self.program['u_time'] = 0.0
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def on_timer(self, event):
        if False:
            while True:
                i = 10
        self.program['u_time'] = event.elapsed
        self.update()

    def on_resize(self, event):
        if False:
            print('Hello World!')
        (width, height) = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.program.draw('triangle_strip')
if __name__ == '__main__':
    canvas = Canvas()
    app.run()