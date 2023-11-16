from __future__ import division
import numpy as np
from vispy.gloo import Program, FrameBuffer, RenderBuffer, set_viewport, clear, set_state
from vispy import app
render_vertex = '\nattribute vec2 position;\nattribute vec2 texcoord;\nvarying vec2 v_texcoord;\nvoid main()\n{\n    gl_Position = vec4(position, 0.0, 1.0);\n    v_texcoord = texcoord;\n}\n'
render_fragment = '\nuniform int pingpong;\nuniform sampler2D texture;\nvarying vec2 v_texcoord;\nvoid main()\n{\n    float v;\n    if( pingpong == 0 )\n        v = texture2D(texture, v_texcoord).r;\n    else\n        v = texture2D(texture, v_texcoord).b;\n    gl_FragColor = vec4(1.0-v, 1.0-v, 1.0-v, 1.0);\n}\n'
compute_vertex = '\nattribute vec2 position;\nattribute vec2 texcoord;\nvarying vec2 v_texcoord;\nvoid main()\n{\n    gl_Position = vec4(position, 0.0, 1.0);\n    v_texcoord = texcoord;\n}\n'
compute_fragment = '\nuniform int pingpong;\nuniform sampler2D texture; // U,V:= r,g, other channels ignored\nuniform sampler2D params;  // rU,rV,f,k := r,g,b,a\nuniform float dx;          // horizontal distance between texels\nuniform float dy;          // vertical distance between texels\nuniform float dd;          // unit of distance\nuniform float dt;          // unit of time\nvarying vec2 v_texcoord;\nvoid main(void)\n{\n    float center = -(4.0+4.0/sqrt(2.0));  // -1 * other weights\n    float diag   = 1.0/sqrt(2.0);         // weight for diagonals\n    vec2 p = v_texcoord;                  // center coordinates\n\n    vec2 c,l;\n    if( pingpong == 0 ) {\n        c = texture2D(texture, p).rg;    // central value\n        // Compute Laplacian\n        l = ( texture2D(texture, p + vec2(-dx,-dy)).rg\n            + texture2D(texture, p + vec2( dx,-dy)).rg\n            + texture2D(texture, p + vec2(-dx, dy)).rg\n            + texture2D(texture, p + vec2( dx, dy)).rg) * diag\n            + texture2D(texture, p + vec2(-dx, 0.0)).rg\n            + texture2D(texture, p + vec2( dx, 0.0)).rg\n            + texture2D(texture, p + vec2(0.0,-dy)).rg\n            + texture2D(texture, p + vec2(0.0, dy)).rg\n            + c * center;\n    } else {\n        c = texture2D(texture, p).ba;    // central value\n        // Compute Laplacian\n        l = ( texture2D(texture, p + vec2(-dx,-dy)).ba\n            + texture2D(texture, p + vec2( dx,-dy)).ba\n            + texture2D(texture, p + vec2(-dx, dy)).ba\n            + texture2D(texture, p + vec2( dx, dy)).ba) * diag\n            + texture2D(texture, p + vec2(-dx, 0.0)).ba\n            + texture2D(texture, p + vec2( dx, 0.0)).ba\n            + texture2D(texture, p + vec2(0.0,-dy)).ba\n            + texture2D(texture, p + vec2(0.0, dy)).ba\n            + c * center;\n    }\n\n    float u = c.r;           // compute some temporary\n    float v = c.g;           // values which might save\n    float lu = l.r;          // a few GPU cycles\n    float lv = l.g;\n    float uvv = u * v * v;\n\n    vec4 q = texture2D(params, p).rgba;\n    float ru = q.r;          // rate of diffusion of U\n    float rv = q.g;          // rate of diffusion of V\n    float f  = q.b;          // some coupling parameter\n    float k  = q.a;          // another coupling parameter\n\n    float du = ru * lu / dd - uvv + f * (1.0 - u); // Gray-Scott equation\n    float dv = rv * lv / dd + uvv - (f + k) * v;   // diffusion+-reaction\n\n    u += du * dt;\n    v += dv * dt;\n\n    if( pingpong == 1 ) {\n        gl_FragColor = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), c);\n    } else {\n        gl_FragColor = vec4(c, clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));\n    }\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            return 10
        app.Canvas.__init__(self, title='Grayscott Reaction-Diffusion', size=(512, 512), keys='interactive')
        self.scale = 4
        self.comp_size = self.size
        (comp_w, comp_h) = self.comp_size
        dt = 1.0
        dd = 1.5
        species = {'Bacteria 1': [0.16, 0.08, 0.035, 0.065], 'Bacteria 2': [0.14, 0.06, 0.035, 0.065], 'Coral': [0.16, 0.08, 0.06, 0.062], 'Fingerprint': [0.19, 0.05, 0.06, 0.062], 'Spirals': [0.1, 0.1, 0.018, 0.05], 'Spirals Dense': [0.12, 0.08, 0.02, 0.05], 'Spirals Fast': [0.1, 0.16, 0.02, 0.05], 'Unstable': [0.16, 0.08, 0.02, 0.055], 'Worms 1': [0.16, 0.08, 0.05, 0.065], 'Worms 2': [0.16, 0.08, 0.054, 0.063], 'Zebrafish': [0.16, 0.08, 0.035, 0.06]}
        P = np.zeros((comp_h, comp_w, 4), dtype=np.float32)
        P[:, :] = species['Unstable']
        UV = np.zeros((comp_h, comp_w, 4), dtype=np.float32)
        UV[:, :, 0] = 1.0
        r = 32
        UV[comp_h // 2 - r:comp_h // 2 + r, comp_w // 2 - r:comp_w // 2 + r, 0] = 0.5
        UV[comp_h // 2 - r:comp_h // 2 + r, comp_w // 2 - r:comp_w // 2 + r, 1] = 0.25
        UV += np.random.uniform(0.0, 0.01, (comp_h, comp_w, 4))
        UV[:, :, 2] = UV[:, :, 0]
        UV[:, :, 3] = UV[:, :, 1]
        self.pingpong = 1
        self.compute = Program(compute_vertex, compute_fragment, 4)
        self.compute['params'] = P
        self.compute['texture'] = UV
        self.compute['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.compute['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.compute['dt'] = dt
        self.compute['dx'] = 1.0 / comp_w
        self.compute['dy'] = 1.0 / comp_h
        self.compute['dd'] = dd
        self.compute['pingpong'] = self.pingpong
        self.render = Program(render_vertex, render_fragment, 4)
        self.render['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.render['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.render['texture'] = self.compute['texture']
        self.render['pingpong'] = self.pingpong
        self.fbo = FrameBuffer(self.compute['texture'], RenderBuffer(self.comp_size))
        set_state(depth_test=False, clear_color='black')
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()

    def on_draw(self, event):
        if False:
            i = 10
            return i + 15
        with self.fbo:
            set_viewport(0, 0, *self.comp_size)
            self.compute['texture'].interpolation = 'nearest'
            self.compute.draw('triangle_strip')
        clear(color=True)
        set_viewport(0, 0, *self.physical_size)
        self.render['texture'].interpolation = 'linear'
        self.render.draw('triangle_strip')
        self.pingpong = 1 - self.pingpong
        self.compute['pingpong'] = self.pingpong
        self.render['pingpong'] = self.pingpong

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        set_viewport(0, 0, *self.physical_size)
if __name__ == '__main__':
    canvas = Canvas()
    app.run()