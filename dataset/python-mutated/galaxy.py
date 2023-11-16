import numpy as np
import sys
from vispy.util.transforms import perspective
from vispy.util import transforms
from vispy import gloo
from vispy import app
from vispy import io
import galaxy_specrend
from galaxy_simulation import Galaxy
VERT_SHADER = '\n#version 120\nuniform mat4  u_model;\nuniform mat4  u_view;\nuniform mat4  u_projection;\n\n//sampler that maps [0, n] -> color according to blackbody law\nuniform sampler1D u_colormap;\n//index to sample the colormap at\nattribute float a_color_index;\n\n//size of the star\nattribute float a_size;\n//type\n//type 0 - stars\n//type 1 - dust\n//type 2 - h2a objects\n//type 3 - h2a objects\nattribute float a_type;\nattribute vec2  a_position;\n//brightness of the star\nattribute float a_brightness;\n\nvarying vec3 v_color;\nvoid main (void)\n{\n    gl_Position = u_projection * u_view * u_model * vec4(a_position, 0.0,1.0);\n\n    //find base color according to physics from our sampler\n    vec3 base_color = texture1D(u_colormap, a_color_index).rgb;\n    //scale it down according to brightness\n    v_color = base_color * a_brightness;\n\n\n    if (a_size > 2.0)\n    {\n        gl_PointSize = a_size;\n    } else {\n        gl_PointSize = 0.0;\n    }\n\n    if (a_type == 2) {\n        v_color *= vec3(2, 1, 1);\n    }\n    else if (a_type == 3) {\n        v_color = vec3(.9);\n    }\n}\n'
FRAG_SHADER = '\n#version 120\n//star texture\nuniform sampler2D u_texture;\n//predicted color from black body\nvarying vec3 v_color;\n\nvoid main()\n{\n    //amount of intensity from the grayscale star\n    float star_tex_intensity = texture2D(u_texture, gl_PointCoord).r;\n    gl_FragColor = vec4(v_color * star_tex_intensity, 0.8);\n}\n'
galaxy = Galaxy(10000)
galaxy.reset(13000, 4000, 0.0004, 0.9, 0.9, 0.5, 200, 300)
(t0, t1) = (200.0, 10000.0)
n = 1000
dt = (t1 - t0) / n
colors = np.zeros(n, dtype=(np.float32, 3))
for i in range(n):
    temperature = t0 + i * dt
    (x, y, z) = galaxy_specrend.spectrum_to_xyz(galaxy_specrend.bb_spectrum, temperature)
    (r, g, b) = galaxy_specrend.xyz_to_rgb(galaxy_specrend.SMPTEsystem, x, y, z)
    r = min((max(r, 0), 1))
    g = min((max(g, 0), 1))
    b = min((max(b, 0), 1))
    colors[i] = galaxy_specrend.norm_rgb(r, g, b)

def load_galaxy_star_image():
    if False:
        for i in range(10):
            print('nop')
    fname = io.load_data_file('galaxy/star-particle.png')
    raw_image = io.read_png(fname)
    return raw_image

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER, count=len(galaxy))
        self.texture = gloo.Texture2D(load_galaxy_star_image(), interpolation='linear')
        self.program['u_texture'] = self.texture
        self.view = transforms.translate((0, 0, -5))
        self.program['u_view'] = self.view
        self.model = np.eye(4, dtype=np.float32)
        self.program['u_model'] = self.model
        self.program['u_colormap'] = colors
        (w, h) = self.size
        self.projection = perspective(45.0, w / float(h), 1.0, 1000.0)
        self.program['u_projection'] = self.projection
        galaxy.update(100000)
        data = self.__create_galaxy_vertex_data()
        self.data_vbo = gloo.VertexBuffer(data)
        self.program.bind(self.data_vbo)
        gloo.set_state(clear_color=(0.0, 0.0, 0.03, 1.0), depth_test=False, blend=True, blend_func=('src_alpha', 'one'))
        self._timer = app.Timer('auto', connect=self.update, start=True)

    def __create_galaxy_vertex_data(self):
        if False:
            return 10
        data = np.zeros(len(galaxy), dtype=[('a_size', np.float32), ('a_position', np.float32, 2), ('a_color_index', np.float32), ('a_brightness', np.float32), ('a_type', np.float32)])
        (pw, ph) = self.physical_size
        data['a_size'] = galaxy['size'] * max(pw / 800.0, ph / 800.0)
        data['a_position'] = galaxy['position'] / 13000.0
        data['a_color_index'] = (galaxy['temperature'] - t0) / (t1 - t0)
        data['a_brightness'] = galaxy['brightness']
        data['a_type'] = galaxy['type']
        return data

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.set_viewport(0, 0, *event.physical_size)
        (w, h) = event.size
        self.projection = perspective(45.0, w / float(h), 1.0, 1000.0)
        self.program['u_projection'] = self.projection

    def on_draw(self, event):
        if False:
            i = 10
            return i + 15
        galaxy.update(50000)
        data = self.__create_galaxy_vertex_data()
        self.data_vbo.set_data(data)
        self.program.bind(self.data_vbo)
        gloo.clear(color=True, depth=True)
        self.program.draw('points')
if __name__ == '__main__':
    c = Canvas()
    c.show()
    if sys.flags.interactive == 0:
        app.run()