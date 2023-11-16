"""
3D brain mesh viewer.
"""
from timeit import default_timer
import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate
from vispy.io import load_data_file
brain = np.load(load_data_file('brain/brain.npz', force_download='2014-09-04'))
data = brain['vertex_buffer']
faces = brain['index_buffer']
VERT_SHADER = '\n#version 120\nuniform mat4 u_model;\nuniform mat4 u_view;\nuniform mat4 u_projection;\nuniform vec4 u_color;\n\nattribute vec3 a_position;\nattribute vec3 a_normal;\nattribute vec4 a_color;\n\nvarying vec3 v_position;\nvarying vec3 v_normal;\nvarying vec4 v_color;\n\nvoid main()\n{\n    v_normal = a_normal;\n    v_position = a_position;\n    v_color = a_color * u_color;\n    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);\n}\n'
FRAG_SHADER = '\n#version 120\nuniform mat4 u_model;\nuniform mat4 u_view;\nuniform mat4 u_normal;\n\nuniform vec3 u_light_intensity;\nuniform vec3 u_light_position;\n\nvarying vec3 v_position;\nvarying vec3 v_normal;\nvarying vec4 v_color;\n\nvoid main()\n{\n    // Calculate normal in world coordinates\n    vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;\n\n    // Calculate the location of this fragment (pixel) in world coordinates\n    vec3 position = vec3(u_view*u_model * vec4(v_position, 1));\n\n    // Calculate the vector from this pixels surface to the light source\n    vec3 surfaceToLight = u_light_position - position;\n\n    // Calculate the cosine of the angle of incidence (brightness)\n    float brightness = dot(normal, surfaceToLight) /\n                      (length(surfaceToLight) * length(normal));\n    brightness = max(min(brightness,1.0),0.0);\n\n    // Calculate final color of the pixel, based on:\n    // 1. The angle of incidence: brightness\n    // 2. The color/intensities of the light: light.intensities\n    // 3. The texture and texture coord: texture(tex, fragTexCoord)\n\n    // Specular lighting.\n    vec3 surfaceToCamera = vec3(0.0, 0.0, 1.0) - position;\n    vec3 K = normalize(normalize(surfaceToLight) + normalize(surfaceToCamera));\n    float specular = clamp(pow(abs(dot(normal, K)), 40.), 0.0, 1.0);\n\n    gl_FragColor = v_color * brightness * vec4(u_light_intensity, 1);\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            return 10
        app.Canvas.__init__(self, keys='interactive')
        self.size = (800, 600)
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        (self.theta, self.phi) = (-80, 180)
        self.translate = 3
        self.faces = gloo.IndexBuffer(faces)
        self.program.bind(gloo.VertexBuffer(data))
        self.program['u_color'] = (1, 1, 1, 1)
        self.program['u_light_position'] = (1.0, 1.0, 1.0)
        self.program['u_light_intensity'] = (1.0, 1.0, 1.0)
        self.apply_zoom()
        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
        self._t0 = default_timer()
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.update_matrices()

    def update_matrices(self):
        if False:
            print('Hello World!')
        self.view = translate((0, 0, -self.translate))
        self.model = np.dot(rotate(self.theta, (1, 0, 0)), rotate(self.phi, (0, 1, 0)))
        self.projection = np.eye(4, dtype=np.float32)
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_normal'] = np.linalg.inv(np.dot(self.view, self.model)).T

    def on_timer(self, event):
        if False:
            i = 10
            return i + 15
        elapsed = default_timer() - self._t0
        self.phi = 180 + elapsed * 50.0
        self.update_matrices()
        self.update()

    def on_resize(self, event):
        if False:
            while True:
                i = 10
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        if False:
            while True:
                i = 10
        self.translate += -event.delta[1] / 5.0
        self.translate = max(2, self.translate)
        self.update_matrices()
        self.update()

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        gloo.clear()
        self.program.draw('triangles', indices=self.faces)

    def apply_zoom(self):
        if False:
            i = 10
            return i + 15
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] / float(self.size[1]), 1.0, 20.0)
        self.program['u_projection'] = self.projection
if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()