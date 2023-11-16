"""
Show a rotating cube with lighting
==================================
"""
import numpy as np
from vispy import gloo, app
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate
from vispy.geometry import create_cube
vertex = '\nuniform mat4 u_model;\nuniform mat4 u_view;\nuniform mat4 u_projection;\nuniform vec4 u_color;\n\nattribute vec3 position;\nattribute vec2 texcoord;\nattribute vec3 normal;\nattribute vec4 color;\n\nvarying vec3 v_position;\nvarying vec3 v_normal;\nvarying vec4 v_color;\n\nvoid main()\n{\n    v_normal = normal;\n    v_position = position;\n    v_color = color * u_color;\n    gl_Position = u_projection * u_view * u_model * vec4(position,1.0);\n}\n'
fragment = '\nuniform mat4 u_model;\nuniform mat4 u_view;\nuniform mat4 u_normal;\n\nuniform vec3 u_light_intensity;\nuniform vec3 u_light_position;\n\nvarying vec3 v_position;\nvarying vec3 v_normal;\nvarying vec4 v_color;\n\nvoid main()\n{\n    // Calculate normal in world coordinates\n    vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;\n\n    // Calculate the location of this fragment (pixel) in world coordinates\n    vec3 position = vec3(u_view*u_model * vec4(v_position, 1));\n\n    // Calculate the vector from this pixels surface to the light source\n    vec3 surfaceToLight = u_light_position - position;\n\n    // Calculate the cosine of the angle of incidence (brightness)\n    float brightness = dot(normal, surfaceToLight) /\n                      (length(surfaceToLight) * length(normal));\n    brightness = max(min(brightness,1.0),0.0);\n\n    // Calculate final color of the pixel, based on:\n    // 1. The angle of incidence: brightness\n    // 2. The color/intensities of the light: light.intensities\n    // 3. The texture and texture coord: texture(tex, fragTexCoord)\n\n    gl_FragColor = v_color * brightness * vec4(u_light_intensity, 1);\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, size=(512, 512), title='Lighted cube', keys='interactive')
        self.timer = app.Timer('auto', self.on_timer)
        (V, F, outline) = create_cube()
        vertices = VertexBuffer(V)
        self.faces = IndexBuffer(F)
        self.outline = IndexBuffer(outline)
        self.view = translate((0, 0, -5))
        model = np.eye(4, dtype=np.float32)
        normal = np.array(np.matrix(np.dot(self.view, model)).I.T)
        self.program = Program(vertex, fragment)
        self.program.bind(vertices)
        self.program['u_light_position'] = (2, 2, 2)
        self.program['u_light_intensity'] = (1, 1, 1)
        self.program['u_model'] = model
        self.program['u_view'] = self.view
        self.program['u_normal'] = normal
        (self.phi, self.theta) = (0, 0)
        self.activate_zoom()
        gloo.set_state(clear_color=(0.3, 0.3, 0.35, 1.0), depth_test=True, polygon_offset=(1, 1), blend_func=('src_alpha', 'one_minus_src_alpha'), line_width=0.75)
        self.timer.start()
        self.show()

    def on_draw(self, event):
        if False:
            return 10
        gloo.clear(color=True, depth=True)
        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
        self.program['u_color'] = (1, 1, 1, 1)
        self.program.draw('triangles', self.faces)
        gloo.set_state(polygon_offset_fill=False, blend=True, depth_mask=False)
        self.program['u_color'] = (0, 0, 0, 1)
        self.program.draw('lines', self.outline)
        gloo.set_state(depth_mask=True)

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.activate_zoom()

    def activate_zoom(self):
        if False:
            i = 10
            return i + 15
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = perspective(45.0, self.size[0] / float(self.size[1]), 2.0, 10.0)
        self.program['u_projection'] = projection

    def on_timer(self, event):
        if False:
            while True:
                i = 10
        self.theta += 0.5
        self.phi += 0.5
        model = np.dot(rotate(self.theta, (0, 0, 1)), rotate(self.phi, (0, 1, 0)))
        normal = np.linalg.inv(np.dot(self.view, model)).T
        self.program['u_model'] = model
        self.program['u_normal'] = normal
        self.update()
if __name__ == '__main__':
    c = Canvas()
    app.run()