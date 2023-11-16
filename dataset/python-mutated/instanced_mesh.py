"""
Instanced rendering of arbitrarily transformed meshes
=====================================================
"""
from vispy import app, gloo, visuals, scene, use
import numpy as np
from scipy.spatial.transform import Rotation
from vispy.io import read_mesh, load_data_file
use(gl='gl+')
vertex_shader = '\n// these attributes will be defined on an instance basis\nattribute vec3 shift;\nattribute vec4 color;\nattribute vec3 transform_x;\nattribute vec3 transform_y;\nattribute vec3 transform_z;\n\nvarying vec4 v_color;\n\nvoid main() {\n    v_color = color;\n    // transform is generated from column vectors (new basis vectors)\n    // https://en.wikibooks.org/wiki/GLSL_Programming/Vector_and_Matrix_Operations#Constructors\n    mat3 instance_transform = mat3(transform_x, transform_y, transform_z);\n    vec3 pos_rotated = instance_transform * $position;\n    vec4 pos_shifted = vec4(pos_rotated + shift, 1);\n    gl_Position = $transform(pos_shifted);\n}\n'
fragment_shader = '\nvarying vec4 v_color;\n\nvoid main() {\n  gl_FragColor = v_color;\n}\n'

class InstancedMeshVisual(visuals.Visual):

    def __init__(self, vertices, faces, positions, colors, transforms, subdivisions=5):
        if False:
            while True:
                i = 10
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        self.set_gl_state('translucent', depth_test=True, cull_face=True)
        self._draw_mode = 'triangles'
        self.vbo = gloo.VertexBuffer(vertices.astype(np.float32))
        self.shared_program.vert['position'] = self.vbo
        self._index_buffer = gloo.IndexBuffer(data=faces.astype(np.uint32))
        self.shifts = gloo.VertexBuffer(positions.astype(np.float32), divisor=1)
        self.shared_program['shift'] = self.shifts
        transforms = transforms.astype(np.float32)
        self.transforms_x = gloo.VertexBuffer(transforms[..., 0].copy(), divisor=1)
        self.transforms_y = gloo.VertexBuffer(transforms[..., 1].copy(), divisor=1)
        self.transforms_z = gloo.VertexBuffer(transforms[..., 2].copy(), divisor=1)
        self.shared_program['transform_x'] = self.transforms_x
        self.shared_program['transform_y'] = self.transforms_y
        self.shared_program['transform_z'] = self.transforms_z
        self.color = gloo.VertexBuffer(colors.astype(np.float32), divisor=1)
        self.shared_program['color'] = self.color

    def _prepare_transforms(self, view):
        if False:
            for i in range(10):
                print('nop')
        view.view_program.vert['transform'] = view.get_transform()
InstancedMesh = scene.visuals.create_visual_node(InstancedMeshVisual)
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'arcball'
view.camera.scale_factor = 1000
N = 1000
mesh_file = load_data_file('orig/triceratops.obj.gz')
(vertices, faces, _, _) = read_mesh(mesh_file)
np.random.seed(0)
pos = (np.random.rand(N, 3) - 0.5) * 1000
colors = np.random.rand(N, 4)
transforms = Rotation.random(N).as_matrix()
multimesh = InstancedMesh(vertices * 10, faces, pos, colors, transforms, parent=view.scene)
multimesh.transform = visuals.transforms.STTransform(scale=(3, 2, 1))
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        app.run()