"""
Tutorial: Creating Visuals
==========================

05. Camera location
-------------------

In this tutorial we will demonstrate how to determine the direction from which
a Visual is being viewed.
"""
from vispy import app, gloo, visuals, scene, io
vertex_shader = '\nvarying vec4 color;\nvoid main() {\n    vec4 visual_pos = vec4($position, 1);\n    vec4 doc_pos = $visual_to_doc(visual_pos);\n    gl_Position = $doc_to_render(doc_pos);\n    \n    vec4 visual_pos2 = $doc_to_visual(doc_pos + vec4(0, 0, -1, 0));\n    vec4 view_direction = (visual_pos2 / visual_pos2.w) - visual_pos;\n    view_direction = vec4(normalize(view_direction.xyz), 0);\n    \n    color = vec4(view_direction.rgb, 1);\n}\n'
fragment_shader = '\nvarying vec4 color;\nvoid main() {\n    gl_FragColor = color;\n}\n'

class MyMeshVisual(visuals.Visual):
    """
    """

    def __init__(self):
        if False:
            return 10
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        fname = io.load_data_file('orig/triceratops.obj.gz')
        (vertices, faces, normals, tex) = io.read_mesh(fname)
        self._ibo = gloo.IndexBuffer(faces)
        self.shared_program.vert['position'] = gloo.VertexBuffer(vertices)
        self.set_gl_state('additive', cull_face=False)
        self._draw_mode = 'triangles'
        self._index_buffer = self._ibo

    def _prepare_transforms(self, view):
        if False:
            print('Hello World!')
        tr = view.transforms
        view_vert = view.view_program.vert
        view_vert['visual_to_doc'] = tr.get_transform('visual', 'document')
        view_vert['doc_to_visual'] = tr.get_transform('document', 'visual')
        view_vert['doc_to_render'] = tr.get_transform('document', 'render')
MyMesh = scene.visuals.create_visual_node(MyMeshVisual)
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 50
view.camera.distance = 2
mesh = MyMesh(parent=view.scene)
mesh.transform = visuals.transforms.MatrixTransform()
mesh.transform.rotate(90, (1, 0, 0))
axis = scene.visuals.XYZAxis(parent=view.scene)
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        app.run()