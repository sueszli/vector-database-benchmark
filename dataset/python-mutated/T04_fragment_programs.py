"""
Tutorial: Creating Visuals
==========================

04. Fragment Programs
---------------------

In this tutorial, we will demonstrate the use of the fragment shader as a 
raycaster to draw complex shapes on a simple rectanglular mesh.

Previous tutorials focused on the use of forward transformation functions to 
map vertices from the local coordinate system of the visual to the "render 
coordinates" output of the vertex shader. In this tutorial, we will use inverse
transformation functions in the fragment shader to map backward from the 
current fragment location to the visual's local coordinate system. 
"""
import numpy as np
from vispy import app, gloo, visuals, scene
vertex_shader = '\nvoid main() {\n   gl_Position = vec4($position, 0, 1);\n}\n'
fragment_shader = '\nvoid main() {\n  vec4 pos = $fb_to_visual(gl_FragCoord);\n  gl_FragColor = vec4(sin(pos.x / 10.), sin(pos.y / 10.), 0, 1);\n}\n'

class MyRectVisual(visuals.Visual):
    """
    """

    def __init__(self):
        if False:
            print('Hello World!')
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        self.vbo = gloo.VertexBuffer(np.array([[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], dtype=np.float32))
        self.shared_program.vert['position'] = self.vbo
        self.set_gl_state(cull_face=False)
        self._draw_mode = 'triangle_fan'

    def _prepare_transforms(self, view):
        if False:
            i = 10
            return i + 15
        view.view_program.frag['fb_to_visual'] = view.transforms.get_transform('framebuffer', 'visual')
MyRect = scene.visuals.create_visual_node(MyRectVisual)
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'panzoom'
view.camera.rect = (0, 0, 800, 800)
vis = MyRect()
view.add(vis)
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        app.run()