"""
Tutorial: Creating Visuals
==========================

03. Antialiasing
----------------

In [tutorial 1] we learned how to draw a simple rectangle, and in [tutorial 2]
we expanded on this by using the Document coordinate system to draw a 
rectangular border of a specific width. In this tutorial we introduce the
Framebuffer coordinate system, which is used for antialiasing measurements. 

In order to antialias our edges, we need to introduce a calculation to the
fragment shader that computes, for each pixel being drawn, the fraction of the 
pixel that is covered by the visual's geometry. At first glance, it may seem
that the Document coordinate system is sufficient for this purpose because it
has unit-length pixels. However, there are two situations when the actual 
pixels being filled by the fragment shader are not the same size as the pixels
on the canvas:

    1. High-resolution displays (such as retina displays) that report a canvas
       resolution smaller than the actual framebuffer resolution.
    2. When exporting to an image with a different size than the canvas.

In most cases the discrepancy between Document and Framebuffer coordinates can
be corrected by a simple scale factor. However, this fails for some interesting
corner cases where the transform is more complex, such as in VR applications 
using optical distortion correction. Decide for yourself: is this Visual for 
my personal use, or is it intended for a broader audience? For simplicity in 
this example, we will use a simple scale factor.
"""
from vispy import app, gloo, visuals, scene
import numpy as np
vertex_shader = '\nvarying float line_pos;  // how far we are across the border line\n\nvoid main() {\n    // First map the vertex to document coordinates\n    vec4 doc_pos = $visual_to_doc(vec4($position, 0, 1));\n    \n    vec4 adjusted;\n    if ( $adjust_dir.x == 0. ) {\n        adjusted = doc_pos;\n        line_pos = $line_width;  // at the outside of the border\n    }\n    else {\n        // Inner vertexes must be adjusted for line width\n        vec4 doc_x = $visual_to_doc(vec4($adjust_dir.x, 0, 0, 0)) - \n                    $visual_to_doc(vec4(0, 0, 0, 0));\n        vec4 doc_y = $visual_to_doc(vec4(0, $adjust_dir.y, 0, 0)) - \n                    $visual_to_doc(vec4(0, 0, 0, 0));\n        doc_x = normalize(doc_x);\n        doc_y = normalize(doc_y);\n                        \n        vec4 proj_y_x = dot(doc_x, doc_y) * doc_x;  // project y onto x\n        float cur_width = length(doc_y - proj_y_x);  // measure current weight\n        \n        // And now we can adjust vertex position for line width:\n        adjusted = doc_pos + ($line_width / cur_width) * (doc_x + doc_y);\n        \n        line_pos = 0;  // at the inside of the border\n    }\n    \n    // Finally map the remainder of the way to render coordinates\n    gl_Position = $doc_to_render(adjusted);\n}\n'
fragment_shader = "\nvarying float line_pos;\n\nvoid main() {\n    // Decrease the alpha linearly as we come within 1 pixel of the edge.\n    // Note: this only approximates the actual fraction of the pixel that is\n    // covered by the visual's geometry. A more accurate measurement would\n    // produce better antialiasing, but the effect would be subtle.\n    float alpha = 1.0;\n    if ((line_pos * $doc_fb_scale) < 1) {\n        alpha = $color.a * line_pos;\n    }\n    else if ((line_pos * $doc_fb_scale) > ($line_width - 1)) {\n        alpha = $color.a * ($line_width - line_pos);\n    }\n    gl_FragColor = vec4($color.rgb, alpha);\n}\n"

class MyRectVisual(visuals.Visual):
    """Visual that draws a rectangular outline.

    Parameters
    ----------
    x : float
        x coordinate of rectangle origin
    y : float
        y coordinate of rectangle origin
    w : float
        width of rectangle
    h : float
        height of rectangle
    weight : float
        width of border (in px)
    """

    def __init__(self, x, y, w, h, weight=4.0):
        if False:
            for i in range(10):
                print('nop')
        self.weight = weight
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        self.vert_buffer = gloo.VertexBuffer(np.array([[x, y], [x, y], [x + w, y], [x + w, y], [x + w, y + h], [x + w, y + h], [x, y + h], [x, y + h], [x, y], [x, y]], dtype=np.float32))
        self.adj_buffer = gloo.VertexBuffer(np.array([[0, 0], [1, 1], [0, 0], [-1, 1], [0, 0], [-1, -1], [0, 0], [1, -1], [0, 0], [1, 1]], dtype=np.float32))
        self.shared_program.vert['position'] = self.vert_buffer
        self.shared_program.vert['adjust_dir'] = self.adj_buffer
        self.shared_program.vert['line_width'] = weight + 1
        self.shared_program.frag['color'] = (1, 0, 0, 1)
        self._draw_mode = 'triangle_strip'
        self.set_gl_state(cull_face=False)

    def _prepare_transforms(self, view):
        if False:
            return 10
        tr = view.transforms
        view_vert = view.view_program.vert
        view_vert['visual_to_doc'] = tr.get_transform('visual', 'document')
        view_vert['doc_to_render'] = tr.get_transform('document', 'render')
        doc_to_fb = tr.get_transform('document', 'framebuffer')
        fbs = np.linalg.norm(doc_to_fb.map([1, 0]) - doc_to_fb.map([0, 0]))
        view_frag = view.view_program.frag
        view_frag['doc_fb_scale'] = fbs
        view_frag['line_width'] = (self.weight + 1) * fbs
MyRect = scene.visuals.create_visual_node(MyRectVisual)
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'panzoom'
view.camera.rect = (0, 0, 800, 800)
rects = [MyRect(100, 100, 200, 300, parent=view.scene), MyRect(500, 100, 200, 300, parent=view.scene)]
tr = visuals.transforms.MatrixTransform()
tr.rotate(25, (0, 0, 1))
rects[1].transform = tr
text = scene.visuals.Text('Drag right mouse button to zoom.', color='w', anchor_x='left', parent=view, pos=(20, 30))
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        app.run()