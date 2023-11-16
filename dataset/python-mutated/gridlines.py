from __future__ import division
from .image import ImageVisual
from ..color import Color
from .shaders import Function
_GRID_COLOR = '\nvec4 grid_color(vec2 pos) {\n    vec4 px_pos = $map_to_doc(vec4(pos, 0, 1));\n    px_pos /= px_pos.w;\n\n    // Compute vectors representing width, height of pixel in local coords\n    float s = 1.;\n    vec4 local_pos = $map_doc_to_local(px_pos);\n    vec4 dx = $map_doc_to_local(px_pos + vec4(1.0 / s, 0, 0, 0));\n    vec4 dy = $map_doc_to_local(px_pos + vec4(0, 1.0 / s, 0, 0));\n    local_pos /= local_pos.w;\n    dx = dx / dx.w - local_pos;\n    dy = dy / dy.w - local_pos;\n\n    // Pixel length along each axis, rounded to the nearest power of 10\n    vec2 px = s * vec2(abs(dx.x) + abs(dy.x), abs(dx.y) + abs(dy.y));\n    float log10 = log(10.0);\n    float sx = pow(10.0, floor(log(px.x) / log10) + 1.) * $scale.x;\n    float sy = pow(10.0, floor(log(px.y) / log10) + 1.) * $scale.y;\n\n    float max_alpha = 0.6;\n    float x_alpha = 0.0;\n\n    if (mod(local_pos.x, 1000. * sx) < px.x) {\n        x_alpha = clamp(1. * sx/px.x, 0., max_alpha);\n    }\n    else if (mod(local_pos.x, 100. * sx) < px.x) {\n        x_alpha = clamp(.1 * sx/px.x, 0., max_alpha);\n    }\n    else if (mod(local_pos.x, 10. * sx) < px.x) {\n        x_alpha = clamp(0.01 * sx/px.x, 0., max_alpha);\n    }\n\n    float y_alpha = 0.0;\n    if (mod(local_pos.y, 1000. * sy) < px.y) {\n        y_alpha = clamp(1. * sy/px.y, 0., max_alpha);\n    }\n    else if (mod(local_pos.y, 100. * sy) < px.y) {\n        y_alpha = clamp(.1 * sy/px.y, 0., max_alpha);\n    }\n    else if (mod(local_pos.y, 10. * sy) < px.y) {\n        y_alpha = clamp(0.01 * sy/px.y, 0., max_alpha);\n    }\n\n    float alpha = (((log(max(x_alpha, y_alpha))/log(10.)) + 2.) / 3.);\n    if (alpha == 0.) {\n        discard;\n    }\n    return vec4($color.rgb, $color.a * alpha);\n}\n'

class GridLinesVisual(ImageVisual):
    """Displays regularly spaced grid lines in any coordinate system and at
    any scale.

    Parameters
    ----------
    scale : tuple
        The scale factors to apply when determining the spacing of grid lines.
    color : Color
        The base color for grid lines. The final color may have its alpha
        channel modified.
    """

    def __init__(self, scale=(1, 1), color='w'):
        if False:
            return 10
        self._grid_color_fn = Function(_GRID_COLOR)
        self._grid_color_fn['color'] = Color(color).rgba
        self._grid_color_fn['scale'] = scale
        ImageVisual.__init__(self, method='impostor')
        self.set_gl_state('additive', cull_face=False)
        self.shared_program.frag['get_data'] = self._grid_color_fn
        cfun = Function('vec4 null(vec4 x) { return x; }')
        self.shared_program.frag['color_transform'] = cfun

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        return (1, 1)

    def _prepare_transforms(self, view):
        if False:
            return 10
        fn = self._grid_color_fn
        fn['map_to_doc'] = self.get_transform('visual', 'document')
        fn['map_doc_to_local'] = self.get_transform('document', 'visual')
        ImageVisual._prepare_transforms(self, view)

    def _prepare_draw(self, view):
        if False:
            print('Hello World!')
        if self._need_vertex_update:
            self._build_vertex_data()
        if view._need_method_update:
            self._update_method(view)