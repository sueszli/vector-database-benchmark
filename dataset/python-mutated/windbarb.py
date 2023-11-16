"""Windbarb Visual and shader definitions."""
import numpy as np
from vispy.color import ColorArray
from vispy.gloo import VertexBuffer
from vispy.visuals.shaders import Variable
from vispy.visuals.visual import Visual
_VERTEX_SHADER = '\nuniform float u_antialias;\nuniform float u_px_scale;\nuniform float u_scale;\n\nattribute vec3  a_position;\nattribute vec2  a_wind;\nattribute vec4  a_fg_color;\nattribute vec4  a_bg_color;\nattribute float a_edgewidth;\nattribute float a_size;\nattribute float a_trig;\n\nvarying vec4 v_fg_color;\nvarying vec4 v_bg_color;\nvarying vec2 v_wind;\nvarying float v_trig;\nvarying float v_edgewidth;\nvarying float v_antialias;\n\nvoid main (void) {\n    $v_size = a_size * u_px_scale * u_scale;\n    v_edgewidth = a_edgewidth * float(u_px_scale);\n    v_wind = a_wind.xy;\n    v_trig = a_trig;\n    v_antialias = u_antialias;\n    v_fg_color  = a_fg_color;\n    v_bg_color  = a_bg_color;\n    gl_Position = $transform(vec4(a_position,1.0));\n    float edgewidth = max(v_edgewidth, 1.0);\n    gl_PointSize = ($v_size) + 4.*(edgewidth + 1.5*v_antialias);\n}\n'
_FRAGMENT_SHADER = '\n#include "math/constants.glsl"\n#include "math/signed-segment-distance.glsl"\n#include "antialias/antialias.glsl"\nvarying vec4 v_fg_color;\nvarying vec4 v_bg_color;\nvarying vec2 v_wind;\nvarying float v_trig;\nvarying float v_edgewidth;\nvarying float v_antialias;\n\n// SDF-Triangle by @rougier\n// https://github.com/rougier/python-opengl/blob/master/code/chapter-06/SDF-triangle.py\nfloat sdf_triangle(vec2 p, vec2 p0, vec2 p1, vec2 p2)\n{\n    vec2 e0 = p1 - p0;\n    vec2 e1 = p2 - p1;\n    vec2 e2 = p0 - p2;\n    vec2 v0 = p - p0;\n    vec2 v1 = p - p1;\n    vec2 v2 = p - p2;\n    vec2 pq0 = v0 - e0*clamp( dot(v0,e0)/dot(e0,e0), 0.0, 1.0 );\n    vec2 pq1 = v1 - e1*clamp( dot(v1,e1)/dot(e1,e1), 0.0, 1.0 );\n    vec2 pq2 = v2 - e2*clamp( dot(v2,e2)/dot(e2,e2), 0.0, 1.0 );\n    float s = sign( e0.x*e2.y - e0.y*e2.x );\n    vec2 d = min( min( vec2( dot( pq0, pq0 ), s*(v0.x*e0.y-v0.y*e0.x) ),\n                     vec2( dot( pq1, pq1 ), s*(v1.x*e1.y-v1.y*e1.x) )),\n                     vec2( dot( pq2, pq2 ), s*(v2.x*e2.y-v2.y*e2.x) ));\n    return -sqrt(d.x)*sign(d.y);\n}\n\nvoid main()\n{\n    // Discard plotting marker body and edge if zero-size\n    if ($v_size <= 0.)\n        discard;\n\n    float edgewidth = max(v_edgewidth, 1.0);\n    float linewidth = max(v_edgewidth, 1.0);\n    float edgealphafactor = min(v_edgewidth, 1.0);\n\n    float size = $v_size + 4.*(edgewidth + 1.5*v_antialias);\n    // factor 6 for acute edge angles that need room as for star marker\n    \n    vec2 wind = v_wind;\n    \n    if (v_trig > 0.)\n    {\n        float u = wind.x * cos(radians(wind.y));\n        float v = wind.x * sin(radians(wind.y));\n        wind = vec2(u, v);\n    }\n    \n    // knots to m/s\n    wind *= 2.;\n    \n    // normalized distance\n    float dx = 0.5;\n    // normalized center point\n    vec2 O = vec2(dx);\n    // normalized x-component\n    vec2 X = normalize(wind) * dx / M_SQRT2 / 1.1 * vec2(1, -1);\n    // normalized y-component\n    // here the barb can be mirrored for southern earth * (vec2(1., -1.)\n    //vec2 Y = X.yx * vec2(1., -1.); // southern hemisphere\n    vec2 Y = X.yx * vec2(-1., 1.); // northern hemisphere\n    // PointCoordinate\n    vec2 P = gl_PointCoord;\n\n    // calculate barb items\n    float speed = length(wind);\n    int flag = int(floor(speed / 50.));\n    speed -= float (50 * flag);\n    int longbarb = int(floor(speed / 10.));\n    speed -= float (longbarb * 10);\n    int shortbarb = int(floor(speed / 5.));\n    int calm = shortbarb + longbarb + flag;\n\n    // starting distance\n    float r;\n    // calm, plot circles\n    if (calm == 0)\n    {\n        r = abs(length(O-P)- dx * 0.2);\n        r = min(r, abs(length(O-P)- dx * 0.1));\n    }\n    else\n    {\n        // plot shaft\n        r = segment_distance(P, O, O-X);\n        float pos = 1.;\n\n        // plot flag(s)\n        while(flag >= 1)\n        {\n            r = min(r, sdf_triangle(P, O-X*pos, O-X*pos-X*.4-Y*.4, O-X*pos-X*.4));\n            flag -= 1;\n            pos -= 0.15;\n        }\n        // plot longbarb(s)\n        while(longbarb >= 1)\n        {\n            r = min(r, segment_distance(P, O-X*pos, O-X*pos-X*.4-Y*.4));\n            longbarb -= 1;\n            pos -= 0.15;\n        }\n        // plot shortbarb\n        while(shortbarb >= 1)\n        {\n            if (pos == 1.0)\n                pos -= 0.15;\n            r = min(r, segment_distance(P, O-X*pos, O-X*pos-X*.2-Y*.2));\n            shortbarb -= 1;\n            pos -= 0.15;\n        }\n    }\n\n    // apply correction for size\n    r *= size;\n\n    vec4 edgecolor = vec4(v_fg_color.rgb, edgealphafactor*v_fg_color.a);\n\n    if (r > 0.5 * v_edgewidth + v_antialias)\n    {\n        // out of the marker (beyond the outer edge of the edge\n        // including transition zone due to antialiasing)\n        discard;\n    }\n\n    gl_FragColor = filled(r, edgewidth, v_antialias, edgecolor);\n}\n'

class WindbarbVisual(Visual):
    """Visual displaying windbarbs."""
    _shaders = {'vertex': _VERTEX_SHADER, 'fragment': _FRAGMENT_SHADER}

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self._vbo = VertexBuffer()
        self._v_size_var = Variable('varying float v_size')
        self._marker_fun = None
        self._data = None
        Visual.__init__(self, vcode=self._shaders['vertex'], fcode=self._shaders['fragment'])
        self.shared_program.vert['v_size'] = self._v_size_var
        self.shared_program.frag['v_size'] = self._v_size_var
        self.set_gl_state(depth_test=True, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self._draw_mode = 'points'
        if len(kwargs) > 0:
            self.set_data(**kwargs)
        self.freeze()

    def set_data(self, pos=None, wind=None, trig=True, size=50.0, antialias=1.0, edge_width=1.0, edge_color='black', face_color='white'):
        if False:
            for i in range(10):
                print('nop')
        'Set the data used to display this visual.\n\n        Parameters\n        ----------\n        pos : array\n            The array of locations to display each windbarb.\n        wind : array\n            The array of wind vector components to display each windbarb.\n            in m/s. For knots divide by two.\n        trig : bool\n            True - wind contains (mag, ang)\n            False - wind contains (u, v)\n            defaults to True\n        size : float or array\n            The windbarb size in px.\n        antialias : float\n            The antialiased area (in pixels).\n        edge_width : float | None\n            The width of the windbarb outline in pixels.\n        edge_color : Color | ColorArray\n            The color used to draw each symbol outline.\n        face_color : Color | ColorArray\n            The color used to draw each symbol interior.\n        '
        assert isinstance(pos, np.ndarray) and pos.ndim == 2 and (pos.shape[1] in (2, 3))
        assert isinstance(wind, np.ndarray) and pos.ndim == 2 and (pos.shape[1] == 2)
        if edge_width < 0:
            raise ValueError('edge_width cannot be negative')
        size *= 2
        edge_color = ColorArray(edge_color).rgba
        if len(edge_color) == 1:
            edge_color = edge_color[0]
        face_color = ColorArray(face_color).rgba
        if len(face_color) == 1:
            face_color = face_color[0]
        n = len(pos)
        data = np.zeros(n, dtype=[('a_position', np.float32, 3), ('a_wind', np.float32, 2), ('a_trig', np.float32, 0), ('a_fg_color', np.float32, 4), ('a_bg_color', np.float32, 4), ('a_size', np.float32), ('a_edgewidth', np.float32)])
        data['a_fg_color'] = edge_color
        data['a_bg_color'] = face_color
        data['a_edgewidth'] = edge_width
        data['a_position'][:, :pos.shape[1]] = pos
        data['a_wind'][:, :wind.shape[1]] = wind
        if trig:
            data['a_trig'] = 1.0
        else:
            data['a_trig'] = 0.0
        data['a_size'] = size
        self.shared_program['u_antialias'] = antialias
        self._data = data
        self._vbo.set_data(data)
        self.shared_program.bind(self._vbo)
        self.update()

    def _prepare_transforms(self, view):
        if False:
            while True:
                i = 10
        xform = view.transforms.get_transform()
        view.view_program.vert['transform'] = xform

    def _prepare_draw(self, view):
        if False:
            i = 10
            return i + 15
        view.view_program['u_px_scale'] = view.transforms.pixel_scale
        view.view_program['u_scale'] = 1

    def _compute_bounds(self, axis, view):
        if False:
            print('Hello World!')
        pos = self._data['a_position']
        if pos is None:
            return None
        if pos.shape[1] > axis:
            return (pos[:, axis].min(), pos[:, axis].max())
        else:
            return (0, 0)