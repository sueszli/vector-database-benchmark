from __future__ import annotations
import os
from xml.etree import ElementTree as ET
import numpy as np
import svgelements as se
import io
from manimlib.constants import RIGHT
from manimlib.logger import log
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Polygon
from manimlib.mobject.geometry import Polyline
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import RoundedRectangle
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.directories import get_mobject_data_dir
from manimlib.utils.images import get_full_vector_image_path
from manimlib.utils.iterables import hash_obj
from manimlib.utils.simple_functions import hash_string
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple
    from manimlib.typing import ManimColor, Vect3Array
SVG_HASH_TO_MOB_MAP: dict[int, list[VMobject]] = {}
PATH_TO_POINTS: dict[str, Vect3Array] = {}

def _convert_point_to_3d(x: float, y: float) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    return np.array([x, y, 0.0])

class SVGMobject(VMobject):
    file_name: str = ''
    height: float | None = 2.0
    width: float | None = None

    def __init__(self, file_name: str='', should_center: bool=True, height: float | None=None, width: float | None=None, color: ManimColor=None, fill_color: ManimColor=None, fill_opacity: float | None=None, stroke_width: float | None=0.0, stroke_color: ManimColor=None, stroke_opacity: float | None=None, svg_default: dict=dict(color=None, opacity=None, fill_color=None, fill_opacity=None, stroke_width=None, stroke_color=None, stroke_opacity=None), path_string_config: dict=dict(), **kwargs):
        if False:
            return 10
        self.file_name = file_name or self.file_name
        self.svg_default = dict(svg_default)
        self.path_string_config = dict(path_string_config)
        super().__init__(**kwargs)
        self.init_svg_mobject()
        self.ensure_positive_orientation()
        self.set_style(fill_color=color or fill_color, fill_opacity=fill_opacity, stroke_color=color or stroke_color, stroke_width=stroke_width, stroke_opacity=stroke_opacity)
        height = height or self.height
        width = width or self.width
        if should_center:
            self.center()
        if height is not None:
            self.set_height(height)
        if width is not None:
            self.set_width(width)

    def init_svg_mobject(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        hash_val = hash_obj(self.hash_seed)
        if hash_val in SVG_HASH_TO_MOB_MAP:
            submobs = [sm.copy() for sm in SVG_HASH_TO_MOB_MAP[hash_val]]
        else:
            submobs = self.mobjects_from_file(self.get_file_path())
            SVG_HASH_TO_MOB_MAP[hash_val] = [sm.copy() for sm in submobs]
        self.add(*submobs)
        self.flip(RIGHT)

    @property
    def hash_seed(self) -> tuple:
        if False:
            while True:
                i = 10
        return (self.__class__.__name__, self.svg_default, self.path_string_config, self.file_name)

    def mobjects_from_file(self, file_path: str) -> list[VMobject]:
        if False:
            i = 10
            return i + 15
        element_tree = ET.parse(file_path)
        new_tree = self.modify_xml_tree(element_tree)
        data_stream = io.BytesIO()
        new_tree.write(data_stream)
        data_stream.seek(0)
        svg = se.SVG.parse(data_stream)
        data_stream.close()
        return self.mobjects_from_svg(svg)

    def get_file_path(self) -> str:
        if False:
            print('Hello World!')
        if self.file_name is None:
            raise Exception('Must specify file for SVGMobject')
        return get_full_vector_image_path(self.file_name)

    def modify_xml_tree(self, element_tree: ET.ElementTree) -> ET.ElementTree:
        if False:
            while True:
                i = 10
        config_style_attrs = self.generate_config_style_dict()
        style_keys = ('fill', 'fill-opacity', 'stroke', 'stroke-opacity', 'stroke-width', 'style')
        root = element_tree.getroot()
        style_attrs = {k: v for (k, v) in root.attrib.items() if k in style_keys}
        SVG_XMLNS = '{http://www.w3.org/2000/svg}'
        new_root = ET.Element('svg')
        config_style_node = ET.SubElement(new_root, f'{SVG_XMLNS}g', config_style_attrs)
        root_style_node = ET.SubElement(config_style_node, f'{SVG_XMLNS}g', style_attrs)
        root_style_node.extend(root)
        return ET.ElementTree(new_root)

    def generate_config_style_dict(self) -> dict[str, str]:
        if False:
            while True:
                i = 10
        keys_converting_dict = {'fill': ('color', 'fill_color'), 'fill-opacity': ('opacity', 'fill_opacity'), 'stroke': ('color', 'stroke_color'), 'stroke-opacity': ('opacity', 'stroke_opacity'), 'stroke-width': ('stroke_width',)}
        svg_default_dict = self.svg_default
        result = {}
        for (svg_key, style_keys) in keys_converting_dict.items():
            for style_key in style_keys:
                if svg_default_dict[style_key] is None:
                    continue
                result[svg_key] = str(svg_default_dict[style_key])
        return result

    def mobjects_from_svg(self, svg: se.SVG) -> list[VMobject]:
        if False:
            return 10
        result = []
        for shape in svg.elements():
            if isinstance(shape, (se.Group, se.Use)):
                continue
            elif isinstance(shape, se.Path):
                mob = self.path_to_mobject(shape)
            elif isinstance(shape, se.SimpleLine):
                mob = self.line_to_mobject(shape)
            elif isinstance(shape, se.Rect):
                mob = self.rect_to_mobject(shape)
            elif isinstance(shape, (se.Circle, se.Ellipse)):
                mob = self.ellipse_to_mobject(shape)
            elif isinstance(shape, se.Polygon):
                mob = self.polygon_to_mobject(shape)
            elif isinstance(shape, se.Polyline):
                mob = self.polyline_to_mobject(shape)
            elif type(shape) == se.SVGElement:
                continue
            else:
                log.warning('Unsupported element type: %s', type(shape))
                continue
            if not mob.has_points():
                continue
            if isinstance(shape, se.GraphicObject):
                self.apply_style_to_mobject(mob, shape)
            if isinstance(shape, se.Transformable) and shape.apply:
                self.handle_transform(mob, shape.transform)
            result.append(mob)
        return result

    @staticmethod
    def handle_transform(mob: VMobject, matrix: se.Matrix) -> VMobject:
        if False:
            i = 10
            return i + 15
        mat = np.array([[matrix.a, matrix.c], [matrix.b, matrix.d]])
        vec = np.array([matrix.e, matrix.f, 0.0])
        mob.apply_matrix(mat)
        mob.shift(vec)
        return mob

    @staticmethod
    def apply_style_to_mobject(mob: VMobject, shape: se.GraphicObject) -> VMobject:
        if False:
            while True:
                i = 10
        mob.set_style(stroke_width=shape.stroke_width, stroke_color=shape.stroke.hexrgb, stroke_opacity=shape.stroke.opacity, fill_color=shape.fill.hexrgb, fill_opacity=shape.fill.opacity)
        return mob

    def path_to_mobject(self, path: se.Path) -> VMobjectFromSVGPath:
        if False:
            while True:
                i = 10
        return VMobjectFromSVGPath(path, **self.path_string_config)

    def line_to_mobject(self, line: se.SimpleLine) -> Line:
        if False:
            while True:
                i = 10
        return Line(start=_convert_point_to_3d(line.x1, line.y1), end=_convert_point_to_3d(line.x2, line.y2))

    def rect_to_mobject(self, rect: se.Rect) -> Rectangle:
        if False:
            for i in range(10):
                print('nop')
        if rect.rx == 0 or rect.ry == 0:
            mob = Rectangle(width=rect.width, height=rect.height)
        else:
            mob = RoundedRectangle(width=rect.width, height=rect.height * rect.rx / rect.ry, corner_radius=rect.rx)
            mob.stretch_to_fit_height(rect.height)
        mob.shift(_convert_point_to_3d(rect.x + rect.width / 2, rect.y + rect.height / 2))
        return mob

    def ellipse_to_mobject(self, ellipse: se.Circle | se.Ellipse) -> Circle:
        if False:
            while True:
                i = 10
        mob = Circle(radius=ellipse.rx)
        mob.stretch_to_fit_height(2 * ellipse.ry)
        mob.shift(_convert_point_to_3d(ellipse.cx, ellipse.cy))
        return mob

    def polygon_to_mobject(self, polygon: se.Polygon) -> Polygon:
        if False:
            for i in range(10):
                print('nop')
        points = [_convert_point_to_3d(*point) for point in polygon]
        return Polygon(*points)

    def polyline_to_mobject(self, polyline: se.Polyline) -> Polyline:
        if False:
            print('Hello World!')
        points = [_convert_point_to_3d(*point) for point in polyline]
        return Polyline(*points)

    def text_to_mobject(self, text: se.Text):
        if False:
            print('Hello World!')
        pass

class VMobjectFromSVGPath(VMobject):

    def __init__(self, path_obj: se.Path, **kwargs):
        if False:
            i = 10
            return i + 15
        path_obj.approximate_arcs_with_quads()
        self.path_obj = path_obj
        super().__init__(**kwargs)

    def init_points(self) -> None:
        if False:
            while True:
                i = 10
        path_string = self.path_obj.d()
        if path_string not in PATH_TO_POINTS:
            self.handle_commands()
            if not self._use_winding_fill:
                self.subdivide_intersections()
            PATH_TO_POINTS[path_string] = self.get_points().copy()
        else:
            points = PATH_TO_POINTS[path_string]
            self.set_points(points)

    def handle_commands(self) -> None:
        if False:
            return 10
        segment_class_to_func_map = {se.Move: (self.start_new_path, ('end',)), se.Close: (self.close_path, ()), se.Line: (self.add_line_to, ('end',)), se.QuadraticBezier: (self.add_quadratic_bezier_curve_to, ('control', 'end')), se.CubicBezier: (self.add_cubic_bezier_curve_to, ('control1', 'control2', 'end'))}
        for segment in self.path_obj:
            segment_class = segment.__class__
            (func, attr_names) = segment_class_to_func_map[segment_class]
            points = [_convert_point_to_3d(*segment.__getattribute__(attr_name)) for attr_name in attr_names]
            func(*points)
        if self.has_new_path_started():
            self.resize_points(self.get_num_points() - 2)