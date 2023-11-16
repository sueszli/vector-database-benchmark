import math
from typing import List, Optional
from .scene_object3d import Object3D

class Group(Object3D):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        "Group\n\n        This element is based on Three.js' `Group <https://threejs.org/docs/index.html#api/en/objects/Group>`_ object.\n        It is used to group objects together.\n        "
        super().__init__('group')

class Box(Object3D):

    def __init__(self, width: float=1.0, height: float=1.0, depth: float=1.0, wireframe: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        "Box\n\n        This element is based on Three.js' `BoxGeometry <https://threejs.org/docs/index.html#api/en/geometries/BoxGeometry>`_ object.\n        It is used to create a box-shaped mesh.\n\n        :param width: width of the box (default: 1.0)\n        :param height: height of the box (default: 1.0)\n        :param depth: depth of the box (default: 1.0)\n        :param wireframe: whether to display the box as a wireframe (default: `False`)\n        "
        super().__init__('box', width, height, depth, wireframe)

class Sphere(Object3D):

    def __init__(self, radius: float=1.0, width_segments: int=32, height_segments: int=16, wireframe: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Sphere\n\n        This element is based on Three.js' `SphereGeometry <https://threejs.org/docs/index.html#api/en/geometries/SphereGeometry>`_ object.\n        It is used to create a sphere-shaped mesh.\n\n        :param radius: radius of the sphere (default: 1.0)\n        :param width_segments: number of horizontal segments (default: 32)\n        :param height_segments: number of vertical segments (default: 16)\n        :param wireframe: whether to display the sphere as a wireframe (default: `False`)\n        "
        super().__init__('sphere', radius, width_segments, height_segments, wireframe)

class Cylinder(Object3D):

    def __init__(self, top_radius: float=1.0, bottom_radius: float=1.0, height: float=1.0, radial_segments: int=8, height_segments: int=1, wireframe: bool=False) -> None:
        if False:
            while True:
                i = 10
        "Cylinder\n\n        This element is based on Three.js' `CylinderGeometry <https://threejs.org/docs/index.html#api/en/geometries/CylinderGeometry>`_ object.\n        It is used to create a cylinder-shaped mesh.\n\n        :param top_radius: radius of the top (default: 1.0)\n        :param bottom_radius: radius of the bottom (default: 1.0)\n        :param height: height of the cylinder (default: 1.0)\n        :param radial_segments: number of horizontal segments (default: 8)\n        :param height_segments: number of vertical segments (default: 1)\n        :param wireframe: whether to display the cylinder as a wireframe (default: `False`)\n        "
        super().__init__('cylinder', top_radius, bottom_radius, height, radial_segments, height_segments, wireframe)

class Ring(Object3D):

    def __init__(self, inner_radius: float=0.5, outer_radius: float=1.0, theta_segments: int=8, phi_segments: int=1, theta_start: float=0, theta_length: float=2 * math.pi, wireframe: bool=False) -> None:
        if False:
            print('Hello World!')
        "Ring\n\n        This element is based on Three.js' `RingGeometry <https://threejs.org/docs/index.html#api/en/geometries/RingGeometry>`_ object.\n        It is used to create a ring-shaped mesh.\n\n        :param inner_radius: inner radius of the ring (default: 0.5)\n        :param outer_radius: outer radius of the ring (default: 1.0)\n        :param theta_segments: number of horizontal segments (default: 8, higher means rounder)\n        :param phi_segments: number of vertical segments (default: 1)\n        :param theta_start: start angle in radians (default: 0)\n        :param theta_length: central angle in radians (default: 2π)\n        :param wireframe: whether to display the ring as a wireframe (default: `False`)\n        "
        super().__init__('ring', inner_radius, outer_radius, theta_segments, phi_segments, theta_start, theta_length, wireframe)

class QuadraticBezierTube(Object3D):

    def __init__(self, start: List[float], mid: List[float], end: List[float], tubular_segments: int=64, radius: float=1.0, radial_segments: int=8, closed: bool=False, wireframe: bool=False) -> None:
        if False:
            while True:
                i = 10
        "Quadratic Bezier Tube\n\n        This element is based on Three.js' `QuadraticBezierCurve3 <https://threejs.org/docs/index.html#api/en/extras/curves/QuadraticBezierCurve3>`_ object.\n        It is used to create a tube-shaped mesh.\n\n        :param start: start point of the curve\n        :param mid: middle point of the curve\n        :param end: end point of the curve\n        :param tubular_segments: number of tubular segments (default: 64)\n        :param radius: radius of the tube (default: 1.0)\n        :param radial_segments: number of radial segments (default: 8)\n        :param closed: whether the tube should be closed (default: `False`)\n        :param wireframe: whether to display the tube as a wireframe (default: `False`)\n        "
        super().__init__('quadratic_bezier_tube', start, mid, end, tubular_segments, radius, radial_segments, closed, wireframe)

class Extrusion(Object3D):

    def __init__(self, outline: List[List[float]], height: float, wireframe: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        "Extrusion\n\n        This element is based on Three.js' `ExtrudeGeometry <https://threejs.org/docs/index.html#api/en/geometries/ExtrudeGeometry>`_ object.\n        It is used to create a 3D shape by extruding a 2D shape to a given height.\n\n        :param outline: list of points defining the outline of the 2D shape\n        :param height: height of the extrusion\n        :param wireframe: whether to display the extrusion as a wireframe (default: `False`)\n        "
        super().__init__('extrusion', outline, height, wireframe)

class Stl(Object3D):

    def __init__(self, url: str, wireframe: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'STL\n\n        This element is used to create a mesh from an STL file.\n\n        :param url: URL of the STL file\n        :param wireframe: whether to display the STL as a wireframe (default: `False`)\n        '
        super().__init__('stl', url, wireframe)

class Line(Object3D):

    def __init__(self, start: List[float], end: List[float]) -> None:
        if False:
            return 10
        "Line\n\n        This element is based on Three.js' `Line <https://threejs.org/docs/index.html#api/en/objects/Line>`_ object.\n        It is used to create a line segment.\n\n        :param start: start point of the line\n        :param end: end point of the line\n        "
        super().__init__('line', start, end)

class Curve(Object3D):

    def __init__(self, start: List[float], control1: List[float], control2: List[float], end: List[float], num_points: int=20) -> None:
        if False:
            i = 10
            return i + 15
        "Curve\n\n        This element is based on Three.js' `CubicBezierCurve3 <https://threejs.org/docs/index.html#api/en/extras/curves/CubicBezierCurve3>`_ object.\n\n        :param start: start point of the curve\n        :param control1: first control point of the curve\n        :param control2: second control point of the curve\n        :param end: end point of the curve\n        :param num_points: number of points to use for the curve (default: 20)\n        "
        super().__init__('curve', start, control1, control2, end, num_points)

class Text(Object3D):

    def __init__(self, text: str, style: str='') -> None:
        if False:
            while True:
                i = 10
        "Text\n\n        This element is used to add 2D text to the scene.\n        It can be moved like any other object, but always faces the camera.\n\n        :param text: text to display\n        :param style: CSS style (default: '')\n        "
        super().__init__('text', text, style)

class Text3d(Object3D):

    def __init__(self, text: str, style: str='') -> None:
        if False:
            print('Hello World!')
        "3D Text\n\n        This element is used to add a 3D text mesh to the scene.\n        It can be moved and rotated like any other object.\n\n        :param text: text to display\n        :param style: CSS style (default: '')\n        "
        super().__init__('text3d', text, style)

class Texture(Object3D):

    def __init__(self, url: str, coordinates: List[List[Optional[List[float]]]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Texture\n\n        This element is used to add a texture to a mesh.\n\n        :param url: URL of the texture image\n        :param coordinates: texture coordinates\n        '
        super().__init__('texture', url, coordinates)

    def set_url(self, url: str) -> None:
        if False:
            while True:
                i = 10
        'Change the URL of the texture image.'
        self.args[0] = url
        self.scene.run_method('set_texture_url', self.id, url)

    def set_coordinates(self, coordinates: List[List[Optional[List[float]]]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Change the texture coordinates.'
        self.args[1] = coordinates
        self.scene.run_method('set_texture_coordinates', self.id, coordinates)

class SpotLight(Object3D):

    def __init__(self, color: str='#ffffff', intensity: float=1.0, distance: float=0.0, angle: float=math.pi / 3, penumbra: float=0.0, decay: float=1.0) -> None:
        if False:
            print('Hello World!')
        "Spot Light\n\n        This element is based on Three.js' `SpotLight <https://threejs.org/docs/index.html#api/en/lights/SpotLight>`_ object.\n        It is used to add a spot light to the scene.\n\n        :param color: CSS color string (default: '#ffffff')\n        :param intensity: light intensity (default: 1.0)\n        :param distance: maximum distance of light (default: 0.0)\n        :param angle: maximum angle of light (default: π/2)\n        :param penumbra: penumbra (default: 0.0)\n        :param decay: decay (default: 2.0)\n        "
        super().__init__('spot_light', color, intensity, distance, angle, penumbra, decay)

class PointCloud(Object3D):

    def __init__(self, points: List[List[float]], colors: List[List[float]], point_size: float=1.0) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Point Cloud\n\n        This element is based on Three.js' `Points <https://threejs.org/docs/index.html#api/en/objects/Points>`_ object.\n\n        :param points: list of points\n        :param colors: list of colors (one per point)\n        :param point_size: size of the points (default: 1.0)\n        "
        super().__init__('point_cloud', points, colors, point_size)