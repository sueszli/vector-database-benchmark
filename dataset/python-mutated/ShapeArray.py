import numpy
import copy
from typing import Optional, Tuple, TYPE_CHECKING, Union
from UM.Math.Polygon import Polygon
if TYPE_CHECKING:
    from UM.Scene.SceneNode import SceneNode

class ShapeArray:
    """Polygon representation as an array for use with :py:class:`cura.Arranging.Arrange.Arrange`"""

    def __init__(self, arr: numpy.ndarray, offset_x: float, offset_y: float, scale: float=1) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.arr = arr
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.scale = scale

    @classmethod
    def fromPolygon(cls, vertices: numpy.ndarray, scale: float=1) -> 'ShapeArray':
        if False:
            print('Hello World!')
        'Instantiate from a bunch of vertices\n\n        :param vertices:\n        :param scale:  scale the coordinates\n        :return: a shape array instantiated from a bunch of vertices\n        '
        vertices = vertices * scale
        flip_vertices = numpy.zeros(vertices.shape)
        flip_vertices[:, 0] = vertices[:, 1]
        flip_vertices[:, 1] = vertices[:, 0]
        flip_vertices = flip_vertices[::-1]
        offset_y = int(numpy.amin(flip_vertices[:, 0]))
        offset_x = int(numpy.amin(flip_vertices[:, 1]))
        flip_vertices[:, 0] = numpy.add(flip_vertices[:, 0], -offset_y)
        flip_vertices[:, 1] = numpy.add(flip_vertices[:, 1], -offset_x)
        shape = numpy.array([int(numpy.amax(flip_vertices[:, 0])), int(numpy.amax(flip_vertices[:, 1]))])
        shape[numpy.where(shape == 0)] = 1
        arr = cls.arrayFromPolygon(shape, flip_vertices)
        if not numpy.ndarray.any(arr):
            arr[0][0] = 1
        return cls(arr, offset_x, offset_y)

    @classmethod
    def fromNode(cls, node: 'SceneNode', min_offset: float, scale: float=0.5, include_children: bool=False) -> Tuple[Optional['ShapeArray'], Optional['ShapeArray']]:
        if False:
            i = 10
            return i + 15
        'Instantiate an offset and hull ShapeArray from a scene node.\n\n        :param node: source node where the convex hull must be present\n        :param min_offset: offset for the offset ShapeArray\n        :param scale: scale the coordinates\n        :return: A tuple containing an offset and hull shape array\n        '
        transform = node._transformation
        transform_x = transform._data[0][3]
        transform_y = transform._data[2][3]
        hull_verts = node.callDecoration('getConvexHull')
        if hull_verts is None or not hull_verts.getPoints().any():
            return (None, None)
        hull_head_verts = node.callDecoration('getConvexHullHead') or hull_verts
        if hull_head_verts is None:
            hull_head_verts = Polygon()
        if include_children:
            children = node.getAllChildren()
            if children is not None:
                for child in children:
                    child_hull = child.callDecoration('getConvexHull')
                    if child_hull is not None:
                        hull_verts = hull_verts.unionConvexHulls(child_hull)
                    child_hull_head = child.callDecoration('getConvexHullHead') or child_hull
                    if child_hull_head is not None:
                        hull_head_verts = hull_head_verts.unionConvexHulls(child_hull_head)
        offset_verts = hull_head_verts.getMinkowskiHull(Polygon.approximatedCircle(min_offset))
        offset_points = copy.deepcopy(offset_verts._points)
        offset_points[:, 0] = numpy.add(offset_points[:, 0], -transform_x)
        offset_points[:, 1] = numpy.add(offset_points[:, 1], -transform_y)
        offset_shape_arr = ShapeArray.fromPolygon(offset_points, scale=scale)
        hull_points = copy.deepcopy(hull_verts._points)
        hull_points[:, 0] = numpy.add(hull_points[:, 0], -transform_x)
        hull_points[:, 1] = numpy.add(hull_points[:, 1], -transform_y)
        hull_shape_arr = ShapeArray.fromPolygon(hull_points, scale=scale)
        return (offset_shape_arr, hull_shape_arr)

    @classmethod
    def arrayFromPolygon(cls, shape: Union[Tuple[int, int], numpy.ndarray], vertices: numpy.ndarray) -> numpy.ndarray:
        if False:
            i = 10
            return i + 15
        'Create :py:class:`numpy.ndarray` with dimensions defined by shape\n\n        Fills polygon defined by vertices with ones, all other values zero\n        Only works correctly for convex hull vertices\n        Originally from: `Stackoverflow - generating a filled polygon inside a numpy array <https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array>`_\n\n        :param shape:  numpy format shape, [x-size, y-size]\n        :param vertices:\n        :return: numpy array with dimensions defined by shape\n        '
        base_array = numpy.zeros(shape, dtype=numpy.int32)
        fill = numpy.ones(base_array.shape) * True
        for k in range(vertices.shape[0]):
            check_array = cls._check(vertices[k - 1], vertices[k], base_array)
            if check_array is not None:
                fill = numpy.all([fill, check_array], axis=0)
        base_array[fill] = 1
        return base_array

    @classmethod
    def _check(cls, p1: numpy.ndarray, p2: numpy.ndarray, base_array: numpy.ndarray) -> Optional[numpy.ndarray]:
        if False:
            return 10
        'Return indices that mark one side of the line, used by arrayFromPolygon\n\n        Uses the line defined by p1 and p2 to check array of\n        input indices against interpolated value\n        Returns boolean array, with True inside and False outside of shape\n        Originally from: `Stackoverflow - generating a filled polygon inside a numpy array <https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array>`_\n\n        :param p1: 2-tuple with x, y for point 1\n        :param p2: 2-tuple with x, y for point 2\n        :param base_array: boolean array to project the line on\n        :return: A numpy array with indices that mark one side of the line\n        '
        if p1[0] == p2[0] and p1[1] == p2[1]:
            return None
        idxs = numpy.indices(base_array.shape)
        p1 = p1.astype(float)
        p2 = p2.astype(float)
        if p2[0] == p1[0]:
            sign = numpy.sign(p2[1] - p1[1])
            return idxs[1] * sign
        if p2[1] == p1[1]:
            sign = numpy.sign(p2[0] - p1[0])
            return idxs[1] * sign
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = numpy.sign(p2[0] - p1[0])
        return idxs[1] * sign <= max_col_idx * sign