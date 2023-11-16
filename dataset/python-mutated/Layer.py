from typing import List
import numpy
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.Mesh.MeshData import MeshData
from cura.LayerPolygon import LayerPolygon

class Layer:

    def __init__(self, layer_id: int) -> None:
        if False:
            i = 10
            return i + 15
        self._id = layer_id
        self._height = 0.0
        self._thickness = 0.0
        self._polygons = []
        self._element_count = 0

    @property
    def height(self):
        if False:
            print('Hello World!')
        return self._height

    @property
    def thickness(self):
        if False:
            for i in range(10):
                print('nop')
        return self._thickness

    @property
    def polygons(self) -> List[LayerPolygon]:
        if False:
            i = 10
            return i + 15
        return self._polygons

    @property
    def elementCount(self):
        if False:
            print('Hello World!')
        return self._element_count

    def setHeight(self, height: float) -> None:
        if False:
            return 10
        self._height = height

    def setThickness(self, thickness: float) -> None:
        if False:
            print('Hello World!')
        self._thickness = thickness

    def lineMeshVertexCount(self) -> int:
        if False:
            print('Hello World!')
        result = 0
        for polygon in self._polygons:
            result += polygon.lineMeshVertexCount()
        return result

    def lineMeshElementCount(self) -> int:
        if False:
            return 10
        result = 0
        for polygon in self._polygons:
            result += polygon.lineMeshElementCount()
        return result

    def build(self, vertex_offset, index_offset, vertices, colors, line_dimensions, feedrates, extruders, line_types, indices):
        if False:
            i = 10
            return i + 15
        result_vertex_offset = vertex_offset
        result_index_offset = index_offset
        self._element_count = 0
        for polygon in self._polygons:
            polygon.build(result_vertex_offset, result_index_offset, vertices, colors, line_dimensions, feedrates, extruders, line_types, indices)
            result_vertex_offset += polygon.lineMeshVertexCount()
            result_index_offset += polygon.lineMeshElementCount()
            self._element_count += polygon.elementCount
        return (result_vertex_offset, result_index_offset)

    def createMesh(self) -> MeshData:
        if False:
            return 10
        return self.createMeshOrJumps(True)

    def createJumps(self) -> MeshData:
        if False:
            print('Hello World!')
        return self.createMeshOrJumps(False)
    __index_pattern = numpy.array([[0, 3, 2, 0, 1, 3]], dtype=numpy.int32)

    def createMeshOrJumps(self, make_mesh: bool) -> MeshData:
        if False:
            while True:
                i = 10
        builder = MeshBuilder()
        line_count = 0
        if make_mesh:
            for polygon in self._polygons:
                line_count += polygon.meshLineCount
        else:
            for polygon in self._polygons:
                line_count += polygon.jumpCount
        builder.reserveFaceAndVertexCount(2 * line_count, 4 * line_count)
        for polygon in self._polygons:
            index_mask = numpy.logical_not(polygon.jumpMask) if make_mesh else polygon.jumpMask
            points = numpy.concatenate((polygon.data[:-1], polygon.data[1:]), 1)[index_mask.ravel()]
            line_types = polygon.types[index_mask]
            if make_mesh:
                points[polygon.isInfillOrSkinType(line_types), 1::3] -= 0.01
            else:
                points[:, 1::3] += 0.01
            normals = numpy.tile(polygon.getNormals()[index_mask.ravel()], (1, 2))
            normals *= polygon.lineWidths[index_mask.ravel()] / 2
            f_points = numpy.concatenate((points - normals, points + normals), 1).reshape((-1, 3))
            f_indices = (self.__index_pattern + numpy.arange(0, 4 * len(normals), 4, dtype=numpy.int32).reshape((-1, 1))).reshape((-1, 3))
            f_colors = numpy.repeat(polygon.mapLineTypeToColor(line_types), 4, 0)
            builder.addFacesWithColor(f_points, f_indices, f_colors)
        return builder.build()