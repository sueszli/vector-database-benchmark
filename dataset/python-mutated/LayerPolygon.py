import numpy
from typing import Optional, cast
from UM.Qt.Bindings.Theme import Theme
from UM.Qt.QtApplication import QtApplication
from UM.Logger import Logger

class LayerPolygon:
    NoneType = 0
    Inset0Type = 1
    InsetXType = 2
    SkinType = 3
    SupportType = 4
    SkirtType = 5
    InfillType = 6
    SupportInfillType = 7
    MoveCombingType = 8
    MoveRetractionType = 9
    SupportInterfaceType = 10
    PrimeTowerType = 11
    __number_of_types = 12
    __jump_map = numpy.logical_or(numpy.logical_or(numpy.arange(__number_of_types) == NoneType, numpy.arange(__number_of_types) == MoveCombingType), numpy.arange(__number_of_types) == MoveRetractionType)

    def __init__(self, extruder: int, line_types: numpy.ndarray, data: numpy.ndarray, line_widths: numpy.ndarray, line_thicknesses: numpy.ndarray, line_feedrates: numpy.ndarray) -> None:
        if False:
            return 10
        'LayerPolygon, used in ProcessSlicedLayersJob\n\n        :param extruder: The position of the extruder\n        :param line_types: array with line_types\n        :param data: new_points\n        :param line_widths: array with line widths\n        :param line_thicknesses: array with type as index and thickness as value\n        :param line_feedrates: array with line feedrates\n        '
        self._extruder = extruder
        self._types = line_types
        unknown_types = numpy.where(self._types >= self.__number_of_types, self._types, None)
        if unknown_types.any():
            for idx in unknown_types:
                Logger.warning(f'Found an unknown line type at: {idx}')
                self._types[idx] = self.NoneType
        self._data = data
        self._line_widths = line_widths
        self._line_thicknesses = line_thicknesses
        self._line_feedrates = line_feedrates
        self._vertex_begin = 0
        self._vertex_end = 0
        self._index_begin = 0
        self._index_end = 0
        self._jump_mask = self.__jump_map[self._types]
        self._jump_count = numpy.sum(self._jump_mask)
        self._mesh_line_count = len(self._types) - self._jump_count
        self._vertex_count = self._mesh_line_count + numpy.sum(self._types[1:] == self._types[:-1])
        self._color_map = LayerPolygon.getColorMap()
        self._colors = self._color_map[self._types]
        self._is_infill_or_skin_type_map = numpy.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
        self._build_cache_line_mesh_mask = None
        self._build_cache_needed_points = None

    def buildCache(self) -> None:
        if False:
            return 10
        self._build_cache_line_mesh_mask = numpy.ones(self._jump_mask.shape, dtype=bool)
        self._index_begin = 0
        self._index_end = cast(int, numpy.sum(self._build_cache_line_mesh_mask))
        self._build_cache_needed_points = numpy.ones((len(self._types), 2), dtype=bool)
        self._build_cache_needed_points[1:, 0][:, numpy.newaxis] = self._types[1:] != self._types[:-1]
        numpy.logical_and(self._build_cache_needed_points, self._build_cache_line_mesh_mask, self._build_cache_needed_points)
        self._vertex_begin = 0
        self._vertex_end = cast(int, numpy.sum(self._build_cache_needed_points))

    def build(self, vertex_offset: int, index_offset: int, vertices: numpy.ndarray, colors: numpy.ndarray, line_dimensions: numpy.ndarray, feedrates: numpy.ndarray, extruders: numpy.ndarray, line_types: numpy.ndarray, indices: numpy.ndarray) -> None:
        if False:
            i = 10
            return i + 15
        'Set all the arrays provided by the function caller, representing the LayerPolygon\n\n        The arrays are either by vertex or by indices.\n\n        :param vertex_offset: determines where to start and end filling the arrays\n        :param index_offset: determines where to start and end filling the arrays\n        :param vertices: vertex numpy array to be filled\n        :param colors: vertex numpy array to be filled\n        :param line_dimensions: vertex numpy array to be filled\n        :param feedrates: vertex numpy array to be filled\n        :param extruders: vertex numpy array to be filled\n        :param line_types: vertex numpy array to be filled\n        :param indices: index numpy array to be filled\n        '
        if self._build_cache_line_mesh_mask is None or self._build_cache_needed_points is None:
            self.buildCache()
        if self._build_cache_line_mesh_mask is None or self._build_cache_needed_points is None:
            Logger.log('w', 'Failed to build cache for layer polygon')
            return
        line_mesh_mask = self._build_cache_line_mesh_mask
        needed_points_list = self._build_cache_needed_points
        index_list = (numpy.arange(len(self._types)).reshape((-1, 1)) + numpy.array([[0, 1]])).reshape((-1, 1))[needed_points_list.reshape((-1, 1))]
        self._vertex_begin += vertex_offset
        self._vertex_end += vertex_offset
        vertices[self._vertex_begin:self._vertex_end, :] = self._data[index_list, :]
        colors[self._vertex_begin:self._vertex_end, :] = numpy.tile(self._colors, (1, 2)).reshape((-1, 4))[needed_points_list.ravel()]
        line_dimensions[self._vertex_begin:self._vertex_end, 0] = numpy.tile(self._line_widths, (1, 2)).reshape((-1, 1))[needed_points_list.ravel()][:, 0]
        line_dimensions[self._vertex_begin:self._vertex_end, 1] = numpy.tile(self._line_thicknesses, (1, 2)).reshape((-1, 1))[needed_points_list.ravel()][:, 0]
        feedrates[self._vertex_begin:self._vertex_end] = numpy.tile(self._line_feedrates, (1, 2)).reshape((-1, 1))[needed_points_list.ravel()][:, 0]
        extruders[self._vertex_begin:self._vertex_end] = self._extruder
        line_types[self._vertex_begin:self._vertex_end] = numpy.tile(self._types, (1, 2)).reshape((-1, 1))[needed_points_list.ravel()][:, 0]
        self._index_begin += index_offset
        self._index_end += index_offset
        indices[self._index_begin:self._index_end, :] = numpy.arange(self._index_end - self._index_begin, dtype=numpy.int32).reshape((-1, 1))
        indices[self._index_begin:self._index_end, :] += numpy.cumsum(needed_points_list[line_mesh_mask.ravel(), 0], dtype=numpy.int32).reshape((-1, 1))
        indices[self._index_begin:self._index_end, :] += numpy.array([self._vertex_begin - 1, self._vertex_begin])
        self._build_cache_line_mesh_mask = None
        self._build_cache_needed_points = None

    def getColors(self):
        if False:
            while True:
                i = 10
        return self._colors

    def mapLineTypeToColor(self, line_types: numpy.ndarray) -> numpy.ndarray:
        if False:
            i = 10
            return i + 15
        return self._color_map[line_types]

    def isInfillOrSkinType(self, line_types: numpy.ndarray) -> numpy.ndarray:
        if False:
            for i in range(10):
                print('nop')
        return self._is_infill_or_skin_type_map[line_types]

    def lineMeshVertexCount(self) -> int:
        if False:
            while True:
                i = 10
        return self._vertex_end - self._vertex_begin

    def lineMeshElementCount(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._index_end - self._index_begin

    @property
    def extruder(self):
        if False:
            return 10
        return self._extruder

    @property
    def types(self):
        if False:
            print('Hello World!')
        return self._types

    @property
    def data(self):
        if False:
            print('Hello World!')
        return self._data

    @property
    def elementCount(self):
        if False:
            return 10
        return (self._index_end - self._index_begin) * 2

    @property
    def lineWidths(self):
        if False:
            print('Hello World!')
        return self._line_widths

    @property
    def lineThicknesses(self):
        if False:
            return 10
        return self._line_thicknesses

    @property
    def lineFeedrates(self):
        if False:
            while True:
                i = 10
        return self._line_feedrates

    @property
    def jumpMask(self):
        if False:
            while True:
                i = 10
        return self._jump_mask

    @property
    def meshLineCount(self):
        if False:
            print('Hello World!')
        return self._mesh_line_count

    @property
    def jumpCount(self):
        if False:
            print('Hello World!')
        return self._jump_count

    def getNormals(self) -> numpy.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Calculate normals for the entire polygon using numpy.\n\n        :return: normals for the entire polygon\n        '
        normals = numpy.copy(self._data)
        normals[:, 1] = 0.0
        normals = numpy.diff(normals, 1, 0)
        lengths = numpy.sqrt(normals[:, 0] ** 2 + normals[:, 2] ** 2)
        normals[:, [0, 2]] = normals[:, [2, 0]]
        normals[:, 0] *= -1
        normals[:, 0] /= lengths
        normals[:, 2] /= lengths
        return normals
    __color_map = None

    @classmethod
    def getColorMap(cls) -> numpy.ndarray:
        if False:
            print('Hello World!')
        'Gets the instance of the VersionUpgradeManager, or creates one.'
        if cls.__color_map is None:
            theme = cast(Theme, QtApplication.getInstance().getTheme())
            cls.__color_map = numpy.array([theme.getColor('layerview_none').getRgbF(), theme.getColor('layerview_inset_0').getRgbF(), theme.getColor('layerview_inset_x').getRgbF(), theme.getColor('layerview_skin').getRgbF(), theme.getColor('layerview_support').getRgbF(), theme.getColor('layerview_skirt').getRgbF(), theme.getColor('layerview_infill').getRgbF(), theme.getColor('layerview_support_infill').getRgbF(), theme.getColor('layerview_move_combing').getRgbF(), theme.getColor('layerview_move_retraction').getRgbF(), theme.getColor('layerview_support_interface').getRgbF(), theme.getColor('layerview_prime_tower').getRgbF()])
        return cls.__color_map