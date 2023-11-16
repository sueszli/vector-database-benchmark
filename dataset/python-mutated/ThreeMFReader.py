import os.path
import zipfile
from typing import List, Optional, Union, TYPE_CHECKING, cast
import pySavitar as Savitar
import numpy
from UM.Logger import Logger
from UM.Math.Matrix import Matrix
from UM.Math.Vector import Vector
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.Mesh.MeshReader import MeshReader
from UM.MimeTypeDatabase import MimeTypeDatabase, MimeType
from UM.Scene.GroupDecorator import GroupDecorator
from UM.Scene.SceneNode import SceneNode
from cura.CuraApplication import CuraApplication
from cura.Machines.ContainerTree import ContainerTree
from cura.Scene.BuildPlateDecorator import BuildPlateDecorator
from cura.Scene.ConvexHullDecorator import ConvexHullDecorator
from cura.Scene.CuraSceneNode import CuraSceneNode
from cura.Scene.SliceableObjectDecorator import SliceableObjectDecorator
from cura.Scene.ZOffsetDecorator import ZOffsetDecorator
from cura.Settings.ExtruderManager import ExtruderManager
try:
    if not TYPE_CHECKING:
        import xml.etree.cElementTree as ET
except ImportError:
    Logger.log('w', 'Unable to load cElementTree, switching to slower version')
    import xml.etree.ElementTree as ET

class ThreeMFReader(MeshReader):
    """Base implementation for reading 3MF files. Has no support for textures. Only loads meshes!"""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        MimeTypeDatabase.addMimeType(MimeType(name='application/vnd.ms-package.3dmanufacturing-3dmodel+xml', comment='3MF', suffixes=['3mf']))
        self._supported_extensions = ['.3mf']
        self._root = None
        self._base_name = ''
        self._unit = None
        self._empty_project = False

    def emptyFileHintSet(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._empty_project

    @staticmethod
    def _createMatrixFromTransformationString(transformation: str) -> Matrix:
        if False:
            print('Hello World!')
        if transformation == '':
            return Matrix()
        split_transformation = transformation.split()
        temp_mat = Matrix()
        'Transformation is saved as:\n            M00 M01 M02 0.0\n\n            M10 M11 M12 0.0\n\n            M20 M21 M22 0.0\n\n            M30 M31 M32 1.0\n        We switch the row & cols as that is how everyone else uses matrices!\n        '
        temp_mat._data[0, 0] = split_transformation[0]
        temp_mat._data[1, 0] = split_transformation[1]
        temp_mat._data[2, 0] = split_transformation[2]
        temp_mat._data[0, 1] = split_transformation[3]
        temp_mat._data[1, 1] = split_transformation[4]
        temp_mat._data[2, 1] = split_transformation[5]
        temp_mat._data[0, 2] = split_transformation[6]
        temp_mat._data[1, 2] = split_transformation[7]
        temp_mat._data[2, 2] = split_transformation[8]
        temp_mat._data[0, 3] = split_transformation[9]
        temp_mat._data[1, 3] = split_transformation[10]
        temp_mat._data[2, 3] = split_transformation[11]
        return temp_mat

    @staticmethod
    def _convertSavitarNodeToUMNode(savitar_node: Savitar.SceneNode, file_name: str='') -> Optional[SceneNode]:
        if False:
            print('Hello World!')
        'Convenience function that converts a SceneNode object (as obtained from libSavitar) to a scene node.\n\n        :returns: Scene node.\n        '
        try:
            node_name = savitar_node.getName()
            node_id = savitar_node.getId()
        except AttributeError:
            Logger.log('e', 'Outdated version of libSavitar detected! Please update to the newest version!')
            node_name = ''
            node_id = ''
        if node_name == '':
            if file_name != '':
                node_name = os.path.basename(file_name)
            else:
                node_name = 'Object {}'.format(node_id)
        active_build_plate = CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate
        um_node = CuraSceneNode()
        um_node.addDecorator(BuildPlateDecorator(active_build_plate))
        try:
            um_node.addDecorator(ConvexHullDecorator())
        except:
            pass
        um_node.setName(node_name)
        um_node.setId(node_id)
        transformation = ThreeMFReader._createMatrixFromTransformationString(savitar_node.getTransformation())
        um_node.setTransformation(transformation)
        mesh_builder = MeshBuilder()
        data = numpy.fromstring(savitar_node.getMeshData().getFlatVerticesAsBytes(), dtype=numpy.float32)
        vertices = numpy.resize(data, (int(data.size / 3), 3))
        mesh_builder.setVertices(vertices)
        mesh_builder.calculateNormals(fast=True)
        if file_name:
            mesh_builder.setFileName(file_name)
        mesh_data = mesh_builder.build()
        if len(mesh_data.getVertices()):
            um_node.setMeshData(mesh_data)
        for child in savitar_node.getChildren():
            child_node = ThreeMFReader._convertSavitarNodeToUMNode(child)
            if child_node:
                um_node.addChild(child_node)
        if um_node.getMeshData() is None and len(um_node.getChildren()) == 0:
            return None
        settings = savitar_node.getSettings()
        if settings:
            global_container_stack = CuraApplication.getInstance().getGlobalContainerStack()
            if global_container_stack:
                default_stack = ExtruderManager.getInstance().getExtruderStack(0)
                if default_stack:
                    um_node.callDecoration('setActiveExtruder', default_stack.getId())
                definition_id = ContainerTree.getInstance().machines[global_container_stack.definition.getId()].quality_definition
                um_node.callDecoration('getStack').getTop().setDefinition(definition_id)
            setting_container = um_node.callDecoration('getStack').getTop()
            known_setting_keys = um_node.callDecoration('getStack').getAllKeys()
            for key in settings:
                setting_value = settings[key].value
                if key == 'extruder_nr':
                    extruder_stack = ExtruderManager.getInstance().getExtruderStack(int(setting_value))
                    if extruder_stack:
                        um_node.callDecoration('setActiveExtruder', extruder_stack.getId())
                    else:
                        Logger.log('w', 'Unable to find extruder in position %s', setting_value)
                    continue
                if key in known_setting_keys:
                    setting_container.setProperty(key, 'value', setting_value)
                else:
                    um_node.metadata[key] = settings[key]
        if len(um_node.getChildren()) > 0 and um_node.getMeshData() is None:
            if len(um_node.getAllChildren()) == 1:
                child_node = um_node.getChildren()[0]
                if child_node.getMeshData():
                    extents = child_node.getMeshData().getExtents()
                    move_matrix = Matrix()
                    move_matrix.translate(-extents.center)
                    child_node.setMeshData(child_node.getMeshData().getTransformed(move_matrix))
                    child_node.translate(extents.center)
                parent_transformation = um_node.getLocalTransformation()
                child_transformation = child_node.getLocalTransformation()
                child_node.setTransformation(parent_transformation.multiply(child_transformation))
                um_node = cast(CuraSceneNode, um_node.getChildren()[0])
            else:
                group_decorator = GroupDecorator()
                um_node.addDecorator(group_decorator)
        um_node.setSelectable(True)
        if um_node.getMeshData():
            sliceable_decorator = SliceableObjectDecorator()
            um_node.addDecorator(sliceable_decorator)
        return um_node

    def _read(self, file_name: str) -> Union[SceneNode, List[SceneNode]]:
        if False:
            print('Hello World!')
        self._empty_project = False
        result = []
        try:
            archive = zipfile.ZipFile(file_name, 'r')
            self._base_name = os.path.basename(file_name)
            parser = Savitar.ThreeMFParser()
            scene_3mf = parser.parse(archive.open('3D/3dmodel.model').read())
            self._unit = scene_3mf.getUnit()
            for (key, value) in scene_3mf.getMetadata().items():
                CuraApplication.getInstance().getController().getScene().setMetaDataEntry(key, value)
            for node in scene_3mf.getSceneNodes():
                um_node = ThreeMFReader._convertSavitarNodeToUMNode(node, file_name)
                if um_node is None:
                    continue
                transform_matrix = Matrix()
                mesh_data = um_node.getMeshData()
                if mesh_data is not None:
                    extents = mesh_data.getExtents()
                    if extents is not None:
                        center_vector = Vector(extents.center.x, extents.center.y, extents.center.z)
                        transform_matrix.setByTranslation(center_vector)
                transform_matrix.multiply(um_node.getLocalTransformation())
                um_node.setTransformation(transform_matrix)
                global_container_stack = CuraApplication.getInstance().getGlobalContainerStack()
                transformation_matrix = Matrix()
                transformation_matrix._data[1, 1] = 0
                transformation_matrix._data[1, 2] = 1
                transformation_matrix._data[2, 1] = -1
                transformation_matrix._data[2, 2] = 0
                if global_container_stack:
                    translation_vector = Vector(x=-global_container_stack.getProperty('machine_width', 'value') / 2, y=-global_container_stack.getProperty('machine_depth', 'value') / 2, z=0)
                    translation_matrix = Matrix()
                    translation_matrix.setByTranslation(translation_vector)
                    transformation_matrix.multiply(translation_matrix)
                scale_matrix = Matrix()
                scale_matrix.setByScaleVector(self._getScaleFromUnit(self._unit))
                transformation_matrix.multiply(scale_matrix)
                um_node.setTransformation(um_node.getLocalTransformation().preMultiply(transformation_matrix))
                node_meshdata = um_node.getMeshData()
                if node_meshdata is not None:
                    aabb = node_meshdata.getExtents(um_node.getWorldTransformation())
                    if aabb is not None:
                        minimum_z_value = aabb.minimum.y
                        if minimum_z_value < 0:
                            um_node.addDecorator(ZOffsetDecorator())
                            um_node.callDecoration('setZOffset', minimum_z_value)
                result.append(um_node)
            if len(result) == 0:
                self._empty_project = True
        except Exception:
            Logger.logException('e', 'An exception occurred in 3mf reader.')
            return []
        return result

    def _getScaleFromUnit(self, unit: Optional[str]) -> Vector:
        if False:
            for i in range(10):
                print('nop')
        'Create a scale vector based on a unit string.\n\n        .. The core spec defines the following:\n        * micron\n        * millimeter (default)\n        * centimeter\n        * inch\n        * foot\n        * meter\n        '
        conversion_to_mm = {'micron': 0.001, 'millimeter': 1, 'centimeter': 10, 'meter': 1000, 'inch': 25.4, 'foot': 304.8}
        if unit is None:
            unit = 'millimeter'
        elif unit not in conversion_to_mm:
            Logger.log('w', 'Unrecognised unit {unit} used. Assuming mm instead.'.format(unit=unit))
            unit = 'millimeter'
        scale = conversion_to_mm[unit]
        return Vector(scale, scale, scale)

    @staticmethod
    def stringToSceneNodes(scene_string: str) -> List[SceneNode]:
        if False:
            i = 10
            return i + 15
        parser = Savitar.ThreeMFParser()
        scene = parser.parse(scene_string)
        nodes = []
        for savitar_node in scene.getSceneNodes():
            scene_node = ThreeMFReader._convertSavitarNodeToUMNode(savitar_node, 'file_name')
            if scene_node is None:
                continue
            nodes.append(scene_node)
        return nodes