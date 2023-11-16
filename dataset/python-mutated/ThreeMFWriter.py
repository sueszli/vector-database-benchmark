import json
import re
from typing import Optional, cast, List, Dict, Pattern, Set
from UM.Mesh.MeshWriter import MeshWriter
from UM.Math.Vector import Vector
from UM.Logger import Logger
from UM.Math.Matrix import Matrix
from UM.Application import Application
from UM.Message import Message
from UM.Resources import Resources
from UM.Scene.SceneNode import SceneNode
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.EmptyInstanceContainer import EmptyInstanceContainer
from cura.CuraApplication import CuraApplication
from cura.CuraPackageManager import CuraPackageManager
from cura.Settings import CuraContainerStack
from cura.Utils.Threading import call_on_qt_thread
from cura.Snapshot import Snapshot
from PyQt6.QtCore import QBuffer
import pySavitar as Savitar
import numpy
import datetime
MYPY = False
try:
    if not MYPY:
        import xml.etree.cElementTree as ET
except ImportError:
    Logger.log('w', 'Unable to load cElementTree, switching to slower version')
    import xml.etree.ElementTree as ET
import zipfile
import UM.Application
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')
THUMBNAIL_PATH = 'Metadata/thumbnail.png'
MODEL_PATH = '3D/3dmodel.model'
PACKAGE_METADATA_PATH = 'Cura/packages.json'

class ThreeMFWriter(MeshWriter):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._namespaces = {'3mf': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02', 'content-types': 'http://schemas.openxmlformats.org/package/2006/content-types', 'relationships': 'http://schemas.openxmlformats.org/package/2006/relationships', 'cura': 'http://software.ultimaker.com/xml/cura/3mf/2015/10'}
        self._unit_matrix_string = ThreeMFWriter._convertMatrixToString(Matrix())
        self._archive: Optional[zipfile.ZipFile] = None
        self._store_archive = False

    @staticmethod
    def _convertMatrixToString(matrix):
        if False:
            print('Hello World!')
        result = ''
        result += str(matrix._data[0, 0]) + ' '
        result += str(matrix._data[1, 0]) + ' '
        result += str(matrix._data[2, 0]) + ' '
        result += str(matrix._data[0, 1]) + ' '
        result += str(matrix._data[1, 1]) + ' '
        result += str(matrix._data[2, 1]) + ' '
        result += str(matrix._data[0, 2]) + ' '
        result += str(matrix._data[1, 2]) + ' '
        result += str(matrix._data[2, 2]) + ' '
        result += str(matrix._data[0, 3]) + ' '
        result += str(matrix._data[1, 3]) + ' '
        result += str(matrix._data[2, 3])
        return result

    def setStoreArchive(self, store_archive):
        if False:
            print('Hello World!')
        'Should we store the archive\n\n        Note that if this is true, the archive will not be closed.\n        The object that set this parameter is then responsible for closing it correctly!\n        '
        self._store_archive = store_archive

    @staticmethod
    def _convertUMNodeToSavitarNode(um_node, transformation=Matrix()):
        if False:
            print('Hello World!')
        'Convenience function that converts an Uranium SceneNode object to a SavitarSceneNode\n\n        :returns: Uranium Scene node.\n        '
        if not isinstance(um_node, SceneNode):
            return None
        active_build_plate_nr = CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate
        if um_node.callDecoration('getBuildPlateNumber') != active_build_plate_nr:
            return
        savitar_node = Savitar.SceneNode()
        savitar_node.setName(um_node.getName())
        node_matrix = Matrix()
        mesh_data = um_node.getMeshData()
        if mesh_data is not None:
            extents = mesh_data.getExtents()
            if extents is not None:
                center_vector = Vector(extents.center.x, extents.center.z, extents.center.y)
                node_matrix.setByTranslation(center_vector)
        node_matrix.multiply(um_node.getLocalTransformation())
        matrix_string = ThreeMFWriter._convertMatrixToString(node_matrix.preMultiply(transformation))
        savitar_node.setTransformation(matrix_string)
        if mesh_data is not None:
            savitar_node.getMeshData().setVerticesFromBytes(mesh_data.getVerticesAsByteArray())
            indices_array = mesh_data.getIndicesAsByteArray()
            if indices_array is not None:
                savitar_node.getMeshData().setFacesFromBytes(indices_array)
            else:
                savitar_node.getMeshData().setFacesFromBytes(numpy.arange(mesh_data.getVertices().size / 3, dtype=numpy.int32).tostring())
        stack = um_node.callDecoration('getStack')
        if stack is not None:
            changed_setting_keys = stack.getTop().getAllKeys()
            if stack.getProperty('machine_extruder_count', 'value') > 1:
                changed_setting_keys.add('extruder_nr')
            for key in changed_setting_keys:
                savitar_node.setSetting('cura:' + key, str(stack.getProperty(key, 'value')))
        for (key, value) in um_node.metadata.items():
            savitar_node.setSetting(key, value)
        for child_node in um_node.getChildren():
            if child_node.callDecoration('getBuildPlateNumber') != active_build_plate_nr:
                continue
            savitar_child_node = ThreeMFWriter._convertUMNodeToSavitarNode(child_node)
            if savitar_child_node is not None:
                savitar_node.addChild(savitar_child_node)
        return savitar_node

    def getArchive(self):
        if False:
            return 10
        return self._archive

    def write(self, stream, nodes, mode=MeshWriter.OutputMode.BinaryMode) -> bool:
        if False:
            return 10
        self._archive = None
        archive = zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED)
        try:
            model_file = zipfile.ZipInfo(MODEL_PATH)
            model_file.compress_type = zipfile.ZIP_DEFLATED
            content_types_file = zipfile.ZipInfo('[Content_Types].xml')
            content_types_file.compress_type = zipfile.ZIP_DEFLATED
            content_types = ET.Element('Types', xmlns=self._namespaces['content-types'])
            rels_type = ET.SubElement(content_types, 'Default', Extension='rels', ContentType='application/vnd.openxmlformats-package.relationships+xml')
            model_type = ET.SubElement(content_types, 'Default', Extension='model', ContentType='application/vnd.ms-package.3dmanufacturing-3dmodel+xml')
            relations_file = zipfile.ZipInfo('_rels/.rels')
            relations_file.compress_type = zipfile.ZIP_DEFLATED
            relations_element = ET.Element('Relationships', xmlns=self._namespaces['relationships'])
            model_relation_element = ET.SubElement(relations_element, 'Relationship', Target='/' + MODEL_PATH, Id='rel0', Type='http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel')
            snapshot = self._createSnapshot()
            if snapshot:
                thumbnail_buffer = QBuffer()
                thumbnail_buffer.open(QBuffer.OpenModeFlag.ReadWrite)
                snapshot.save(thumbnail_buffer, 'PNG')
                thumbnail_file = zipfile.ZipInfo(THUMBNAIL_PATH)
                archive.writestr(thumbnail_file, thumbnail_buffer.data())
                thumbnail_type = ET.SubElement(content_types, 'Default', Extension='png', ContentType='image/png')
                thumbnail_relation_element = ET.SubElement(relations_element, 'Relationship', Target='/' + THUMBNAIL_PATH, Id='rel1', Type='http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail')
            packages_metadata = self._getMaterialPackageMetadata() + self._getPluginPackageMetadata()
            self._storeMetadataJson({'packages': packages_metadata}, archive, PACKAGE_METADATA_PATH)
            savitar_scene = Savitar.Scene()
            scene_metadata = CuraApplication.getInstance().getController().getScene().getMetaData()
            for (key, value) in scene_metadata.items():
                savitar_scene.setMetaDataEntry(key, value)
            current_time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if 'Application' not in scene_metadata:
                savitar_scene.setMetaDataEntry('Application', CuraApplication.getInstance().getApplicationDisplayName())
            if 'CreationDate' not in scene_metadata:
                savitar_scene.setMetaDataEntry('CreationDate', current_time_string)
            savitar_scene.setMetaDataEntry('ModificationDate', current_time_string)
            transformation_matrix = Matrix()
            transformation_matrix._data[1, 1] = 0
            transformation_matrix._data[1, 2] = -1
            transformation_matrix._data[2, 1] = 1
            transformation_matrix._data[2, 2] = 0
            global_container_stack = Application.getInstance().getGlobalContainerStack()
            if global_container_stack:
                translation_vector = Vector(x=global_container_stack.getProperty('machine_width', 'value') / 2, y=global_container_stack.getProperty('machine_depth', 'value') / 2, z=0)
                translation_matrix = Matrix()
                translation_matrix.setByTranslation(translation_vector)
                transformation_matrix.preMultiply(translation_matrix)
            root_node = UM.Application.Application.getInstance().getController().getScene().getRoot()
            for node in nodes:
                if node == root_node:
                    for root_child in node.getChildren():
                        savitar_node = ThreeMFWriter._convertUMNodeToSavitarNode(root_child, transformation_matrix)
                        if savitar_node:
                            savitar_scene.addSceneNode(savitar_node)
                else:
                    savitar_node = self._convertUMNodeToSavitarNode(node, transformation_matrix)
                    if savitar_node:
                        savitar_scene.addSceneNode(savitar_node)
            parser = Savitar.ThreeMFParser()
            scene_string = parser.sceneToString(savitar_scene)
            archive.writestr(model_file, scene_string)
            archive.writestr(content_types_file, b'<?xml version="1.0" encoding="UTF-8"?> \n' + ET.tostring(content_types))
            archive.writestr(relations_file, b'<?xml version="1.0" encoding="UTF-8"?> \n' + ET.tostring(relations_element))
        except Exception as error:
            Logger.logException('e', 'Error writing zip file')
            self.setInformation(str(error))
            return False
        finally:
            if not self._store_archive:
                archive.close()
            else:
                self._archive = archive
        return True

    @staticmethod
    def _storeMetadataJson(metadata: Dict[str, List[Dict[str, str]]], archive: zipfile.ZipFile, path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Stores metadata inside archive path as json file'
        metadata_file = zipfile.ZipInfo(path)
        metadata_file.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(metadata_file, json.dumps(metadata, separators=(', ', ': '), indent=4, skipkeys=True, ensure_ascii=False))

    @staticmethod
    def _getPluginPackageMetadata() -> List[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        'Get metadata for all backend plugins that are used in the project.\n\n        :return: List of material metadata dictionaries.\n        '
        backend_plugin_enum_value_regex = re.compile('PLUGIN::(?P<plugin_id>\\w+)@(?P<version>\\d+.\\d+.\\d+)::(?P<value>\\w+)')
        plugin_ids = set()

        def addPluginIdsInStack(stack: CuraContainerStack) -> None:
            if False:
                while True:
                    i = 10
            for key in stack.getAllKeys():
                value = str(stack.getProperty(key, 'value'))
                for (plugin_id, _version, _value) in backend_plugin_enum_value_regex.findall(value):
                    plugin_ids.add(plugin_id)
        global_stack = CuraApplication.getInstance().getMachineManager().activeMachine
        addPluginIdsInStack(global_stack)
        for container in global_stack.getContainers():
            addPluginIdsInStack(container)
        for extruder_stack in global_stack.extruderList:
            addPluginIdsInStack(extruder_stack)
            for container in extruder_stack.getContainers():
                addPluginIdsInStack(container)
        metadata = {}
        package_manager = cast(CuraPackageManager, CuraApplication.getInstance().getPackageManager())
        for plugin_id in plugin_ids:
            package_data = package_manager.getInstalledPackageInfo(plugin_id)
            metadata[plugin_id] = {'id': plugin_id, 'display_name': package_data.get('display_name') if package_data.get('display_name') else '', 'package_version': package_data.get('package_version') if package_data.get('package_version') else '', 'sdk_version_semver': package_data.get('sdk_version_semver') if package_data.get('sdk_version_semver') else '', 'type': 'plugin'}
        return list(metadata.values())

    @staticmethod
    def _getMaterialPackageMetadata() -> List[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        'Get metadata for installed materials in active extruder stack, this does not include bundled materials.\n\n        :return: List of material metadata dictionaries.\n        '
        metadata = {}
        package_manager = cast(CuraPackageManager, CuraApplication.getInstance().getPackageManager())
        for extruder in CuraApplication.getInstance().getExtruderManager().getActiveExtruderStacks():
            if not extruder.isEnabled:
                continue
            if isinstance(extruder.material, type(ContainerRegistry.getInstance().getEmptyInstanceContainer())):
                continue
            if package_manager.isMaterialBundled(extruder.material.getFileName(), extruder.material.getMetaDataEntry('GUID')):
                continue
            package_id = package_manager.getMaterialFilePackageId(extruder.material.getFileName(), extruder.material.getMetaDataEntry('GUID'))
            package_data = package_manager.getInstalledPackageInfo(package_id)
            if not package_data:
                Logger.info(f'Could not find package for material in extruder {extruder.id}, skipping.')
                continue
            material_metadata = {'id': package_id, 'display_name': package_data.get('display_name') if package_data.get('display_name') else '', 'package_version': package_data.get('package_version') if package_data.get('package_version') else '', 'sdk_version_semver': package_data.get('sdk_version_semver') if package_data.get('sdk_version_semver') else '', 'type': 'material'}
            metadata[package_id] = material_metadata
        return list(metadata.values())

    @call_on_qt_thread
    def _createSnapshot(self):
        if False:
            while True:
                i = 10
        Logger.log('d', 'Creating thumbnail image...')
        if not CuraApplication.getInstance().isVisible:
            Logger.log('w', "Can't create snapshot when renderer not initialized.")
            return None
        try:
            snapshot = Snapshot.snapshot(width=300, height=300)
        except:
            Logger.logException('w', 'Failed to create snapshot image')
            return None
        return snapshot

    @staticmethod
    def sceneNodesToString(scene_nodes: [SceneNode]) -> str:
        if False:
            print('Hello World!')
        savitar_scene = Savitar.Scene()
        for scene_node in scene_nodes:
            savitar_node = ThreeMFWriter._convertUMNodeToSavitarNode(scene_node)
            savitar_scene.addSceneNode(savitar_node)
        parser = Savitar.ThreeMFParser()
        scene_string = parser.sceneToString(savitar_scene)
        return scene_string