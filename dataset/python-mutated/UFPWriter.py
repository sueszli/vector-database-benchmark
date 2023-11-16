import json
from dataclasses import asdict
from typing import cast, List, Dict
from Charon.VirtualFile import VirtualFile
from Charon.OpenMode import OpenMode
from Charon.filetypes.OpenPackagingConvention import OPCError
from io import StringIO
from PyQt6.QtCore import QBuffer
from UM.Application import Application
from UM.Logger import Logger
from UM.Settings.SettingFunction import SettingFunction
from UM.Mesh.MeshWriter import MeshWriter
from UM.MimeTypeDatabase import MimeTypeDatabase, MimeType
from UM.PluginRegistry import PluginRegistry
from UM.Scene.Iterator.DepthFirstIterator import DepthFirstIterator
from UM.Scene.SceneNode import SceneNode
from UM.Settings.InstanceContainer import InstanceContainer
from cura.CuraApplication import CuraApplication
from cura.Settings.CuraStackBuilder import CuraStackBuilder
from cura.Settings.GlobalStack import GlobalStack
from cura.Utils.Threading import call_on_qt_thread
from UM.i18n import i18nCatalog
METADATA_OBJECTS_PATH = 'metadata/objects'
SLICE_METADATA_PATH = 'Cura/slicemetadata.json'
catalog = i18nCatalog('cura')

class UFPWriter(MeshWriter):

    def __init__(self):
        if False:
            return 10
        super().__init__(add_to_recent_files=False)
        MimeTypeDatabase.addMimeType(MimeType(name='application/x-ufp', comment='UltiMaker Format Package', suffixes=['ufp']))

    @call_on_qt_thread
    def write(self, stream, nodes, mode=MeshWriter.OutputMode.BinaryMode):
        if False:
            print('Hello World!')
        archive = VirtualFile()
        archive.openStream(stream, 'application/x-ufp', OpenMode.WriteOnly)
        try:
            self._writeObjectList(archive)
            archive.addContentType(extension='gcode', mime_type='text/x-gcode')
        except EnvironmentError as e:
            error_msg = catalog.i18nc('@info:error', "Can't write to UFP file:") + ' ' + str(e)
            self.setInformation(error_msg)
            Logger.error(error_msg)
            return False
        gcode_textio = StringIO()
        gcode_writer = cast(MeshWriter, PluginRegistry.getInstance().getPluginObject('GCodeWriter'))
        success = gcode_writer.write(gcode_textio, None)
        if not success:
            self.setInformation(gcode_writer.getInformation())
            return False
        try:
            gcode = archive.getStream('/3D/model.gcode')
            gcode.write(gcode_textio.getvalue().encode('UTF-8'))
            archive.addRelation(virtual_path='/3D/model.gcode', relation_type='http://schemas.ultimaker.org/package/2018/relationships/gcode')
        except EnvironmentError as e:
            error_msg = catalog.i18nc('@info:error', "Can't write to UFP file:") + ' ' + str(e)
            self.setInformation(error_msg)
            Logger.error(error_msg)
            return False
        try:
            archive.addContentType(extension='json', mime_type='application/json')
            setting_textio = StringIO()
            json.dump(self._getSliceMetadata(), setting_textio, separators=(', ', ': '), indent=4)
            steam = archive.getStream(SLICE_METADATA_PATH)
            steam.write(setting_textio.getvalue().encode('UTF-8'))
        except EnvironmentError as e:
            error_msg = catalog.i18nc('@info:error', "Can't write to UFP file:") + ' ' + str(e)
            self.setInformation(error_msg)
            Logger.error(error_msg)
            return False
        backend = CuraApplication.getInstance().getBackend()
        snapshot = None if getattr(backend, 'getLatestSnapshot', None) is None else backend.getLatestSnapshot()
        if snapshot:
            try:
                archive.addContentType(extension='png', mime_type='image/png')
                thumbnail = archive.getStream('/Metadata/thumbnail.png')
                thumbnail_buffer = QBuffer()
                thumbnail_buffer.open(QBuffer.OpenModeFlag.ReadWrite)
                snapshot.save(thumbnail_buffer, 'PNG')
                thumbnail.write(thumbnail_buffer.data())
                archive.addRelation(virtual_path='/Metadata/thumbnail.png', relation_type='http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail', origin='/3D/model.gcode')
            except EnvironmentError as e:
                error_msg = catalog.i18nc('@info:error', "Can't write to UFP file:") + ' ' + str(e)
                self.setInformation(error_msg)
                Logger.error(error_msg)
                return False
        else:
            Logger.log('w', 'Thumbnail not created, cannot save it')
        application = CuraApplication.getInstance()
        machine_manager = application.getMachineManager()
        container_registry = application.getContainerRegistry()
        global_stack = machine_manager.activeMachine
        material_extension = 'xml.fdm_material'
        material_mime_type = 'application/x-ultimaker-material-profile'
        try:
            archive.addContentType(extension=material_extension, mime_type=material_mime_type)
        except OPCError:
            Logger.log('w', 'The material extension: %s was already added', material_extension)
        added_materials = []
        for extruder_stack in global_stack.extruderList:
            material = extruder_stack.material
            try:
                material_file_name = material.getMetaData()['base_file'] + '.xml.fdm_material'
            except KeyError:
                Logger.log('w', 'Unable to get base_file for the material %s', material.getId())
                continue
            material_file_name = '/Materials/' + material_file_name
            if material_file_name in added_materials:
                continue
            material_root_id = material.getMetaDataEntry('base_file')
            material_root_query = container_registry.findContainers(id=material_root_id)
            if not material_root_query:
                Logger.log('e', 'Cannot find material container with root id {root_id}'.format(root_id=material_root_id))
                return False
            material_container = material_root_query[0]
            try:
                serialized_material = material_container.serialize()
            except NotImplementedError:
                Logger.log('e', 'Unable serialize material container with root id: %s', material_root_id)
                return False
            try:
                material_file = archive.getStream(material_file_name)
                material_file.write(serialized_material.encode('UTF-8'))
                archive.addRelation(virtual_path=material_file_name, relation_type='http://schemas.ultimaker.org/package/2018/relationships/material', origin='/3D/model.gcode')
            except EnvironmentError as e:
                error_msg = catalog.i18nc('@info:error', "Can't write to UFP file:") + ' ' + str(e)
                self.setInformation(error_msg)
                Logger.error(error_msg)
                return False
            added_materials.append(material_file_name)
        try:
            archive.close()
        except EnvironmentError as e:
            error_msg = catalog.i18nc('@info:error', "Can't write to UFP file:") + ' ' + str(e)
            self.setInformation(error_msg)
            Logger.error(error_msg)
            return False
        return True

    @staticmethod
    def _writeObjectList(archive):
        if False:
            for i in range(10):
                print('nop')
        'Write a json list of object names to the METADATA_OBJECTS_PATH metadata field\n\n        To retrieve, use: `archive.getMetadata(METADATA_OBJECTS_PATH)`\n        '
        objects_model = CuraApplication.getInstance().getObjectsModel()
        object_metas = []
        for item in objects_model.items:
            object_metas.extend(UFPWriter._getObjectMetadata(item['node']))
        data = {METADATA_OBJECTS_PATH: object_metas}
        archive.setMetadata(data)

    @staticmethod
    def _getObjectMetadata(node: SceneNode) -> List[Dict[str, str]]:
        if False:
            i = 10
            return i + 15
        'Get object metadata to write for a Node.\n\n        :return: List of object metadata dictionaries.\n                 Might contain > 1 element in case of a group node.\n                 Might be empty in case of nonPrintingMesh\n        '
        return [{'name': item.getName()} for item in DepthFirstIterator(node) if item.getMeshData() is not None and (not item.callDecoration('isNonPrintingMesh'))]

    def _getSliceMetadata(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        if False:
            for i in range(10):
                print('nop')
        'Get all changed settings and all settings. For each extruder and the global stack'
        print_information = CuraApplication.getInstance().getPrintInformation()
        machine_manager = CuraApplication.getInstance().getMachineManager()
        settings = {'material': {'length': print_information.materialLengths, 'weight': print_information.materialWeights, 'cost': print_information.materialCosts}, 'global': {'changes': {}, 'all_settings': {}}, 'quality': asdict(machine_manager.activeQualityDisplayNameMap())}

        def _retrieveValue(container: InstanceContainer, setting_: str):
            if False:
                return 10
            value_ = container.getProperty(setting_, 'value')
            for _ in range(0, 1024):
                if not isinstance(value_, SettingFunction):
                    return value_
                value_ = value_(container)
            return 0
        global_stack = cast(GlobalStack, Application.getInstance().getGlobalContainerStack())
        global_flattened_changes = InstanceContainer.createMergedInstanceContainer(global_stack.userChanges, global_stack.qualityChanges)
        for setting in global_flattened_changes.getAllKeys():
            settings['global']['changes'][setting] = _retrieveValue(global_flattened_changes, setting)
        for setting in global_stack.getAllKeys():
            settings['global']['all_settings'][setting] = _retrieveValue(global_stack, setting)
        for (i, extruder) in enumerate(global_stack.extruderList):
            settings[f'extruder_{i}'] = {'changes': {}, 'all_settings': {}}
            extruder_flattened_changes = InstanceContainer.createMergedInstanceContainer(extruder.userChanges, extruder.qualityChanges)
            for setting in extruder_flattened_changes.getAllKeys():
                settings[f'extruder_{i}']['changes'][setting] = _retrieveValue(extruder_flattened_changes, setting)
            for setting in extruder.getAllKeys():
                settings[f'extruder_{i}']['all_settings'][setting] = _retrieveValue(extruder, setting)
        return settings