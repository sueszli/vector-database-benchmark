import copy
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject, QUrl
from PyQt6.QtGui import QDesktopServices
from typing import Any, Dict, Optional, TYPE_CHECKING
import uuid
from UM.Message import Message
from UM.i18n import i18nCatalog
from UM.Logger import Logger
from UM.Resources import Resources
from UM.Signal import postponeSignals, CompressTechnique
import cura.CuraApplication
from cura.Machines.ContainerTree import ContainerTree
from cura.Settings.CuraContainerRegistry import CuraContainerRegistry
from cura.UltimakerCloud.CloudMaterialSync import CloudMaterialSync
if TYPE_CHECKING:
    from cura.Machines.MaterialNode import MaterialNode
catalog = i18nCatalog('cura')

class MaterialManagementModel(QObject):
    favoritesChanged = pyqtSignal(str)
    'Triggered when a favorite is added or removed.\n\n    :param The base file of the material is provided as parameter when this emits\n    '

    def __init__(self, parent: Optional[QObject]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        self._material_sync = CloudMaterialSync(parent=self)

    @pyqtSlot('QVariant', result=bool)
    def canMaterialBeRemoved(self, material_node: 'MaterialNode') -> bool:
        if False:
            i = 10
            return i + 15
        "Can a certain material be deleted, or is it still in use in one of the container stacks anywhere?\n\n        We forbid the user from deleting a material if it's in use in any stack. Deleting it while it's in use can\n        lead to corrupted stacks. In the future we might enable this functionality again (deleting the material from\n        those stacks) but for now it is easier to prevent the user from doing this.\n\n        :param material_node: The ContainerTree node of the material to check.\n\n        :return: Whether or not the material can be removed.\n        "
        container_registry = CuraContainerRegistry.getInstance()
        ids_to_remove = {metadata.get('id', '') for metadata in container_registry.findInstanceContainersMetadata(base_file=material_node.base_file)}
        for extruder_stack in container_registry.findContainerStacks(type='extruder_train'):
            if extruder_stack.material.getId() in ids_to_remove:
                return False
        return True

    @pyqtSlot('QVariant', str)
    def setMaterialName(self, material_node: 'MaterialNode', name: str) -> None:
        if False:
            while True:
                i = 10
        'Change the user-visible name of a material.\n\n        :param material_node: The ContainerTree node of the material to rename.\n        :param name: The new name for the material.\n        '
        container_registry = CuraContainerRegistry.getInstance()
        root_material_id = material_node.base_file
        if container_registry.isReadOnly(root_material_id):
            Logger.log('w', 'Cannot set name of read-only container %s.', root_material_id)
            return
        return container_registry.findContainers(id=root_material_id)[0].setName(name)

    @pyqtSlot('QVariant')
    def removeMaterial(self, material_node: 'MaterialNode') -> None:
        if False:
            while True:
                i = 10
        'Deletes a material from Cura.\n\n        This function does not do any safety checking any more. Please call this function only if:\n            - The material is not read-only.\n            - The material is not used in any stacks.\n\n        If the material was not lazy-loaded yet, this will fully load the container. When removing this material\n        node, all other materials with the same base fill will also be removed.\n\n        :param material_node: The material to remove.\n        '
        Logger.info(f'Removing material {material_node.container_id}')
        container_registry = CuraContainerRegistry.getInstance()
        materials_this_base_file = container_registry.findContainersMetadata(base_file=material_node.base_file)
        with postponeSignals(container_registry.containerRemoved, compress=CompressTechnique.CompressPerParameterValue):
            for material_metadata in materials_this_base_file:
                container_registry.findInstanceContainers(id=material_metadata['id'])
            for material_metadata in materials_this_base_file:
                container_registry.removeContainer(material_metadata['id'])

    def duplicateMaterialByBaseFile(self, base_file: str, new_base_id: Optional[str]=None, new_metadata: Optional[Dict[str, Any]]=None) -> Optional[str]:
        if False:
            return 10
        'Creates a duplicate of a material with the same GUID and base_file metadata\n\n        :param base_file: The base file of the material to duplicate.\n        :param new_base_id: A new material ID for the base material. The IDs of the submaterials will be based off this\n        one. If not provided, a material ID will be generated automatically.\n        :param new_metadata: Metadata for the new material. If not provided, this will be duplicated from the original\n        material.\n\n        :return: The root material ID of the duplicate material.\n        '
        container_registry = CuraContainerRegistry.getInstance()
        root_materials = container_registry.findContainers(id=base_file)
        if not root_materials:
            Logger.log('i', "Unable to duplicate the root material with ID {root_id}, because it doesn't exist.".format(root_id=base_file))
            return None
        root_material = root_materials[0]
        application = cura.CuraApplication.CuraApplication.getInstance()
        application.saveSettings()
        if new_base_id is None:
            new_base_id = container_registry.uniqueName(root_material.getId())
        new_root_material = copy.deepcopy(root_material)
        new_root_material.getMetaData()['id'] = new_base_id
        new_root_material.getMetaData()['base_file'] = new_base_id
        if new_metadata is not None:
            new_root_material.getMetaData().update(new_metadata)
        new_containers = [new_root_material]
        for container_to_copy in container_registry.findInstanceContainers(base_file=base_file):
            if container_to_copy.getId() == base_file:
                continue
            new_id = new_base_id
            definition = container_to_copy.getMetaDataEntry('definition')
            if definition != 'fdmprinter':
                new_id += '_' + definition
                variant_name = container_to_copy.getMetaDataEntry('variant_name')
                if variant_name:
                    new_id += '_' + variant_name.replace(' ', '_')
            new_container = copy.deepcopy(container_to_copy)
            new_container.getMetaData()['id'] = new_id
            new_container.getMetaData()['base_file'] = new_base_id
            if new_metadata is not None:
                new_container.getMetaData().update(new_metadata)
            new_containers.append(new_container)
        new_containers = sorted(new_containers, key=lambda x: x.getId(), reverse=True)
        with postponeSignals(container_registry.containerAdded, compress=CompressTechnique.CompressPerParameterValue):
            for container_to_add in new_containers:
                container_to_add.setDirty(True)
                container_registry.addContainer(container_to_add)
            favorites_set = set(application.getPreferences().getValue('cura/favorite_materials').split(';'))
            if base_file in favorites_set:
                favorites_set.add(new_base_id)
                application.getPreferences().setValue('cura/favorite_materials', ';'.join(favorites_set))
        return new_base_id

    @pyqtSlot('QVariant', result=str)
    def duplicateMaterial(self, material_node: 'MaterialNode', new_base_id: Optional[str]=None, new_metadata: Optional[Dict[str, Any]]=None) -> Optional[str]:
        if False:
            while True:
                i = 10
        'Creates a duplicate of a material with the same GUID and base_file metadata\n\n        :param material_node: The node representing the material to duplicate.\n        :param new_base_id: A new material ID for the base material. The IDs of the submaterials will be based off this\n        one. If not provided, a material ID will be generated automatically.\n        :param new_metadata: Metadata for the new material. If not provided, this will be duplicated from the original\n        material.\n\n        :return: The root material ID of the duplicate material.\n        '
        Logger.info(f'Duplicating material {material_node.base_file} to {new_base_id}')
        return self.duplicateMaterialByBaseFile(material_node.base_file, new_base_id, new_metadata)

    @pyqtSlot(result=str)
    def createMaterial(self) -> str:
        if False:
            print('Hello World!')
        'Create a new material by cloning the preferred material for the current material diameter and generate a new\n        GUID.\n\n        The material type is explicitly left to be the one from the preferred material, since this allows the user to\n        still have SOME profiles to work with.\n\n        :return: The ID of the newly created material.\n        '
        application = cura.CuraApplication.CuraApplication.getInstance()
        application.saveSettings()
        extruder_stack = application.getMachineManager().activeStack
        active_variant_name = extruder_stack.variant.getName()
        approximate_diameter = int(extruder_stack.approximateMaterialDiameter)
        global_container_stack = application.getGlobalContainerStack()
        if not global_container_stack:
            return ''
        machine_node = ContainerTree.getInstance().machines[global_container_stack.definition.getId()]
        preferred_material_node = machine_node.variants[active_variant_name].preferredMaterial(approximate_diameter)
        new_id = CuraContainerRegistry.getInstance().uniqueName('custom_material')
        new_metadata = {'name': catalog.i18nc('@label', 'Custom Material'), 'brand': catalog.i18nc('@label', 'Custom'), 'GUID': str(uuid.uuid4())}
        self.duplicateMaterial(preferred_material_node, new_base_id=new_id, new_metadata=new_metadata)
        return new_id

    @pyqtSlot(str)
    def addFavorite(self, material_base_file: str) -> None:
        if False:
            i = 10
            return i + 15
        'Adds a certain material to the favorite materials.\n\n        :param material_base_file: The base file of the material to add.\n        '
        application = cura.CuraApplication.CuraApplication.getInstance()
        favorites = application.getPreferences().getValue('cura/favorite_materials').split(';')
        if material_base_file not in favorites:
            favorites.append(material_base_file)
            application.getPreferences().setValue('cura/favorite_materials', ';'.join(favorites))
            application.saveSettings()
            self.favoritesChanged.emit(material_base_file)

    @pyqtSlot(str)
    def removeFavorite(self, material_base_file: str) -> None:
        if False:
            print('Hello World!')
        'Removes a certain material from the favorite materials.\n\n        If the material was not in the favorite materials, nothing happens.\n        '
        application = cura.CuraApplication.CuraApplication.getInstance()
        favorites = application.getPreferences().getValue('cura/favorite_materials').split(';')
        try:
            favorites.remove(material_base_file)
            application.getPreferences().setValue('cura/favorite_materials', ';'.join(favorites))
            application.saveSettings()
            self.favoritesChanged.emit(material_base_file)
        except ValueError:
            Logger.log('w', 'Material {material_base_file} was already not a favorite material.'.format(material_base_file=material_base_file))

    @pyqtSlot()
    def openSyncAllWindow(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Opens the window to sync all materials.\n        '
        self._material_sync.openSyncAllWindow()