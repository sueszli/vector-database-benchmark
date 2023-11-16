from PyQt6.QtCore import QObject, pyqtProperty, pyqtSignal, pyqtSlot
from typing import Any, Dict, List, Set, Tuple, TYPE_CHECKING
from UM.Logger import Logger
from UM.Settings.InstanceContainer import InstanceContainer
import cura.CuraApplication
from UM.Signal import Signal
from cura.Machines.ContainerTree import ContainerTree
from cura.Settings.cura_empty_instance_containers import empty_intent_container
if TYPE_CHECKING:
    from UM.Settings.InstanceContainer import InstanceContainer

class IntentManager(QObject):
    """Front-end for querying which intents are available for a certain configuration.
    """
    __instance = None

    @classmethod
    def getInstance(cls):
        if False:
            for i in range(10):
                print('nop')
        'This class is a singleton.'
        if not cls.__instance:
            cls.__instance = IntentManager()
        return cls.__instance
    intentCategoryChanged = pyqtSignal()
    intentCategoryChangedSignal = Signal()

    def intentMetadatas(self, definition_id: str, nozzle_name: str, material_base_file: str) -> List[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Gets the metadata dictionaries of all intent profiles for a given\n\n        configuration.\n\n        :param definition_id: ID of the printer.\n        :param nozzle_name: Name of the nozzle.\n        :param material_base_file: The base_file of the material.\n        :return: A list of metadata dictionaries matching the search criteria, or\n            an empty list if nothing was found.\n        '
        intent_metadatas = []
        try:
            materials = ContainerTree.getInstance().machines[definition_id].variants[nozzle_name].materials
        except KeyError:
            Logger.log('w', 'Unable to find the machine %s or the variant %s', definition_id, nozzle_name)
            materials = {}
        if material_base_file not in materials:
            return intent_metadatas
        material_node = materials[material_base_file]
        for quality_node in material_node.qualities.values():
            for intent_node in quality_node.intents.values():
                intent_metadatas.append(intent_node.getMetadata())
        return intent_metadatas

    def intentCategories(self, definition_id: str, nozzle_id: str, material_id: str) -> List[str]:
        if False:
            print('Hello World!')
        "Collects and returns all intent categories available for the given\n\n        parameters. Note that the 'default' category is always available.\n\n        :param definition_id: ID of the printer.\n        :param nozzle_name: Name of the nozzle.\n        :param material_id: ID of the material.\n        :return: A set of intent category names.\n        "
        categories = set()
        for intent in self.intentMetadatas(definition_id, nozzle_id, material_id):
            categories.add(intent['intent_category'])
        categories.add('default')
        return list(categories)

    def getCurrentAvailableIntents(self) -> List[Tuple[str, str]]:
        if False:
            print('Hello World!')
        'List of intents to be displayed in the interface.\n\n        For the interface this will have to be broken up into the different\n        intent categories. That is up to the model there.\n\n        :return: A list of tuples of intent_category and quality_type. The actual\n            instance may vary per extruder.\n        '
        application = cura.CuraApplication.CuraApplication.getInstance()
        global_stack = application.getGlobalContainerStack()
        if global_stack is None:
            return [('default', 'normal')]
        quality_groups = ContainerTree.getInstance().getCurrentQualityGroups()
        available_quality_types = {quality_group.quality_type for quality_group in quality_groups.values() if quality_group.node_for_global is not None}
        final_intent_ids = set()
        current_definition_id = global_stack.definition.getId()
        for extruder_stack in global_stack.extruderList:
            if not extruder_stack.isEnabled:
                continue
            nozzle_name = extruder_stack.variant.getMetaDataEntry('name')
            material_id = extruder_stack.material.getMetaDataEntry('base_file')
            final_intent_ids |= {metadata['id'] for metadata in self.intentMetadatas(current_definition_id, nozzle_name, material_id) if metadata.get('quality_type') in available_quality_types}
        result = set()
        for intent_id in final_intent_ids:
            intent_metadata = application.getContainerRegistry().findContainersMetadata(id=intent_id)[0]
            result.add((intent_metadata['intent_category'], intent_metadata['quality_type']))
        return list(result)

    def currentAvailableIntentCategories(self) -> List[str]:
        if False:
            return 10
        'List of intent categories available in either of the extruders.\n\n        This is purposefully inconsistent with the way that the quality types\n        are listed. The quality types will show all quality types available in\n        the printer using any configuration. This will only list the intent\n        categories that are available using the current configuration (but the\n        union over the extruders).\n        :return: List of all categories in the current configurations of all\n            extruders.\n        '
        global_stack = cura.CuraApplication.CuraApplication.getInstance().getGlobalContainerStack()
        if global_stack is None:
            return ['default']
        current_definition_id = global_stack.definition.getId()
        final_intent_categories = set()
        for extruder_stack in global_stack.extruderList:
            if not extruder_stack.isEnabled:
                continue
            nozzle_name = extruder_stack.variant.getMetaDataEntry('name')
            material_id = extruder_stack.material.getMetaDataEntry('base_file')
            final_intent_categories.update(self.intentCategories(current_definition_id, nozzle_name, material_id))
        return list(final_intent_categories)

    def getDefaultIntent(self) -> 'InstanceContainer':
        if False:
            while True:
                i = 10
        "The intent that gets selected by default when no intent is available for\n\n        the configuration, an extruder can't match the intent that the user\n        selects, or just when creating a new printer.\n        "
        return empty_intent_container

    @pyqtProperty(str, notify=intentCategoryChanged)
    def currentIntentCategory(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        application = cura.CuraApplication.CuraApplication.getInstance()
        active_extruder_stack = application.getMachineManager().activeStack
        if active_extruder_stack is None:
            return ''
        return active_extruder_stack.intent.getMetaDataEntry('intent_category', '')

    @pyqtSlot(str, str)
    def selectIntent(self, intent_category: str, quality_type: str) -> None:
        if False:
            i = 10
            return i + 15
        'Apply intent on the stacks.'
        Logger.log('i', 'Attempting to set intent_category to [%s] and quality type to [%s]', intent_category, quality_type)
        old_intent_category = self.currentIntentCategory
        application = cura.CuraApplication.CuraApplication.getInstance()
        global_stack = application.getGlobalContainerStack()
        if global_stack is None:
            return
        current_definition_id = global_stack.definition.getId()
        machine_node = ContainerTree.getInstance().machines[current_definition_id]
        for extruder_stack in global_stack.extruderList:
            nozzle_name = extruder_stack.variant.getMetaDataEntry('name')
            material_id = extruder_stack.material.getMetaDataEntry('base_file')
            material_node = machine_node.variants[nozzle_name].materials[material_id]
            quality_node = None
            for q_node in material_node.qualities.values():
                if q_node.quality_type == quality_type:
                    quality_node = q_node
            if quality_node is None:
                Logger.log('w', 'Unable to find quality_type [%s] for extruder [%s]', quality_type, extruder_stack.getId())
                continue
            intent_id = None
            for (id, intent_node) in quality_node.intents.items():
                if intent_node.intent_category == intent_category:
                    intent_id = id
            intent = application.getContainerRegistry().findContainers(id=intent_id)
            if intent:
                extruder_stack.intent = intent[0]
            else:
                extruder_stack.intent = self.getDefaultIntent()
        application.getMachineManager().setQualityGroupByQualityType(quality_type)
        if old_intent_category != intent_category:
            self.intentCategoryChanged.emit()
            self.intentCategoryChangedSignal.emit()