from typing import Any, cast, List, Optional, Dict
from PyQt6.QtCore import pyqtProperty, pyqtSignal, QObject
from UM.Application import Application
from UM.Decorators import override
from UM.FlameProfiler import pyqtSlot
from UM.Logger import Logger
from UM.Settings.ContainerStack import ContainerStack, InvalidContainerStackError
from UM.Settings.InstanceContainer import InstanceContainer
from UM.Settings.DefinitionContainer import DefinitionContainer
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.Interfaces import ContainerInterface, DefinitionContainerInterface
from cura.Settings import cura_empty_instance_containers
from . import Exceptions

class CuraContainerStack(ContainerStack):
    """Base class for Cura related stacks that want to enforce certain containers are available.

    This class makes sure that the stack has the following containers set: user changes, quality
    changes, quality, material, variant, definition changes and finally definition. Initially,
    these will be equal to the empty instance container.

    The container types are determined based on the following criteria:
    - user: An InstanceContainer with the metadata entry "type" set to "user".
    - quality changes: An InstanceContainer with the metadata entry "type" set to "quality_changes".
    - quality: An InstanceContainer with the metadata entry "type" set to "quality".
    - material: An InstanceContainer with the metadata entry "type" set to "material".
    - variant: An InstanceContainer with the metadata entry "type" set to "variant".
    - definition changes: An InstanceContainer with the metadata entry "type" set to "definition_changes".
    - definition: A DefinitionContainer.

    Internally, this class ensures the mentioned containers are always there and kept in a specific order.
    This also means that operations on the stack that modifies the container ordering is prohibited and
    will raise an exception.
    """

    def __init__(self, container_id: str) -> None:
        if False:
            while True:
                i = 10
        super().__init__(container_id)
        self._empty_instance_container = cura_empty_instance_containers.empty_container
        self._empty_quality_changes = cura_empty_instance_containers.empty_quality_changes_container
        self._empty_quality = cura_empty_instance_containers.empty_quality_container
        self._empty_material = cura_empty_instance_containers.empty_material_container
        self._empty_variant = cura_empty_instance_containers.empty_variant_container
        self._containers: List[ContainerInterface] = [self._empty_instance_container for i in _ContainerIndexes.IndexTypeMap]
        self._containers[_ContainerIndexes.QualityChanges] = self._empty_quality_changes
        self._containers[_ContainerIndexes.Quality] = self._empty_quality
        self._containers[_ContainerIndexes.Material] = self._empty_material
        self._containers[_ContainerIndexes.Variant] = self._empty_variant
        self.containersChanged.connect(self._onContainersChanged)
        import cura.CuraApplication
        self.setMetaDataEntry('setting_version', cura.CuraApplication.CuraApplication.SettingVersion)
        self._settable_per_extruder_cache = {}
        self.setDirty(False)
    pyqtContainersChanged = pyqtSignal()

    def setUserChanges(self, new_user_changes: InstanceContainer) -> None:
        if False:
            while True:
                i = 10
        'Set the user changes container.\n\n        :param new_user_changes: The new user changes container. It is expected to have a "type" metadata entry with the value "user".\n        '
        self.replaceContainer(_ContainerIndexes.UserChanges, new_user_changes)

    @pyqtProperty(QObject, fset=setUserChanges, notify=pyqtContainersChanged)
    def userChanges(self) -> InstanceContainer:
        if False:
            print('Hello World!')
        'Get the user changes container.\n\n        :return: The user changes container. Should always be a valid container, but can be equal to the empty InstanceContainer.\n        '
        return cast(InstanceContainer, self._containers[_ContainerIndexes.UserChanges])

    def setQualityChanges(self, new_quality_changes: InstanceContainer, postpone_emit=False) -> None:
        if False:
            i = 10
            return i + 15
        'Set the quality changes container.\n\n        :param new_quality_changes: The new quality changes container. It is expected to have a "type" metadata entry with the value "quality_changes".\n        '
        self.replaceContainer(_ContainerIndexes.QualityChanges, new_quality_changes, postpone_emit=postpone_emit)

    @pyqtProperty(QObject, fset=setQualityChanges, notify=pyqtContainersChanged)
    def qualityChanges(self) -> InstanceContainer:
        if False:
            while True:
                i = 10
        'Get the quality changes container.\n\n        :return: The quality changes container. Should always be a valid container, but can be equal to the empty InstanceContainer.\n        '
        return cast(InstanceContainer, self._containers[_ContainerIndexes.QualityChanges])

    def setIntent(self, new_intent: InstanceContainer, postpone_emit: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the intent container.\n\n        :param new_intent: The new intent container. It is expected to have a "type" metadata entry with the value "intent".\n        '
        self.replaceContainer(_ContainerIndexes.Intent, new_intent, postpone_emit=postpone_emit)

    @pyqtProperty(QObject, fset=setIntent, notify=pyqtContainersChanged)
    def intent(self) -> InstanceContainer:
        if False:
            i = 10
            return i + 15
        'Get the quality container.\n\n        :return: The intent container. Should always be a valid container, but can be equal to the empty InstanceContainer.\n        '
        return cast(InstanceContainer, self._containers[_ContainerIndexes.Intent])

    def setQuality(self, new_quality: InstanceContainer, postpone_emit: bool=False) -> None:
        if False:
            return 10
        'Set the quality container.\n\n        :param new_quality: The new quality container. It is expected to have a "type" metadata entry with the value "quality".\n        '
        self.replaceContainer(_ContainerIndexes.Quality, new_quality, postpone_emit=postpone_emit)

    @pyqtProperty(QObject, fset=setQuality, notify=pyqtContainersChanged)
    def quality(self) -> InstanceContainer:
        if False:
            print('Hello World!')
        'Get the quality container.\n\n        :return: The quality container. Should always be a valid container, but can be equal to the empty InstanceContainer.\n        '
        return cast(InstanceContainer, self._containers[_ContainerIndexes.Quality])

    def setMaterial(self, new_material: InstanceContainer, postpone_emit: bool=False) -> None:
        if False:
            return 10
        'Set the material container.\n\n        :param new_material: The new material container. It is expected to have a "type" metadata entry with the value "material".\n        '
        self.replaceContainer(_ContainerIndexes.Material, new_material, postpone_emit=postpone_emit)

    @pyqtProperty(QObject, fset=setMaterial, notify=pyqtContainersChanged)
    def material(self) -> InstanceContainer:
        if False:
            print('Hello World!')
        'Get the material container.\n\n        :return: The material container. Should always be a valid container, but can be equal to the empty InstanceContainer.\n        '
        return cast(InstanceContainer, self._containers[_ContainerIndexes.Material])

    def setVariant(self, new_variant: InstanceContainer) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the variant container.\n\n        :param new_variant: The new variant container. It is expected to have a "type" metadata entry with the value "variant".\n        '
        self.replaceContainer(_ContainerIndexes.Variant, new_variant)

    @pyqtProperty(QObject, fset=setVariant, notify=pyqtContainersChanged)
    def variant(self) -> InstanceContainer:
        if False:
            while True:
                i = 10
        'Get the variant container.\n\n        :return: The variant container. Should always be a valid container, but can be equal to the empty InstanceContainer.\n        '
        return cast(InstanceContainer, self._containers[_ContainerIndexes.Variant])

    def setDefinitionChanges(self, new_definition_changes: InstanceContainer) -> None:
        if False:
            while True:
                i = 10
        'Set the definition changes container.\n\n        :param new_definition_changes: The new definition changes container. It is expected to have a "type" metadata entry with the value "definition_changes".\n        '
        self.replaceContainer(_ContainerIndexes.DefinitionChanges, new_definition_changes)

    @pyqtProperty(QObject, fset=setDefinitionChanges, notify=pyqtContainersChanged)
    def definitionChanges(self) -> InstanceContainer:
        if False:
            while True:
                i = 10
        'Get the definition changes container.\n\n        :return: The definition changes container. Should always be a valid container, but can be equal to the empty InstanceContainer.\n        '
        return cast(InstanceContainer, self._containers[_ContainerIndexes.DefinitionChanges])

    def setDefinition(self, new_definition: DefinitionContainerInterface) -> None:
        if False:
            i = 10
            return i + 15
        'Set the definition container.\n\n        :param new_definition: The new definition container. It is expected to have a "type" metadata entry with the value "definition".\n        '
        self.replaceContainer(_ContainerIndexes.Definition, new_definition)

    def getDefinition(self) -> 'DefinitionContainer':
        if False:
            return 10
        return cast(DefinitionContainer, self._containers[_ContainerIndexes.Definition])
    definition = pyqtProperty(QObject, fget=getDefinition, fset=setDefinition, notify=pyqtContainersChanged)

    @override(ContainerStack)
    def getBottom(self) -> 'DefinitionContainer':
        if False:
            return 10
        return self.definition

    @override(ContainerStack)
    def getTop(self) -> 'InstanceContainer':
        if False:
            i = 10
            return i + 15
        return self.userChanges

    @pyqtSlot(str, result=bool)
    def hasUserValue(self, key: str) -> bool:
        if False:
            while True:
                i = 10
        "Check whether the specified setting has a 'user' value.\n\n        A user value here is defined as the setting having a value in either\n        the UserChanges or QualityChanges container.\n\n        :return: True if the setting has a user value, False if not.\n        "
        if self._containers[_ContainerIndexes.UserChanges].hasProperty(key, 'value'):
            return True
        if self._containers[_ContainerIndexes.QualityChanges].hasProperty(key, 'value'):
            return True
        return False

    def setProperty(self, key: str, property_name: str, property_value: Any, container: 'ContainerInterface'=None, set_from_cache: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set a property of a setting.\n\n        This will set a property of a specified setting. Since the container stack does not contain\n        any settings itself, it is required to specify a container to set the property on. The target\n        container is matched by container type.\n\n        :param key: The key of the setting to set.\n        :param property_name: The name of the property to set.\n        :param new_value: The new value to set the property to.\n        '
        container_index = _ContainerIndexes.UserChanges
        self._containers[container_index].setProperty(key, property_name, property_value, container, set_from_cache)

    @override(ContainerStack)
    def addContainer(self, container: ContainerInterface) -> None:
        if False:
            i = 10
            return i + 15
        'Overridden from ContainerStack\n\n        Since we have a fixed order of containers in the stack and this method would modify the container\n        ordering, we disallow this operation.\n        '
        raise Exceptions.InvalidOperationError('Cannot add a container to Global stack')

    @override(ContainerStack)
    def insertContainer(self, index: int, container: ContainerInterface) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Overridden from ContainerStack\n\n        Since we have a fixed order of containers in the stack and this method would modify the container\n        ordering, we disallow this operation.\n        '
        raise Exceptions.InvalidOperationError('Cannot insert a container into Global stack')

    @override(ContainerStack)
    def removeContainer(self, index: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Overridden from ContainerStack\n\n        Since we have a fixed order of containers in the stack and this method would modify the container\n        ordering, we disallow this operation.\n        '
        raise Exceptions.InvalidOperationError('Cannot remove a container from Global stack')

    @override(ContainerStack)
    def replaceContainer(self, index: int, container: ContainerInterface, postpone_emit: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Overridden from ContainerStack\n\n        Replaces the container at the specified index with another container.\n        This version performs checks to make sure the new container has the expected metadata and type.\n\n        :throws Exception.InvalidContainerError Raised when trying to replace a container with a container that has an incorrect type.\n        '
        expected_type = _ContainerIndexes.IndexTypeMap[index]
        if expected_type == 'definition':
            if not isinstance(container, DefinitionContainer):
                raise Exceptions.InvalidContainerError('Cannot replace container at index {index} with a container that is not a DefinitionContainer'.format(index=index))
        elif container != self._empty_instance_container and container.getMetaDataEntry('type') != expected_type:
            raise Exceptions.InvalidContainerError('Cannot replace container at index {index} with a container that is not of {type} type, but {actual_type} type.'.format(index=index, type=expected_type, actual_type=container.getMetaDataEntry('type')))
        current_container = self._containers[index]
        if current_container.getId() == container.getId():
            return
        super().replaceContainer(index, container, postpone_emit)

    @override(ContainerStack)
    def deserialize(self, serialized: str, file_name: Optional[str]=None) -> str:
        if False:
            print('Hello World!')
        'Overridden from ContainerStack\n\n        This deserialize will make sure the internal list of containers matches with what we expect.\n        It will first check to see if the container at a certain index already matches with what we\n        expect. If it does not, it will search for a matching container with the correct type. Should\n        no container with the correct type be found, it will use the empty container.\n\n        :raise InvalidContainerStackError: Raised when no definition can be found for the stack.\n        '
        serialized = super().deserialize(serialized, file_name)
        new_containers = self._containers.copy()
        while len(new_containers) < len(_ContainerIndexes.IndexTypeMap):
            new_containers.append(self._empty_instance_container)
        for (index, type_name) in _ContainerIndexes.IndexTypeMap.items():
            container = None
            try:
                container = new_containers[index]
            except IndexError:
                pass
            if type_name == 'definition':
                if not container or not isinstance(container, DefinitionContainer):
                    definition = self.findContainer(container_type=DefinitionContainer)
                    if not definition:
                        raise InvalidContainerStackError('Stack {id} does not have a definition!'.format(id=self.getId()))
                    new_containers[index] = definition
                continue
            if not container or container.getMetaDataEntry('type') != type_name:
                actual_container = self.findContainer(type=type_name)
                if actual_container:
                    new_containers[index] = actual_container
                else:
                    new_containers[index] = self._empty_instance_container
        self._containers = new_containers
        if isinstance(new_containers[_ContainerIndexes.DefinitionChanges], type(self._empty_instance_container)):
            from cura.Settings.CuraStackBuilder import CuraStackBuilder
            CuraStackBuilder.createDefinitionChangesContainer(self, self.getId() + '_settings')
        return serialized

    def _onContainersChanged(self, container: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Helper to make sure we emit a PyQt signal on container changes.'
        Application.getInstance().callLater(self.pyqtContainersChanged.emit)

    def _getMachineDefinition(self) -> DefinitionContainer:
        if False:
            while True:
                i = 10
        return self.definition

    @classmethod
    def findInstanceContainerDefinitionId(cls, machine_definition: DefinitionContainerInterface) -> str:
        if False:
            while True:
                i = 10
        'Find the ID that should be used when searching for instance containers for a specified definition.\n\n        This handles the situation where the definition specifies we should use a different definition when\n        searching for instance containers.\n\n        :param machine_definition: The definition to find the "quality definition" for.\n\n        :return: The ID of the definition container to use when searching for instance containers.\n        '
        quality_definition = machine_definition.getMetaDataEntry('quality_definition')
        if not quality_definition:
            return machine_definition.id
        definitions = ContainerRegistry.getInstance().findDefinitionContainers(id=quality_definition)
        if not definitions:
            Logger.log('w', 'Unable to find parent definition {parent} for machine {machine}', parent=quality_definition, machine=machine_definition.id)
            return machine_definition.id
        return cls.findInstanceContainerDefinitionId(definitions[0])

    def getExtruderPositionValueWithDefault(self, key):
        if False:
            i = 10
            return i + 15
        'getProperty for extruder positions, with translation from -1 to default extruder number'
        value = self.getProperty(key, 'value')
        if value == -1:
            value = int(Application.getInstance().getMachineManager().defaultExtruderPosition)
        return value

    def getProperty(self, key: str, property_name: str, context=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if property_name == 'settable_per_extruder':
            try:
                return self._settable_per_extruder_cache[key]
            except KeyError:
                self._settable_per_extruder_cache[key] = super().getProperty(key, property_name, context)
                return self._settable_per_extruder_cache[key]
        return super().getProperty(key, property_name, context)

class _ContainerIndexes:
    """Private helper class to keep track of container positions and their types."""
    UserChanges = 0
    QualityChanges = 1
    Intent = 2
    Quality = 3
    Material = 4
    Variant = 5
    DefinitionChanges = 6
    Definition = 7
    IndexTypeMap = {UserChanges: 'user', QualityChanges: 'quality_changes', Intent: 'intent', Quality: 'quality', Material: 'material', Variant: 'variant', DefinitionChanges: 'definition_changes', Definition: 'definition'}
    TypeIndexMap = {v: k for (k, v) in IndexTypeMap.items()}