from collections import defaultdict
import threading
from typing import Any, Dict, Optional, Set, TYPE_CHECKING, List
import uuid
from PyQt6.QtCore import pyqtProperty, pyqtSlot, pyqtSignal
from UM.Decorators import override
from UM.MimeTypeDatabase import MimeType, MimeTypeDatabase
from UM.Settings.ContainerStack import ContainerStack
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.Interfaces import PropertyEvaluationContext
from UM.Logger import Logger
from UM.Resources import Resources
from UM.Platform import Platform
from UM.Util import parseBool
import cura.CuraApplication
from cura.PrinterOutput.PrinterOutputDevice import ConnectionType
from . import Exceptions
from .CuraContainerStack import CuraContainerStack
if TYPE_CHECKING:
    from cura.Settings.ExtruderStack import ExtruderStack

class GlobalStack(CuraContainerStack):
    """Represents the Global or Machine stack and its related containers."""

    def __init__(self, container_id: str) -> None:
        if False:
            print('Hello World!')
        super().__init__(container_id)
        self.setMetaDataEntry('type', 'machine')
        self.setMetaDataEntry('group_id', str(uuid.uuid4()))
        self._extruders = {}
        self._resolving_settings = defaultdict(set)
        self.metaDataChanged.connect(self.configuredConnectionTypesChanged)
        self.setDirty(False)
    extrudersChanged = pyqtSignal()
    configuredConnectionTypesChanged = pyqtSignal()

    @pyqtProperty('QVariantList', notify=extrudersChanged)
    def extruderList(self) -> List['ExtruderStack']:
        if False:
            return 10
        result_tuple_list = sorted(list(self._extruders.items()), key=lambda x: int(x[0]))
        result_list = [item[1] for item in result_tuple_list]
        machine_extruder_count = self.getProperty('machine_extruder_count', 'value')
        return result_list[:machine_extruder_count]

    @pyqtProperty(int, constant=True)
    def maxExtruderCount(self):
        if False:
            print('Hello World!')
        return len(self.getMetaDataEntry('machine_extruder_trains'))

    @pyqtProperty(bool, notify=configuredConnectionTypesChanged)
    def supportsNetworkConnection(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getMetaDataEntry('supports_network_connection', False)

    @pyqtProperty(bool, constant=True)
    def supportsMaterialExport(self):
        if False:
            while True:
                i = 10
        "\n        Whether the printer supports Cura's export format of material profiles.\n        :return: ``True`` if it supports it, or ``False`` if not.\n        "
        return self.getMetaDataEntry('supports_material_export', False)

    @classmethod
    def getLoadingPriority(cls) -> int:
        if False:
            return 10
        return 2

    @pyqtProperty('QVariantList', notify=configuredConnectionTypesChanged)
    def configuredConnectionTypes(self) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        "The configured connection types can be used to find out if the global\n        stack is configured to be connected with a printer, without having to\n        know all the details as to how this is exactly done (and without\n        actually setting the stack to be active).\n\n        This data can then in turn also be used when the global stack is active;\n        If we can't get a network connection, but it is configured to have one,\n        we can display a different icon to indicate the difference.\n        "
        connection_types = self.getMetaDataEntry('connection_type', '').split(',')
        result = []
        for connection_type in connection_types:
            if connection_type != '':
                try:
                    result.append(int(connection_type))
                except ValueError:
                    pass
        return result

    @pyqtProperty(bool, notify=configuredConnectionTypesChanged)
    def hasRemoteConnection(self) -> bool:
        if False:
            print('Hello World!')
        has_remote_connection = False
        for connection_type in self.configuredConnectionTypes:
            has_remote_connection |= connection_type in [ConnectionType.NetworkConnection.value, ConnectionType.CloudConnection.value]
        return has_remote_connection

    def addConfiguredConnectionType(self, connection_type: int) -> None:
        if False:
            return 10
        ':sa configuredConnectionTypes'
        configured_connection_types = self.configuredConnectionTypes
        if connection_type not in configured_connection_types:
            configured_connection_types.append(connection_type)
            self.setMetaDataEntry('connection_type', ','.join([str(c_type) for c_type in configured_connection_types]))

    def removeConfiguredConnectionType(self, connection_type: int) -> None:
        if False:
            return 10
        ':sa configuredConnectionTypes'
        configured_connection_types = self.configuredConnectionTypes
        if connection_type in configured_connection_types:
            configured_connection_types.remove(connection_type)
            self.setMetaDataEntry('connection_type', ','.join([str(c_type) for c_type in configured_connection_types]))

    @classmethod
    def getConfigurationTypeFromSerialized(cls, serialized: str) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        configuration_type = super().getConfigurationTypeFromSerialized(serialized)
        if configuration_type == 'machine':
            return 'machine_stack'
        return configuration_type

    def getIntentCategory(self) -> str:
        if False:
            return 10
        intent_category = 'default'
        for extruder in self.extruderList:
            category = extruder.intent.getMetaDataEntry('intent_category', 'default')
            if category != 'default' and category != intent_category:
                intent_category = category
        return intent_category

    def getBuildplateName(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        name = None
        if self.variant.getId() != 'empty_variant':
            name = self.variant.getName()
        return name

    @pyqtProperty(str, constant=True)
    def preferred_output_file_formats(self) -> str:
        if False:
            print('Hello World!')
        return self.getMetaDataEntry('file_formats')

    def addExtruder(self, extruder: ContainerStack) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add an extruder to the list of extruders of this stack.\n\n        :param extruder: The extruder to add.\n\n        :raise Exceptions.TooManyExtrudersError: Raised when trying to add an extruder while we\n            already have the maximum number of extruders.\n        '
        position = extruder.getMetaDataEntry('position')
        if position is None:
            Logger.log('w', 'No position defined for extruder {extruder}, cannot add it to stack {stack}', extruder=extruder.id, stack=self.id)
            return
        if any((item.getId() == extruder.id for item in self._extruders.values())):
            Logger.log('w', 'Extruder [%s] has already been added to this stack [%s]', extruder.id, self.getId())
            return
        self._extruders[position] = extruder
        self.extrudersChanged.emit()
        Logger.log('i', 'Extruder[%s] added to [%s] at position [%s]', extruder.id, self.id, position)

    @override(ContainerStack)
    def getProperty(self, key: str, property_name: str, context: Optional[PropertyEvaluationContext]=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Overridden from ContainerStack\n\n        This will return the value of the specified property for the specified setting,\n        unless the property is "value" and that setting has a "resolve" function set.\n        When a resolve is set, it will instead try and execute the resolve first and\n        then fall back to the normal "value" property.\n\n        :param key: The setting key to get the property of.\n        :param property_name: The property to get the value of.\n\n        :return: The value of the property for the specified setting, or None if not found.\n        '
        if not self.definition.findDefinitions(key=key):
            return None
        if context:
            context.pushContainer(self)
        if self._shouldResolve(key, property_name, context):
            current_thread = threading.current_thread()
            self._resolving_settings[current_thread.name].add(key)
            resolve = super().getProperty(key, 'resolve', context)
            self._resolving_settings[current_thread.name].remove(key)
            if resolve is not None:
                return resolve
        limit_to_extruder = super().getProperty(key, 'limit_to_extruder', context)
        if limit_to_extruder is not None:
            limit_to_extruder = str(limit_to_extruder)
        if limit_to_extruder is not None and limit_to_extruder != '-1' and (limit_to_extruder in self._extruders):
            if super().getProperty(key, 'settable_per_extruder', context):
                result = self._extruders[str(limit_to_extruder)].getProperty(key, property_name, context)
                if result is not None:
                    if context:
                        context.popContainer()
                    return result
            else:
                Logger.log('e', 'Setting {setting} has limit_to_extruder but is not settable per extruder!', setting=key)
        result = super().getProperty(key, property_name, context)
        if context:
            context.popContainer()
        return result

    @override(ContainerStack)
    def setNextStack(self, stack: CuraContainerStack, connect_signals: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Overridden from ContainerStack\n\n        This will simply raise an exception since the Global stack cannot have a next stack.\n        '
        raise Exceptions.InvalidOperationError('Global stack cannot have a next stack!')

    def _shouldResolve(self, key: str, property_name: str, context: Optional[PropertyEvaluationContext]=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if property_name != 'value':
            return False
        if not self.definition.getProperty(key, 'resolve'):
            return False
        current_thread = threading.current_thread()
        if key in self._resolving_settings[current_thread.name]:
            return False
        if self.hasUserValue(key):
            return False
        return True

    def isValid(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Perform some sanity checks on the global stack\n\n        Sanity check for extruders; they must have positions 0 and up to machine_extruder_count - 1\n        '
        container_registry = ContainerRegistry.getInstance()
        extruder_trains = container_registry.findContainerStacks(type='extruder_train', machine=self.getId())
        machine_extruder_count = self.getProperty('machine_extruder_count', 'value')
        extruder_check_position = set()
        for extruder_train in extruder_trains:
            extruder_position = extruder_train.getMetaDataEntry('position')
            extruder_check_position.add(extruder_position)
        for check_position in range(machine_extruder_count):
            if str(check_position) not in extruder_check_position:
                return False
        return True

    def getHeadAndFansCoordinates(self):
        if False:
            i = 10
            return i + 15
        return self.getProperty('machine_head_with_fans_polygon', 'value')

    @pyqtProperty(bool, constant=True)
    def hasMaterials(self) -> bool:
        if False:
            print('Hello World!')
        return parseBool(self.getMetaDataEntry('has_materials', False))

    @pyqtProperty(bool, constant=True)
    def hasVariants(self) -> bool:
        if False:
            i = 10
            return i + 15
        return parseBool(self.getMetaDataEntry('has_variants', False))

    @pyqtProperty(bool, constant=True)
    def hasVariantBuildplates(self) -> bool:
        if False:
            return 10
        return parseBool(self.getMetaDataEntry('has_variant_buildplates', False))

    @pyqtSlot(result=str)
    def getDefaultFirmwareName(self) -> str:
        if False:
            i = 10
            return i + 15
        'Get default firmware file name if one is specified in the firmware'
        machine_has_heated_bed = self.getProperty('machine_heated_bed', 'value')
        baudrate = 250000
        if Platform.isLinux():
            baudrate = 115200
        hex_file = self.getMetaDataEntry('firmware_file', None)
        if machine_has_heated_bed:
            hex_file = self.getMetaDataEntry('firmware_hbk_file', hex_file)
        if not hex_file:
            Logger.log('w', 'There is no firmware for machine %s.', self.getBottom().id)
            return ''
        try:
            return Resources.getPath(cura.CuraApplication.CuraApplication.ResourceTypes.Firmware, hex_file.format(baudrate=baudrate))
        except FileNotFoundError:
            Logger.log('w', 'Firmware file %s not found.', hex_file)
            return ''

    def getName(self) -> str:
        if False:
            print('Hello World!')
        return self._metadata.get('group_name', self._metadata.get('name', ''))

    def setName(self, name: str) -> None:
        if False:
            return 10
        super().setName(name)
    nameChanged = pyqtSignal()
    name = pyqtProperty(str, fget=getName, fset=setName, notify=nameChanged)

    def hasNetworkedConnection(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        has_connection = False
        for connection_type in [ConnectionType.NetworkConnection.value, ConnectionType.CloudConnection.value]:
            has_connection |= connection_type in self.configuredConnectionTypes
        return has_connection
global_stack_mime = MimeType(name='application/x-cura-globalstack', comment='Cura Global Stack', suffixes=['global.cfg'])
MimeTypeDatabase.addMimeType(global_stack_mime)
ContainerRegistry.addContainerTypeByName(GlobalStack, 'global_stack', global_stack_mime.name)