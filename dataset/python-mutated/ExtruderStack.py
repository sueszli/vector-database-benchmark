from typing import Any, Dict, TYPE_CHECKING, Optional
from PyQt6.QtCore import pyqtProperty, pyqtSignal
from UM.Decorators import override
from UM.MimeTypeDatabase import MimeType, MimeTypeDatabase
from UM.Settings.ContainerStack import ContainerStack
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Settings.Interfaces import ContainerInterface, PropertyEvaluationContext
from UM.Util import parseBool
import cura.CuraApplication
from . import Exceptions
from .CuraContainerStack import CuraContainerStack, _ContainerIndexes
from .ExtruderManager import ExtruderManager
if TYPE_CHECKING:
    from cura.Settings.GlobalStack import GlobalStack

class ExtruderStack(CuraContainerStack):
    """Represents an Extruder and its related containers."""

    def __init__(self, container_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(container_id)
        self.setMetaDataEntry('type', 'extruder_train')
        self.propertiesChanged.connect(self._onPropertiesChanged)
        self.setDirty(False)
    enabledChanged = pyqtSignal()

    @override(ContainerStack)
    def setNextStack(self, stack: CuraContainerStack, connect_signals: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        'Overridden from ContainerStack\n\n        This will set the next stack and ensure that we register this stack as an extruder.\n        '
        super().setNextStack(stack)
        stack.addExtruder(self)
        self.setMetaDataEntry('machine', stack.id)

    @override(ContainerStack)
    def getNextStack(self) -> Optional['GlobalStack']:
        if False:
            return 10
        return super().getNextStack()

    @pyqtProperty(int, constant=True)
    def position(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return int(self.getMetaDataEntry('position'))

    def setEnabled(self, enabled: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.getMetaDataEntry('enabled', True) == enabled:
            return
        self.setMetaDataEntry('enabled', str(enabled))
        self.enabledChanged.emit()

    @pyqtProperty(bool, notify=enabledChanged)
    def isEnabled(self) -> bool:
        if False:
            i = 10
            return i + 15
        return parseBool(self.getMetaDataEntry('enabled', 'True'))

    @classmethod
    def getLoadingPriority(cls) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 3
    compatibleMaterialDiameterChanged = pyqtSignal()

    def getCompatibleMaterialDiameter(self) -> float:
        if False:
            print('Hello World!')
        'Return the filament diameter that the machine requires.\n\n        If the machine has no requirement for the diameter, -1 is returned.\n        :return: The filament diameter for the printer\n        '
        context = PropertyEvaluationContext(self)
        context.context['evaluate_from_container_index'] = _ContainerIndexes.Variant
        return float(self.getProperty('material_diameter', 'value', context=context))

    def setCompatibleMaterialDiameter(self, value: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        old_approximate_diameter = self.getApproximateMaterialDiameter()
        if self.getCompatibleMaterialDiameter() != value:
            self.definitionChanges.setProperty('material_diameter', 'value', value)
            self.compatibleMaterialDiameterChanged.emit()
            if old_approximate_diameter != self.getApproximateMaterialDiameter():
                self.approximateMaterialDiameterChanged.emit()
    compatibleMaterialDiameter = pyqtProperty(float, fset=setCompatibleMaterialDiameter, fget=getCompatibleMaterialDiameter, notify=compatibleMaterialDiameterChanged)
    approximateMaterialDiameterChanged = pyqtSignal()

    def getApproximateMaterialDiameter(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Return the approximate filament diameter that the machine requires.\n\n        The approximate material diameter is the material diameter rounded to\n        the nearest millimetre.\n\n        If the machine has no requirement for the diameter, -1 is returned.\n\n        :return: The approximate filament diameter for the printer\n        '
        return round(self.getCompatibleMaterialDiameter())
    approximateMaterialDiameter = pyqtProperty(float, fget=getApproximateMaterialDiameter, notify=approximateMaterialDiameterChanged)

    @override(ContainerStack)
    def getProperty(self, key: str, property_name: str, context: Optional[PropertyEvaluationContext]=None) -> Any:
        if False:
            return 10
        'Overridden from ContainerStack\n\n        It will perform a few extra checks when trying to get properties.\n\n        The two extra checks it currently does is to ensure a next stack is set and to bypass\n        the extruder when the property is not settable per extruder.\n\n        :throws Exceptions.NoGlobalStackError Raised when trying to get a property from an extruder without\n        having a next stack set.\n        '
        if not self._next_stack:
            raise Exceptions.NoGlobalStackError('Extruder {id} is missing the next stack!'.format(id=self.id))
        if context:
            context.pushContainer(self)
        if not super().getProperty(key, 'settable_per_extruder', context):
            result = self.getNextStack().getProperty(key, property_name, context)
            if context:
                context.popContainer()
            return result
        limit_to_extruder = super().getProperty(key, 'limit_to_extruder', context)
        if limit_to_extruder is not None:
            limit_to_extruder = str(limit_to_extruder)
        if (limit_to_extruder is not None and limit_to_extruder != '-1') and self.getMetaDataEntry('position') != str(limit_to_extruder):
            try:
                result = self.getNextStack().extruderList[int(limit_to_extruder)].getProperty(key, property_name, context)
                if result is not None:
                    if context:
                        context.popContainer()
                    return result
            except IndexError:
                pass
        result = super().getProperty(key, property_name, context)
        if context:
            context.popContainer()
        return result

    @override(CuraContainerStack)
    def _getMachineDefinition(self) -> ContainerInterface:
        if False:
            print('Hello World!')
        if not self.getNextStack():
            raise Exceptions.NoGlobalStackError('Extruder {id} is missing the next stack!'.format(id=self.id))
        return self.getNextStack()._getMachineDefinition()

    @override(CuraContainerStack)
    def deserialize(self, contents: str, file_name: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        super().deserialize(contents, file_name)
        if 'enabled' not in self.getMetaData():
            self.setMetaDataEntry('enabled', 'True')

    def _onPropertiesChanged(self, key: str, properties: Dict[str, Any]) -> None:
        if False:
            return 10
        if not self.getNextStack():
            return
        definitions = self.getNextStack().definition.findDefinitions(key=key)
        if definitions:
            has_global_dependencies = False
            for relation in definitions[0].relations:
                if not getattr(relation.target, 'settable_per_extruder', True):
                    has_global_dependencies = True
                    break
            if has_global_dependencies:
                self.getNextStack().propertiesChanged.emit(key, properties)
extruder_stack_mime = MimeType(name='application/x-cura-extruderstack', comment='Cura Extruder Stack', suffixes=['extruder.cfg'])
MimeTypeDatabase.addMimeType(extruder_stack_mime)
ContainerRegistry.addContainerTypeByName(ExtruderStack, 'extruder_stack', extruder_stack_mime.name)