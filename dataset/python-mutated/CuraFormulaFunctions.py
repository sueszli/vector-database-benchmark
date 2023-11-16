from typing import Any, List, Optional, TYPE_CHECKING
from UM.Settings.PropertyEvaluationContext import PropertyEvaluationContext
from UM.Settings.SettingFunction import SettingFunction
from UM.Logger import Logger
if TYPE_CHECKING:
    from cura.CuraApplication import CuraApplication
    from cura.Settings.CuraContainerStack import CuraContainerStack

class CuraFormulaFunctions:

    def __init__(self, application: 'CuraApplication') -> None:
        if False:
            for i in range(10):
                print('nop')
        self._application = application

    def getDefaultExtruderPosition(self) -> str:
        if False:
            i = 10
            return i + 15
        machine_manager = self._application.getMachineManager()
        return machine_manager.defaultExtruderPosition

    def getValueInExtruder(self, extruder_position: int, property_key: str, context: Optional['PropertyEvaluationContext']=None) -> Any:
        if False:
            while True:
                i = 10
        machine_manager = self._application.getMachineManager()
        if extruder_position == -1:
            extruder_position = int(machine_manager.defaultExtruderPosition)
        global_stack = machine_manager.activeMachine
        try:
            extruder_stack = global_stack.extruderList[int(extruder_position)]
        except IndexError:
            if extruder_position != 0:
                Logger.log('w', 'Value for %s of extruder %s was requested, but that extruder is not available. Returning the result from extruder 0 instead' % (property_key, extruder_position))
                return self.getValueInExtruder(0, property_key, context)
            Logger.log('w', 'Value for %s of extruder %s was requested, but that extruder is not available. ' % (property_key, extruder_position))
            return None
        value = extruder_stack.getRawProperty(property_key, 'value', context=context)
        if isinstance(value, SettingFunction):
            value = value(extruder_stack, context=context)
        if isinstance(value, str):
            value = value.lower()
        return value

    def _getActiveExtruders(self, context: Optional['PropertyEvaluationContext']=None) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        machine_manager = self._application.getMachineManager()
        extruder_manager = self._application.getExtruderManager()
        global_stack = machine_manager.activeMachine
        result = []
        for extruder in extruder_manager.getActiveExtruderStacks():
            if not extruder.isEnabled:
                continue
            if int(extruder.getMetaDataEntry('position')) >= global_stack.getProperty('machine_extruder_count', 'value', context=context):
                continue
            result.append(extruder)
        return result

    def getValuesInAllExtruders(self, property_key: str, context: Optional['PropertyEvaluationContext']=None) -> List[Any]:
        if False:
            for i in range(10):
                print('nop')
        global_stack = self._application.getMachineManager().activeMachine
        result = []
        for extruder in self._getActiveExtruders(context):
            value = extruder.getRawProperty(property_key, 'value', context=context)
            if value is None:
                continue
            if isinstance(value, SettingFunction):
                value = value(extruder, context=context)
            result.append(value)
        if not result:
            result.append(global_stack.getProperty(property_key, 'value', context=context))
        return result

    def getAnyExtruderPositionWithOrDefault(self, filter_key: str, context: Optional['PropertyEvaluationContext']=None) -> str:
        if False:
            print('Hello World!')
        for extruder in self._getActiveExtruders(context):
            value = extruder.getRawProperty(filter_key, 'value', context=context)
            if value is None or not value:
                continue
            return str(extruder.position)

    def getExtruderPositionWithMaterial(self, filter_key: str, context: Optional['PropertyEvaluationContext']=None) -> str:
        if False:
            while True:
                i = 10
        for extruder in self._getActiveExtruders(context):
            material_container = extruder.material
            value = material_container.getProperty(filter_key, 'value', context)
            if value is not None:
                return str(extruder.position)
        return self.getDefaultExtruderPosition()

    def getResolveOrValue(self, property_key: str, context: Optional['PropertyEvaluationContext']=None) -> Any:
        if False:
            i = 10
            return i + 15
        machine_manager = self._application.getMachineManager()
        global_stack = machine_manager.activeMachine
        resolved_value = global_stack.getProperty(property_key, 'value', context=context)
        return resolved_value

    def getDefaultValueInExtruder(self, extruder_position: int, property_key: str) -> Any:
        if False:
            i = 10
            return i + 15
        machine_manager = self._application.getMachineManager()
        global_stack = machine_manager.activeMachine
        try:
            extruder_stack = global_stack.extruderList[extruder_position]
        except IndexError:
            Logger.log('w', 'Unable to find extruder on in index %s', extruder_position)
        else:
            context = self.createContextForDefaultValueEvaluation(extruder_stack)
            return self.getValueInExtruder(extruder_position, property_key, context=context)

    def getDefaultValuesInAllExtruders(self, property_key: str) -> List[Any]:
        if False:
            i = 10
            return i + 15
        machine_manager = self._application.getMachineManager()
        global_stack = machine_manager.activeMachine
        context = self.createContextForDefaultValueEvaluation(global_stack)
        return self.getValuesInAllExtruders(property_key, context=context)

    def getDefaultResolveOrValue(self, property_key: str) -> Any:
        if False:
            i = 10
            return i + 15
        machine_manager = self._application.getMachineManager()
        global_stack = machine_manager.activeMachine
        context = self.createContextForDefaultValueEvaluation(global_stack)
        return self.getResolveOrValue(property_key, context=context)

    def getValueFromContainerAtIndex(self, property_key: str, container_index: int, context: Optional['PropertyEvaluationContext']=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        machine_manager = self._application.getMachineManager()
        global_stack = machine_manager.activeMachine
        context = self.createContextForDefaultValueEvaluation(global_stack)
        context.context['evaluate_from_container_index'] = container_index
        return global_stack.getProperty(property_key, 'value', context=context)

    def getValueFromContainerAtIndexInExtruder(self, extruder_position: int, property_key: str, container_index: int, context: Optional['PropertyEvaluationContext']=None) -> Any:
        if False:
            print('Hello World!')
        machine_manager = self._application.getMachineManager()
        global_stack = machine_manager.activeMachine
        if extruder_position == -1:
            extruder_position = int(machine_manager.defaultExtruderPosition)
        global_stack = machine_manager.activeMachine
        try:
            extruder_stack = global_stack.extruderList[int(extruder_position)]
        except IndexError:
            Logger.log('w', 'Value for %s of extruder %s was requested, but that extruder is not available. ' % (property_key, extruder_position))
            return None
        context = self.createContextForDefaultValueEvaluation(extruder_stack)
        context.context['evaluate_from_container_index'] = container_index
        return self.getValueInExtruder(extruder_position, property_key, context)

    def createContextForDefaultValueEvaluation(self, source_stack: 'CuraContainerStack') -> 'PropertyEvaluationContext':
        if False:
            print('Hello World!')
        context = PropertyEvaluationContext(source_stack)
        context.context['evaluate_from_container_index'] = 1
        context.context['override_operators'] = {'extruderValue': self.getDefaultValueInExtruder, 'extruderValues': self.getDefaultValuesInAllExtruders, 'resolveOrValue': self.getDefaultResolveOrValue}
        return context