from typing import Any, Optional
from UM.Application import Application
from UM.Decorators import override
from UM.Settings.Interfaces import PropertyEvaluationContext
from UM.Settings.SettingInstance import InstanceState
from .CuraContainerStack import CuraContainerStack

class PerObjectContainerStack(CuraContainerStack):

    def isDirty(self):
        if False:
            i = 10
            return i + 15
        return False

    @override(CuraContainerStack)
    def getProperty(self, key: str, property_name: str, context: Optional[PropertyEvaluationContext]=None) -> Any:
        if False:
            print('Hello World!')
        if context is None:
            context = PropertyEvaluationContext()
        context.pushContainer(self)
        global_stack = Application.getInstance().getGlobalContainerStack()
        if not global_stack:
            return None
        if self.getContainer(0).hasProperty(key, property_name):
            if self.getContainer(0).getProperty(key, 'state') == InstanceState.User:
                result = super().getProperty(key, property_name, context)
                context.popContainer()
                return result
        limit_to_extruder = super().getProperty(key, 'limit_to_extruder', context)
        if limit_to_extruder is not None:
            limit_to_extruder = str(limit_to_extruder)
        if limit_to_extruder == '-1':
            if 'original_limit_to_extruder' in context.context:
                limit_to_extruder = context.context['original_limit_to_extruder']
        if limit_to_extruder is not None and limit_to_extruder != '-1' and (int(limit_to_extruder) <= len(global_stack.extruderList)):
            if 'original_limit_to_extruder' not in context.context:
                context.context['original_limit_to_extruder'] = limit_to_extruder
            if super().getProperty(key, 'settable_per_extruder', context):
                result = global_stack.extruderList[int(limit_to_extruder)].getProperty(key, property_name, context)
                if result is not None:
                    context.popContainer()
                    return result
        result = super().getProperty(key, property_name, context)
        context.popContainer()
        return result

    @override(CuraContainerStack)
    def setNextStack(self, stack: CuraContainerStack) -> None:
        if False:
            i = 10
            return i + 15
        super().setNextStack(stack)
        for key in self.getContainer(0).getAllKeys():
            if self.getContainer(0).getProperty(key, 'state') != InstanceState.Default:
                continue
            self._collectPropertyChanges(key, 'value')
        self._emitCollectedPropertyChanges()