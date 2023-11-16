from typing import TYPE_CHECKING, Optional, List, Set, Dict
from PyQt6.QtCore import QObject
from UM.FlameProfiler import pyqtSlot
from UM.Logger import Logger
from UM.PluginRegistry import PluginRegistry
if TYPE_CHECKING:
    from cura.CuraApplication import CuraApplication
    from cura.Settings.GlobalStack import GlobalStack
    from cura.MachineAction import MachineAction

class UnknownMachineActionError(Exception):
    """Raised when trying to add an unknown machine action as a required action"""
    pass

class NotUniqueMachineActionError(Exception):
    """Raised when trying to add a machine action that does not have an unique key."""
    pass

class MachineActionManager(QObject):

    def __init__(self, application: 'CuraApplication', parent: Optional['QObject']=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent=parent)
        self._application = application
        self._container_registry = self._application.getContainerRegistry()
        self._definition_ids_with_default_actions_added = set()
        self._machine_actions = {}
        self._required_actions = {}
        self._supported_actions = {}
        self._first_start_actions = {}

    def initialize(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        PluginRegistry.addType('machine_action', self.addMachineAction)

    def addDefaultMachineActions(self, global_stack: 'GlobalStack') -> None:
        if False:
            for i in range(10):
                print('nop')
        definition_id = global_stack.definition.getId()
        if definition_id in self._definition_ids_with_default_actions_added:
            Logger.log('i', 'Default machine actions have been added for machine definition [%s], do nothing.', definition_id)
            return
        supported_actions = global_stack.getMetaDataEntry('supported_actions', [])
        for action_key in supported_actions:
            self.addSupportedAction(definition_id, action_key)
        required_actions = global_stack.getMetaDataEntry('required_actions', [])
        for action_key in required_actions:
            self.addRequiredAction(definition_id, action_key)
        first_start_actions = global_stack.getMetaDataEntry('first_start_actions', [])
        for action_key in first_start_actions:
            self.addFirstStartAction(definition_id, action_key)
        self._definition_ids_with_default_actions_added.add(definition_id)
        Logger.log('i', 'Default machine actions added for machine definition [%s]', definition_id)

    def addRequiredAction(self, definition_id: str, action_key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a required action to a machine\n\n        Raises an exception when the action is not recognised.\n        '
        if action_key in self._machine_actions:
            if definition_id in self._required_actions:
                if self._machine_actions[action_key] not in self._required_actions[definition_id]:
                    self._required_actions[definition_id].append(self._machine_actions[action_key])
            else:
                self._required_actions[definition_id] = [self._machine_actions[action_key]]
        else:
            raise UnknownMachineActionError('Action %s, which is required for %s is not known.' % (action_key, definition_id))

    def addSupportedAction(self, definition_id: str, action_key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a supported action to a machine.'
        if action_key in self._machine_actions:
            if definition_id in self._supported_actions:
                if self._machine_actions[action_key] not in self._supported_actions[definition_id]:
                    self._supported_actions[definition_id].append(self._machine_actions[action_key])
            else:
                self._supported_actions[definition_id] = [self._machine_actions[action_key]]
        else:
            Logger.log('w', 'Unable to add %s to %s, as the action is not recognised', action_key, definition_id)

    def addFirstStartAction(self, definition_id: str, action_key: str) -> None:
        if False:
            i = 10
            return i + 15
        'Add an action to the first start list of a machine.'
        if action_key in self._machine_actions:
            if definition_id in self._first_start_actions:
                self._first_start_actions[definition_id].append(self._machine_actions[action_key])
            else:
                self._first_start_actions[definition_id] = [self._machine_actions[action_key]]
        else:
            Logger.log('w', 'Unable to add %s to %s, as the action is not recognised', action_key, definition_id)

    def addMachineAction(self, action: 'MachineAction') -> None:
        if False:
            while True:
                i = 10
        'Add a (unique) MachineAction\n\n        if the Key of the action is not unique, an exception is raised.\n        '
        if action.getKey() not in self._machine_actions:
            self._machine_actions[action.getKey()] = action
        else:
            raise NotUniqueMachineActionError('MachineAction with key %s was already added. Actions must have unique keys.', action.getKey())

    @pyqtSlot(str, result='QVariantList')
    def getSupportedActions(self, definition_id: str) -> List['MachineAction']:
        if False:
            for i in range(10):
                print('nop')
        'Get all actions supported by given machine\n\n        :param definition_id: The ID of the definition you want the supported actions of\n        :returns: set of supported actions.\n        '
        if definition_id in self._supported_actions:
            return list(self._supported_actions[definition_id])
        else:
            return list()

    def getRequiredActions(self, definition_id: str) -> List['MachineAction']:
        if False:
            i = 10
            return i + 15
        'Get all actions required by given machine\n\n        :param definition_id: The ID of the definition you want the required actions of\n        :returns: set of required actions.\n        '
        if definition_id in self._required_actions:
            return self._required_actions[definition_id]
        else:
            return list()

    @pyqtSlot(str, result='QVariantList')
    def getFirstStartActions(self, definition_id: str) -> List['MachineAction']:
        if False:
            return 10
        'Get all actions that need to be performed upon first start of a given machine.\n\n        Note that contrary to required / supported actions a list is returned (as it could be required to run the same\n        action multiple times).\n        :param definition_id: The ID of the definition that you want to get the "on added" actions for.\n        :returns: List of actions.\n        '
        if definition_id in self._first_start_actions:
            return self._first_start_actions[definition_id]
        else:
            return []

    def removeMachineAction(self, action: 'MachineAction') -> None:
        if False:
            print('Hello World!')
        'Remove Machine action from manager\n\n        :param action: to remove\n        '
        try:
            del self._machine_actions[action.getKey()]
        except KeyError:
            Logger.log('w', 'Trying to remove MachineAction (%s) that was already removed', action.getKey())

    def getMachineAction(self, key: str) -> Optional['MachineAction']:
        if False:
            i = 10
            return i + 15
        'Get MachineAction by key\n\n        :param key: String of key to select\n        :return: Machine action if found, None otherwise\n        '
        if key in self._machine_actions:
            return self._machine_actions[key]
        else:
            return None