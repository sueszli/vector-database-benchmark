import pytest
from cura.MachineAction import MachineAction
from cura.UI.MachineActionManager import NotUniqueMachineActionError, UnknownMachineActionError
from cura.Settings.GlobalStack import GlobalStack

@pytest.fixture()
def global_stack():
    if False:
        for i in range(10):
            print('nop')
    gs = GlobalStack('test_global_stack')
    gs._metadata = {'supported_actions': ['supported_action_1', 'supported_action_2'], 'required_actions': ['required_action_1', 'required_action_2'], 'first_start_actions': ['first_start_actions_1', 'first_start_actions_2']}
    return gs

class Machine:

    def __init__(self, key=''):
        if False:
            i = 10
            return i + 15
        self._key = key

    def getKey(self):
        if False:
            i = 10
            return i + 15
        return self._key

def test_addDefaultMachineActions(machine_action_manager, global_stack):
    if False:
        return 10
    all_actions = []
    for action_key_list in global_stack._metadata.values():
        for key in action_key_list:
            all_actions.append(MachineAction(key=key))
    for action in all_actions:
        machine_action_manager.addMachineAction(action)
    machine_action_manager.addDefaultMachineActions(global_stack)
    definition_id = global_stack.getDefinition().getId()
    support_action_keys = [a.getKey() for a in machine_action_manager.getSupportedActions(definition_id)]
    assert support_action_keys == global_stack.getMetaDataEntry('supported_actions')
    required_action_keys = [a.getKey() for a in machine_action_manager.getRequiredActions(definition_id)]
    assert required_action_keys == global_stack.getMetaDataEntry('required_actions')
    first_start_action_keys = [a.getKey() for a in machine_action_manager.getFirstStartActions(definition_id)]
    assert first_start_action_keys == global_stack.getMetaDataEntry('first_start_actions')

def test_addMachineAction(machine_action_manager):
    if False:
        i = 10
        return i + 15
    test_action = MachineAction(key='test_action')
    test_action_2 = MachineAction(key='test_action_2')
    test_machine = Machine('test_machine')
    machine_action_manager.addMachineAction(test_action)
    machine_action_manager.addMachineAction(test_action_2)
    assert machine_action_manager.getMachineAction('test_action') == test_action
    assert machine_action_manager.getMachineAction('key_that_doesnt_exist') is None
    with pytest.raises(NotUniqueMachineActionError):
        machine_action_manager.addMachineAction(test_action)
    assert machine_action_manager.getSupportedActions(test_machine) == list()
    machine_action_manager.addSupportedAction(test_machine, 'test_action')
    assert machine_action_manager.getSupportedActions(test_machine) == [test_action]
    machine_action_manager.addSupportedAction(test_machine, 'key_that_doesnt_exist')
    assert machine_action_manager.getSupportedActions(test_machine) == [test_action]
    machine_action_manager.addSupportedAction(test_machine, 'test_action_2')
    assert machine_action_manager.getSupportedActions(test_machine) == [test_action, test_action_2]
    assert machine_action_manager.getRequiredActions(test_machine) == list()
    with pytest.raises(UnknownMachineActionError):
        machine_action_manager.addRequiredAction(test_machine, 'key_that_doesnt_exist')
    machine_action_manager.addRequiredAction(test_machine, 'test_action')
    assert machine_action_manager.getRequiredActions(test_machine) == [test_action]
    machine_action_manager.addRequiredAction(test_machine, 'test_action_2')
    assert machine_action_manager.getRequiredActions(test_machine) == [test_action, test_action_2]
    assert machine_action_manager.getFirstStartActions(test_machine) == []
    machine_action_manager.addFirstStartAction(test_machine, 'test_action')
    machine_action_manager.addFirstStartAction(test_machine, 'test_action')
    assert machine_action_manager.getFirstStartActions(test_machine) == [test_action, test_action]
    machine_action_manager.addFirstStartAction(test_machine, 'key_that_doesnt_exists')

def test_removeMachineAction(machine_action_manager):
    if False:
        print('Hello World!')
    test_action = MachineAction(key='test_action')
    machine_action_manager.addMachineAction(test_action)
    machine_action_manager.removeMachineAction(test_action)
    assert machine_action_manager.getMachineAction('test_action') is None
    machine_action_manager.removeMachineAction(test_action)