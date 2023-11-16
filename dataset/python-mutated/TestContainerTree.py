from unittest.mock import patch, MagicMock
import pytest
from cura.Machines.ContainerTree import ContainerTree
from cura.Settings.GlobalStack import GlobalStack

def createMockedStack(definition_id: str):
    if False:
        return 10
    result = MagicMock(spec=GlobalStack)
    result.definition.getId = MagicMock(return_value=definition_id)
    extruder_left_mock = MagicMock()
    extruder_left_mock.variant.getName = MagicMock(return_value=definition_id + '_left_variant_name')
    extruder_left_mock.material.getMetaDataEntry = MagicMock(return_value=definition_id + '_left_material_base_file')
    extruder_left_mock.isEnabled = True
    extruder_right_mock = MagicMock()
    extruder_right_mock.variant.getName = MagicMock(return_value=definition_id + '_right_variant_name')
    extruder_right_mock.material.getMetaDataEntry = MagicMock(return_value=definition_id + '_right_material_base_file')
    extruder_right_mock.isEnabled = True
    extruder_list = [extruder_left_mock, extruder_right_mock]
    result.extruderList = extruder_list
    return result

@pytest.fixture
def container_registry():
    if False:
        i = 10
        return i + 15
    result = MagicMock()
    result.findContainerStacks = MagicMock(return_value=[createMockedStack('machine_1'), createMockedStack('machine_2')])
    result.findContainersMetadata = lambda id: [{'id': id}] if id in {'machine_1', 'machine_2'} else []
    return result

@pytest.fixture
def application():
    if False:
        return 10
    return MagicMock(getGlobalContainerStack=MagicMock(return_value=createMockedStack('current_global_stack')))

def test_containerTreeInit(container_registry):
    if False:
        i = 10
        return i + 15
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with patch('UM.Application.Application.getInstance'):
            container_tree = ContainerTree()
        assert 'machine_1' in container_tree.machines
        assert 'machine_2' in container_tree.machines

def test_getCurrentQualityGroupsNoGlobalStack(container_registry):
    if False:
        print('Hello World!')
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with patch('cura.CuraApplication.CuraApplication.getInstance', MagicMock(return_value=MagicMock(getGlobalContainerStack=MagicMock(return_value=None)))):
            container_tree = ContainerTree()
            result = container_tree.getCurrentQualityGroups()
    assert len(result) == 0

def test_getCurrentQualityGroups(container_registry, application):
    if False:
        i = 10
        return i + 15
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with patch('cura.CuraApplication.CuraApplication.getInstance', MagicMock(return_value=application)):
            container_tree = ContainerTree()
            container_tree.machines._machines['current_global_stack'] = MagicMock()
            result = container_tree.getCurrentQualityGroups()
    expected_variant_names = ['current_global_stack_left_variant_name', 'current_global_stack_right_variant_name']
    expected_material_base_files = ['current_global_stack_left_material_base_file', 'current_global_stack_right_material_base_file']
    expected_is_enabled = [True, True]
    container_tree.machines['current_global_stack'].getQualityGroups.assert_called_with(expected_variant_names, expected_material_base_files, expected_is_enabled)
    assert result == container_tree.machines['current_global_stack'].getQualityGroups.return_value

def test_getCurrentQualityChangesGroupsNoGlobalStack(container_registry):
    if False:
        print('Hello World!')
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with patch('cura.CuraApplication.CuraApplication.getInstance', MagicMock(return_value=MagicMock(getGlobalContainerStack=MagicMock(return_value=None)))):
            container_tree = ContainerTree()
            result = container_tree.getCurrentQualityChangesGroups()
    assert len(result) == 0

def test_getCurrentQualityChangesGroups(container_registry, application):
    if False:
        return 10
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with patch('cura.CuraApplication.CuraApplication.getInstance', MagicMock(return_value=application)):
            container_tree = ContainerTree()
            container_tree.machines._machines['current_global_stack'] = MagicMock()
            result = container_tree.getCurrentQualityChangesGroups()
    expected_variant_names = ['current_global_stack_left_variant_name', 'current_global_stack_right_variant_name']
    expected_material_base_files = ['current_global_stack_left_material_base_file', 'current_global_stack_right_material_base_file']
    expected_is_enabled = [True, True]
    container_tree.machines['current_global_stack'].getQualityChangesGroups.assert_called_with(expected_variant_names, expected_material_base_files, expected_is_enabled)
    assert result == container_tree.machines['current_global_stack'].getQualityChangesGroups.return_value