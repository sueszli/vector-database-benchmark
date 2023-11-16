import pytest
import unittest.mock
import cura.Settings.CuraContainerStack
import UM.Settings.ContainerRegistry
import UM.Settings.ContainerStack
from UM.Settings.DefinitionContainer import DefinitionContainer
from UM.Settings.InstanceContainer import InstanceContainer
from cura.Settings import Exceptions
from cura.Settings.Exceptions import InvalidContainerError, InvalidOperationError
from cura.Settings.ExtruderManager import ExtruderManager
from cura.Settings.cura_empty_instance_containers import empty_container

def getInstanceContainer(container_type) -> InstanceContainer:
    if False:
        i = 10
        return i + 15
    'Gets an instance container with a specified container type.\n\n    :param container_type: The type metadata for the instance container.\n    :return: An instance container instance.\n    '
    container = InstanceContainer(container_id='InstanceContainer')
    container.setMetaDataEntry('type', container_type)
    return container

class DefinitionContainerSubClass(DefinitionContainer):

    def __init__(self):
        if False:
            return 10
        super().__init__(container_id='SubDefinitionContainer')

class InstanceContainerSubClass(InstanceContainer):

    def __init__(self, container_type):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(container_id='SubInstanceContainer')
        self.setMetaDataEntry('type', container_type)

def test_addContainer(extruder_stack):
    if False:
        i = 10
        return i + 15
    'Tests whether adding a container is properly forbidden.'
    with pytest.raises(InvalidOperationError):
        extruder_stack.addContainer(unittest.mock.MagicMock())

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='wrong container type'), getInstanceContainer(container_type='material'), DefinitionContainer(container_id='wrong class')])
def test_constrainUserChangesInvalid(container, extruder_stack):
    if False:
        return 10
    with pytest.raises(InvalidContainerError):
        extruder_stack.userChanges = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='user'), InstanceContainerSubClass(container_type='user')])
def test_constrainUserChangesValid(container, extruder_stack):
    if False:
        while True:
            i = 10
    extruder_stack.userChanges = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='wrong container type'), getInstanceContainer(container_type='material'), DefinitionContainer(container_id='wrong class')])
def test_constrainQualityChangesInvalid(container, extruder_stack):
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidContainerError):
        extruder_stack.qualityChanges = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='quality_changes'), InstanceContainerSubClass(container_type='quality_changes')])
def test_constrainQualityChangesValid(container, extruder_stack):
    if False:
        while True:
            i = 10
    extruder_stack.qualityChanges = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='wrong container type'), getInstanceContainer(container_type='material'), DefinitionContainer(container_id='wrong class')])
def test_constrainQualityInvalid(container, extruder_stack):
    if False:
        return 10
    with pytest.raises(InvalidContainerError):
        extruder_stack.quality = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='quality'), InstanceContainerSubClass(container_type='quality')])
def test_constrainQualityValid(container, extruder_stack):
    if False:
        for i in range(10):
            print('nop')
    extruder_stack.quality = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='wrong container type'), getInstanceContainer(container_type='quality'), DefinitionContainer(container_id='wrong class')])
def test_constrainMaterialInvalid(container, extruder_stack):
    if False:
        print('Hello World!')
    with pytest.raises(InvalidContainerError):
        extruder_stack.material = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='material'), InstanceContainerSubClass(container_type='material')])
def test_constrainMaterialValid(container, extruder_stack):
    if False:
        while True:
            i = 10
    extruder_stack.material = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='wrong container type'), getInstanceContainer(container_type='material'), DefinitionContainer(container_id='wrong class')])
def test_constrainVariantInvalid(container, extruder_stack):
    if False:
        return 10
    with pytest.raises(InvalidContainerError):
        extruder_stack.variant = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='variant'), InstanceContainerSubClass(container_type='variant')])
def test_constrainVariantValid(container, extruder_stack):
    if False:
        while True:
            i = 10
    extruder_stack.variant = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='wrong container type'), getInstanceContainer(container_type='material'), DefinitionContainer(container_id='wrong class')])
def test_constrainDefinitionChangesInvalid(container, global_stack):
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidContainerError):
        global_stack.definitionChanges = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='definition_changes'), InstanceContainerSubClass(container_type='definition_changes')])
def test_constrainDefinitionChangesValid(container, global_stack):
    if False:
        print('Hello World!')
    global_stack.definitionChanges = container

@pytest.mark.parametrize('container', [getInstanceContainer(container_type='wrong class'), getInstanceContainer(container_type='material')])
def test_constrainDefinitionInvalid(container, extruder_stack):
    if False:
        while True:
            i = 10
    with pytest.raises(InvalidContainerError):
        extruder_stack.definition = container

@pytest.mark.parametrize('container', [DefinitionContainer(container_id='DefinitionContainer'), DefinitionContainerSubClass()])
def test_constrainDefinitionValid(container, extruder_stack):
    if False:
        while True:
            i = 10
    extruder_stack.definition = container

def test_deserializeCompletesEmptyContainers(extruder_stack):
    if False:
        while True:
            i = 10
    'Tests whether deserialising completes the missing containers with empty ones.'
    extruder_stack._containers = [DefinitionContainer(container_id='definition'), extruder_stack.definitionChanges]
    with unittest.mock.patch('UM.Settings.ContainerStack.ContainerStack.deserialize', unittest.mock.MagicMock()):
        extruder_stack.deserialize('')
    assert len(extruder_stack.getContainers()) == len(cura.Settings.CuraContainerStack._ContainerIndexes.IndexTypeMap)
    for container_type_index in cura.Settings.CuraContainerStack._ContainerIndexes.IndexTypeMap:
        if container_type_index in (cura.Settings.CuraContainerStack._ContainerIndexes.Definition, cura.Settings.CuraContainerStack._ContainerIndexes.DefinitionChanges):
            continue
        assert extruder_stack.getContainer(container_type_index) == empty_container

def test_deserializeRemovesWrongInstanceContainer(extruder_stack):
    if False:
        return 10
    'Tests whether an instance container with the wrong type gets removed when deserialising.'
    extruder_stack._containers[cura.Settings.CuraContainerStack._ContainerIndexes.Quality] = getInstanceContainer(container_type='wrong type')
    extruder_stack._containers[cura.Settings.CuraContainerStack._ContainerIndexes.Definition] = DefinitionContainer(container_id='some definition')
    with unittest.mock.patch('UM.Settings.ContainerStack.ContainerStack.deserialize', unittest.mock.MagicMock()):
        extruder_stack.deserialize('')
    assert extruder_stack.quality == extruder_stack._empty_instance_container

def test_deserializeRemovesWrongContainerClass(extruder_stack):
    if False:
        for i in range(10):
            print('nop')
    'Tests whether a container with the wrong class gets removed when deserialising.'
    extruder_stack._containers[cura.Settings.CuraContainerStack._ContainerIndexes.Quality] = DefinitionContainer(container_id='wrong class')
    extruder_stack._containers[cura.Settings.CuraContainerStack._ContainerIndexes.Definition] = DefinitionContainer(container_id='some definition')
    with unittest.mock.patch('UM.Settings.ContainerStack.ContainerStack.deserialize', unittest.mock.MagicMock()):
        extruder_stack.deserialize('')
    assert extruder_stack.quality == extruder_stack._empty_instance_container

def test_deserializeWrongDefinitionClass(extruder_stack):
    if False:
        for i in range(10):
            print('nop')
    'Tests whether an instance container in the definition spot results in an error.'
    extruder_stack._containers[cura.Settings.CuraContainerStack._ContainerIndexes.Definition] = getInstanceContainer(container_type='definition')
    with unittest.mock.patch('UM.Settings.ContainerStack.ContainerStack.deserialize', unittest.mock.MagicMock()):
        with pytest.raises(UM.Settings.ContainerStack.InvalidContainerStackError):
            extruder_stack.deserialize('')

def test_deserializeMoveInstanceContainer(extruder_stack):
    if False:
        i = 10
        return i + 15
    'Tests whether an instance container with the wrong type is moved into the correct slot by deserialising.'
    extruder_stack._containers[cura.Settings.CuraContainerStack._ContainerIndexes.Quality] = getInstanceContainer(container_type='material')
    extruder_stack._containers[cura.Settings.CuraContainerStack._ContainerIndexes.Definition] = DefinitionContainer(container_id='some definition')
    with unittest.mock.patch('UM.Settings.ContainerStack.ContainerStack.deserialize', unittest.mock.MagicMock()):
        extruder_stack.deserialize('')
    assert extruder_stack.quality == empty_container
    assert extruder_stack.material != empty_container

def test_deserializeMoveDefinitionContainer(extruder_stack):
    if False:
        while True:
            i = 10
    'Tests whether a definition container in the wrong spot is moved into the correct spot by deserialising.'
    extruder_stack._containers[cura.Settings.CuraContainerStack._ContainerIndexes.Material] = DefinitionContainer(container_id='some definition')
    with unittest.mock.patch('UM.Settings.ContainerStack.ContainerStack.deserialize', unittest.mock.MagicMock()):
        extruder_stack.deserialize('')
    assert extruder_stack.material == empty_container
    assert extruder_stack.definition != empty_container

def test_getPropertyFallThrough(global_stack, extruder_stack):
    if False:
        while True:
            i = 10
    'Tests whether getProperty properly applies the stack-like behaviour on its containers.'
    ExtruderManager._ExtruderManager__instance = unittest.mock.MagicMock()
    mock_layer_heights = {}
    mock_no_settings = {}
    container_indices = cura.Settings.CuraContainerStack._ContainerIndexes
    for (type_id, type_name) in container_indices.IndexTypeMap.items():
        container = unittest.mock.MagicMock()
        container.getProperty = lambda key, property, context=None, type_id=type_id: type_id if key == 'layer_height' and property == 'value' else None if property != 'settable_per_extruder' else '-1'
        container.hasProperty = lambda key, property: key == 'layer_height'
        container.getMetaDataEntry = unittest.mock.MagicMock(return_value=type_name)
        mock_layer_heights[type_id] = container
        container = unittest.mock.MagicMock()
        container.getProperty = unittest.mock.MagicMock(return_value=None)
        container.hasProperty = unittest.mock.MagicMock(return_value=False)
        container.getMetaDataEntry = unittest.mock.MagicMock(return_value=type_name)
        mock_no_settings[type_id] = container
    extruder_stack.userChanges = mock_no_settings[container_indices.UserChanges]
    extruder_stack.qualityChanges = mock_no_settings[container_indices.QualityChanges]
    extruder_stack.quality = mock_no_settings[container_indices.Quality]
    extruder_stack.material = mock_no_settings[container_indices.Material]
    extruder_stack.variant = mock_no_settings[container_indices.Variant]
    with unittest.mock.patch('cura.Settings.CuraContainerStack.DefinitionContainer', unittest.mock.MagicMock):
        extruder_stack.definition = mock_layer_heights[container_indices.Definition]
    extruder_stack.setNextStack(global_stack)
    assert extruder_stack.getProperty('layer_height', 'value') == container_indices.Definition
    extruder_stack.variant = mock_layer_heights[container_indices.Variant]
    assert extruder_stack.getProperty('layer_height', 'value') == container_indices.Variant
    extruder_stack.material = mock_layer_heights[container_indices.Material]
    assert extruder_stack.getProperty('layer_height', 'value') == container_indices.Material
    extruder_stack.quality = mock_layer_heights[container_indices.Quality]
    assert extruder_stack.getProperty('layer_height', 'value') == container_indices.Quality
    extruder_stack.qualityChanges = mock_layer_heights[container_indices.QualityChanges]
    assert extruder_stack.getProperty('layer_height', 'value') == container_indices.QualityChanges
    extruder_stack.userChanges = mock_layer_heights[container_indices.UserChanges]
    assert extruder_stack.getProperty('layer_height', 'value') == container_indices.UserChanges

def test_insertContainer(extruder_stack):
    if False:
        while True:
            i = 10
    'Tests whether inserting a container is properly forbidden.'
    with pytest.raises(InvalidOperationError):
        extruder_stack.insertContainer(0, unittest.mock.MagicMock())

def test_removeContainer(extruder_stack):
    if False:
        i = 10
        return i + 15
    'Tests whether removing a container is properly forbidden.'
    with pytest.raises(InvalidOperationError):
        extruder_stack.removeContainer(unittest.mock.MagicMock())

@pytest.mark.parametrize('key,              property,         value', [('layer_height', 'value', 0.1337), ('foo', 'value', 100), ('support_enabled', 'value', True), ('layer_height', 'default_value', 0.1337), ('layer_height', 'is_bright_pink', 'of course')])
def test_setPropertyUser(key, property, value, extruder_stack):
    if False:
        for i in range(10):
            print('nop')
    user_changes = unittest.mock.MagicMock()
    user_changes.getMetaDataEntry = unittest.mock.MagicMock(return_value='user')
    extruder_stack.userChanges = user_changes
    extruder_stack.setProperty(key, property, value)
    extruder_stack.userChanges.setProperty.assert_called_once_with(key, property, value, None, False)

def test_setEnabled(extruder_stack):
    if False:
        for i in range(10):
            print('nop')
    extruder_stack.setEnabled(True)
    assert extruder_stack.isEnabled
    extruder_stack.setEnabled(False)
    assert not extruder_stack.isEnabled

def test_getPropertyWithoutGlobal(extruder_stack):
    if False:
        for i in range(10):
            print('nop')
    assert extruder_stack.getNextStack() is None
    with pytest.raises(Exceptions.NoGlobalStackError):
        extruder_stack.getProperty('whatever', 'value')

def test_getMachineDefinitionWithoutGlobal(extruder_stack):
    if False:
        print('Hello World!')
    assert extruder_stack.getNextStack() is None
    with pytest.raises(Exceptions.NoGlobalStackError):
        extruder_stack._getMachineDefinition()

def test_getMachineDefinition(extruder_stack):
    if False:
        while True:
            i = 10
    mocked_next_stack = unittest.mock.MagicMock()
    mocked_next_stack._getMachineDefinition = unittest.mock.MagicMock(return_value='ZOMG')
    extruder_stack.getNextStack = unittest.mock.MagicMock(return_value=mocked_next_stack)
    assert extruder_stack._getMachineDefinition() == 'ZOMG'