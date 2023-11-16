from unittest.mock import patch, MagicMock
import pytest
from cura.Machines.MaterialNode import MaterialNode
instance_container_metadata_dict = {'fdmprinter': {'no_variant': [{'id': 'quality_1', 'material': 'material_1'}]}, 'machine_1': {'variant_1': {'material_1': [{'id': 'quality_2', 'material': 'material_1'}, {'id': 'quality_3', 'material': 'material_1'}]}}}
metadata_dict = {}

def getMetadataEntrySideEffect(*args, **kwargs):
    if False:
        while True:
            i = 10
    return metadata_dict.get(args[0])

def createMockedInstanceContainer(container_id):
    if False:
        for i in range(10):
            print('nop')
    result = MagicMock()
    result.getId = MagicMock(return_value=container_id)
    result.getMetaDataEntry = MagicMock(side_effect=getMetadataEntrySideEffect)
    return result

def getInstanceContainerSideEffect(*args, **kwargs):
    if False:
        return 10
    variant = kwargs.get('variant')
    definition = kwargs.get('definition')
    type = kwargs.get('type')
    material = kwargs.get('material')
    if material is not None and variant is not None:
        definition_dict = instance_container_metadata_dict.get(definition)
        variant_dict = definition_dict.get(variant)
        material_dict = variant_dict.get('material_1')
        return material_dict
    if type == 'quality':
        if variant is None:
            return instance_container_metadata_dict.get(definition).get('no_variant')
        else:
            return instance_container_metadata_dict.get(definition).get(variant).get('material_1')
    if definition is None:
        return [{'id': 'material_1', 'material': 'material_1'}]
    return instance_container_metadata_dict.get(definition).get('no_variant')

@pytest.fixture
def container_registry():
    if False:
        for i in range(10):
            print('nop')
    result = MagicMock()
    result.findInstanceContainersMetadata = MagicMock(side_effect=getInstanceContainerSideEffect)
    result.findContainersMetadata = MagicMock(return_value=[{'base_file': 'material_1', 'material': 'test_material_type', 'GUID': 'omg zomg', 'brand': 'test_brand'}])
    return result

def test_materialNodeInit_noMachineQuality(container_registry):
    if False:
        return 10
    variant_node = MagicMock()
    variant_node.variant_name = 'variant_1'
    variant_node.machine.has_machine_quality = False
    with patch('cura.Machines.MaterialNode.QualityNode'):
        with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
            node = MaterialNode('material_1', variant_node)
    assert len(node.qualities) == 1
    assert 'quality_1' in node.qualities

def test_materialNodeInit_MachineQuality(container_registry):
    if False:
        i = 10
        return i + 15
    variant_node = MagicMock()
    variant_node.variant_name = 'variant_1'
    variant_node.machine.has_machine_quality = True
    variant_node.machine.quality_definition = 'machine_1'
    with patch('cura.Machines.MaterialNode.QualityNode'):
        with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
            node = MaterialNode('material_1', variant_node)
    assert len(node.qualities) == 2
    assert 'quality_2' in node.qualities
    assert 'quality_3' in node.qualities

def test_onRemoved_wrongContainer(container_registry):
    if False:
        for i in range(10):
            print('nop')
    variant_node = MagicMock()
    variant_node.variant_name = 'variant_1'
    variant_node.machine.has_machine_quality = True
    variant_node.machine.quality_definition = 'machine_1'
    variant_node.materials = {'material_1': MagicMock()}
    with patch('cura.Machines.MaterialNode.QualityNode'):
        with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
            node = MaterialNode('material_1', variant_node)
    container = createMockedInstanceContainer('material_2')
    node._onRemoved(container)
    assert 'material_1' in variant_node.materials

def test_onRemoved_rightContainer(container_registry):
    if False:
        return 10
    variant_node = MagicMock()
    variant_node.variant_name = 'variant_1'
    variant_node.machine.has_machine_quality = True
    variant_node.machine.quality_definition = 'machine_1'
    with patch('cura.Machines.MaterialNode.QualityNode'):
        with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
            node = MaterialNode('material_1', variant_node)
            container = createMockedInstanceContainer('material_1')
            variant_node.materials = {'material_1': MagicMock()}
            node._onRemoved(container)
    assert 'material_1' not in variant_node.materials

def test_onMetadataChanged(container_registry):
    if False:
        return 10
    variant_node = MagicMock()
    variant_node.variant_name = 'variant_1'
    variant_node.machine.has_machine_quality = True
    variant_node.machine.quality_definition = 'machine_1'
    with patch('cura.Machines.MaterialNode.QualityNode'):
        with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
            node = MaterialNode('material_1', variant_node)
    node._loadAll = MagicMock()
    container = createMockedInstanceContainer('material_1')
    container.getMetaData = MagicMock(return_value={'base_file': 'new_base_file', 'material': 'new_material_type', 'GUID': 'new_guid'})
    node._onMetadataChanged(container)
    assert node.material_type == 'new_material_type'
    assert node.guid == 'new_guid'
    assert node.base_file == 'new_base_file'

def test_onMetadataChanged_wrongContainer(container_registry):
    if False:
        for i in range(10):
            print('nop')
    variant_node = MagicMock()
    variant_node.variant_name = 'variant_1'
    variant_node.machine.has_machine_quality = True
    variant_node.machine.quality_definition = 'machine_1'
    with patch('cura.Machines.MaterialNode.QualityNode'):
        with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
            node = MaterialNode('material_1', variant_node)
    node._loadAll = MagicMock()
    container = createMockedInstanceContainer('material_2')
    container.getMetaData = MagicMock(return_value={'base_file': 'new_base_file', 'material': 'new_material_type', 'GUID': 'new_guid'})
    node._onMetadataChanged(container)
    assert node.material_type == 'test_material_type'
    assert node.guid == 'omg zomg'
    assert node.base_file == 'material_1'