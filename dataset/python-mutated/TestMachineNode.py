from unittest.mock import patch, MagicMock
import pytest
from UM.Settings.Interfaces import ContainerInterface
from cura.Machines.MachineNode import MachineNode
metadata_dict = {'has_materials': 'false', 'has_variants': 'true', 'has_machine_quality': 'true', 'quality_definition': 'test_quality_definition', 'exclude_materials': ['excluded_material_1', 'excluded_material_2'], 'preferred_variant_name': 'beautiful_nozzle', 'preferred_material': 'beautiful_material', 'preferred_quality_type': 'beautiful_quality_type'}

@pytest.fixture
def container_registry():
    if False:
        i = 10
        return i + 15
    result = MagicMock()
    result.findInstanceContainersMetadata = MagicMock(return_value=[{'id': 'variant_1', 'name': 'Variant One', 'quality_type': 'normal'}, {'id': 'variant_2', 'name': 'Variant Two', 'quality_type': 'great'}])
    result.findContainersMetadata = MagicMock(return_value=[metadata_dict])
    return result

@pytest.fixture
def empty_machine_node():
    if False:
        return 10
    "Creates a machine node without anything underneath it. No sub-nodes.\n\n    For testing stuff with machine nodes without testing _loadAll(). You'll need\n    to add subnodes manually in your test.\n    "
    empty_container_registry = MagicMock()
    empty_container_registry.findContainersMetadata = MagicMock(return_value=[metadata_dict])
    empty_container_registry.findInstanceContainersMetadata = MagicMock(return_value=[])
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=empty_container_registry)):
        with patch('cura.Machines.MachineNode.MachineNode._loadAll', MagicMock()):
            return MachineNode('machine_1')

def getMetadataEntrySideEffect(*args, **kwargs):
    if False:
        while True:
            i = 10
    return metadata_dict.get(args[0])

def createMockedInstanceContainer():
    if False:
        for i in range(10):
            print('nop')
    result = MagicMock(spec=ContainerInterface)
    result.getMetaDataEntry = MagicMock(side_effect=getMetadataEntrySideEffect)
    return result

def createMachineNode(container_id, container_registry):
    if False:
        for i in range(10):
            print('nop')
    with patch('cura.Machines.MachineNode.VariantNode'):
        with patch('cura.Machines.MachineNode.QualityNode'):
            with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
                return MachineNode(container_id)

def test_machineNodeInit(container_registry):
    if False:
        for i in range(10):
            print('nop')
    machine_node = createMachineNode('machine_1', container_registry)
    assert 'Variant One' in machine_node.variants
    assert 'Variant Two' in machine_node.variants
    assert len(machine_node.variants) == 2

def test_metadataProperties(container_registry):
    if False:
        for i in range(10):
            print('nop')
    node = createMachineNode('machine_1', container_registry)
    assert not node.has_materials
    assert node.has_variants
    assert node.has_machine_quality
    assert node.quality_definition == metadata_dict['quality_definition']
    assert node.exclude_materials == metadata_dict['exclude_materials']
    assert node.preferred_variant_name == metadata_dict['preferred_variant_name']
    assert node.preferred_material == metadata_dict['preferred_material']
    assert node.preferred_quality_type == metadata_dict['preferred_quality_type']

def test_getQualityGroupsBothExtrudersAvailable(empty_machine_node):
    if False:
        while True:
            i = 10
    'Test getting quality groups when there are quality profiles available for\n\n    the requested configurations on two extruders.\n    '
    extruder_0_node = MagicMock(quality_type='quality_type_1')
    extruder_1_node = MagicMock(quality_type='quality_type_1')
    empty_machine_node.variants = {'variant_1': MagicMock(materials={'material_1': MagicMock(qualities={'quality_1': extruder_0_node})}), 'variant_2': MagicMock(materials={'material_2': MagicMock(qualities={'quality_2': extruder_1_node})})}
    global_node = MagicMock(container=MagicMock(id='global_quality_container'), getMetaDataEntry=lambda _, __: 'Global Quality Profile Name')
    empty_machine_node.global_qualities = {'quality_type_1': global_node}
    result = empty_machine_node.getQualityGroups(['variant_1', 'variant_2'], ['material_1', 'material_2'], [True, True])
    assert 'quality_type_1' in result, 'This quality type was available for both extruders.'
    assert result['quality_type_1'].node_for_global == global_node
    assert result['quality_type_1'].nodes_for_extruders[0] == extruder_0_node
    assert result['quality_type_1'].nodes_for_extruders[1] == extruder_1_node
    assert result['quality_type_1'].name == global_node.getMetaDataEntry('name', 'Unnamed Profile')
    assert result['quality_type_1'].quality_type == 'quality_type_1'

def test_getQualityGroupsAvailability(empty_machine_node):
    if False:
        while True:
            i = 10
    'Test the "is_available" flag on quality groups.\n\n    If a profile is available for a quality type on an extruder but not on all\n    extruders, there should be a quality group for it but it should not be made\n    available.\n    '
    extruder_0_both = MagicMock(quality_type='quality_type_both')
    extruder_1_both = MagicMock(quality_type='quality_type_both')
    extruder_0_exclusive = MagicMock(quality_type='quality_type_0')
    extruder_1_exclusive = MagicMock(quality_type='quality_type_1')
    empty_machine_node.variants = {'variant_1': MagicMock(materials={'material_1': MagicMock(qualities={'quality_0_both': extruder_0_both, 'quality_0_exclusive': extruder_0_exclusive})}), 'variant_2': MagicMock(materials={'material_2': MagicMock(qualities={'quality_1_both': extruder_1_both, 'quality_1_exclusive': extruder_1_exclusive})})}
    global_both = MagicMock(container=MagicMock(id='global_quality_both'), getMetaDataEntry=lambda _, __: 'Global Quality Both')
    global_0 = MagicMock(container=MagicMock(id='global_quality_0'), getMetaDataEntry=lambda _, __: 'Global Quality 0 Exclusive')
    global_1 = MagicMock(container=MagicMock(id='global_quality_1'), getMetaDataEntry=lambda _, __: 'Global Quality 1 Exclusive')
    empty_machine_node.global_qualities = {'quality_type_both': global_both, 'quality_type_0': global_0, 'quality_type_1': global_1}
    result = empty_machine_node.getQualityGroups(['variant_1', 'variant_2'], ['material_1', 'material_2'], [True, True])
    assert 'quality_type_both' in result, 'This quality type was available on both extruders.'
    assert result['quality_type_both'].is_available, 'This quality type was available on both extruders and thus should be made available.'
    assert 'quality_type_0' in result, "This quality type was available for one of the extruders, and so there must be a group for it (even though it's unavailable)."
    assert not result['quality_type_0'].is_available, "This quality type was only available for one of the extruders and thus can't be activated."
    assert 'quality_type_1' in result, "This quality type was available for one of the extruders, and so there must be a group for it (even though it's unavailable)."
    assert not result['quality_type_1'].is_available, "This quality type was only available for one of the extruders and thus can't be activated."