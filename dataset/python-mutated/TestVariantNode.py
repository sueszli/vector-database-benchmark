import collections
from unittest.mock import patch, MagicMock
import pytest
from cura.Machines.VariantNode import VariantNode
import copy
instance_container_metadata_dict = {'fdmprinter': {'no_variant': [{'base_file': 'material_1', 'id': 'material_1'}, {'base_file': 'material_2', 'id': 'material_2'}]}, 'machine_1': {'no_variant': [{'base_file': 'material_1', 'id': 'material_1'}, {'base_file': 'material_2', 'id': 'material_2'}], 'Variant One': [{'base_file': 'material_1', 'id': 'material_1'}, {'base_file': 'material_2', 'id': 'material_2'}]}}
material_node_added_test_data = [({'type': 'Not a material'}, ['material_1', 'material_2']), ({'type': 'material', 'base_file': 'material_3'}, ['material_1', 'material_2']), ({'type': 'material', 'base_file': 'material_4', 'definition': 'machine_3'}, ['material_1', 'material_2']), ({'type': 'material', 'base_file': 'material_4', 'definition': 'machine_1'}, ['material_1', 'material_2', 'material_4']), ({'type': 'material', 'base_file': 'material_4', 'definition': 'machine_1', 'variant_name': 'Variant Three'}, ['material_1', 'material_2']), ({'type': 'material', 'base_file': 'material_4', 'definition': 'machine_1', 'variant_name': 'Variant One'}, ['material_1', 'material_2', 'material_4'])]
material_node_update_test_data = [({'type': 'material', 'base_file': 'material_1', 'definition': 'machine_1', 'variant_name': 'Variant One'}, ['material_1'], ['material_2']), ({'type': 'material', 'base_file': 'material_1', 'definition': 'fdmprinter', 'variant_name': 'Variant One'}, [], ['material_2', 'material_1']), ({'type': 'material', 'base_file': 'material_1', 'definition': 'machine_2', 'variant_name': 'Variant One'}, [], ['material_2', 'material_1'])]
metadata_dict = {}

def getMetadataEntrySideEffect(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return metadata_dict.get(args[0])

def getInstanceContainerSideEffect(*args, **kwargs):
    if False:
        while True:
            i = 10
    variant = kwargs.get('variant')
    definition = kwargs.get('definition')
    if variant is not None:
        return instance_container_metadata_dict.get(definition).get(variant)
    return instance_container_metadata_dict.get(definition).get('no_variant')

@pytest.fixture
def machine_node():
    if False:
        return 10
    mocked_machine_node = MagicMock()
    mocked_machine_node.container_id = 'machine_1'
    mocked_machine_node.isExcludedMaterial = MagicMock(return_value=False)
    mocked_machine_node.preferred_material = 'preferred_material'
    return mocked_machine_node

@pytest.fixture
def empty_variant_node(machine_node):
    if False:
        i = 10
        return i + 15
    'Constructs a variant node without any subnodes.\n\n    This is useful for performing tests on VariantNode without being dependent\n    on how _loadAll works.\n    '
    container_registry = MagicMock(findContainersMetadata=MagicMock(return_value=[{'name': 'test variant name'}]))
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with patch('cura.Machines.VariantNode.VariantNode._loadAll', MagicMock()):
            result = VariantNode('test_variant', machine=machine_node)
    return result

@pytest.fixture
def container_registry():
    if False:
        for i in range(10):
            print('nop')
    result = MagicMock()
    result.findInstanceContainersMetadata = MagicMock(side_effect=getInstanceContainerSideEffect)
    result.findContainersMetadata = MagicMock(return_value=[{'name': 'Variant One'}])
    return result

def createMockedInstanceContainer():
    if False:
        print('Hello World!')
    result = MagicMock()
    result.getMetaDataEntry = MagicMock(side_effect=getMetadataEntrySideEffect)
    return result

def createVariantNode(container_id, machine_node, container_registry):
    if False:
        print('Hello World!')
    with patch('cura.Machines.VariantNode.MaterialNode'):
        with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
            return VariantNode(container_id, machine_node)

def test_variantNodeInit(container_registry, machine_node):
    if False:
        i = 10
        return i + 15
    node = createVariantNode('variant_1', machine_node, container_registry)
    assert 'material_1' in node.materials
    assert 'material_2' in node.materials
    assert len(node.materials) == 2

def test_variantNodeInit_excludedMaterial(container_registry, machine_node):
    if False:
        return 10
    machine_node.exclude_materials = ['material_1']
    machine_node.isExcludedMaterial = MagicMock(side_effect=lambda material: material['id'] == 'material_1')
    node = createVariantNode('variant_1', machine_node, container_registry)
    assert 'material_1' not in node.materials
    assert 'material_2' in node.materials
    assert len(node.materials) == 1

@pytest.mark.parametrize('metadata,material_result_list', material_node_added_test_data)
def test_materialAdded(container_registry, machine_node, metadata, material_result_list):
    if False:
        return 10
    variant_node = createVariantNode('machine_1', machine_node, container_registry)
    machine_node.exclude_materials = ['material_3']
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with patch('cura.Machines.VariantNode.MaterialNode'):
            with patch.dict(metadata_dict, metadata):
                mocked_container = createMockedInstanceContainer()
                variant_node._materialAdded(mocked_container)
    assert len(material_result_list) == len(variant_node.materials)
    for name in material_result_list:
        assert name in variant_node.materials

@pytest.mark.parametrize('metadata,changed_material_list,unchanged_material_list', material_node_update_test_data)
def test_materialAdded_update(container_registry, machine_node, metadata, changed_material_list, unchanged_material_list):
    if False:
        while True:
            i = 10
    variant_node = createVariantNode('machine_1', machine_node, container_registry)
    original_material_nodes = copy.copy(variant_node.materials)
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with patch('cura.Machines.VariantNode.MaterialNode'):
            with patch.dict(metadata_dict, metadata):
                mocked_container = createMockedInstanceContainer()
                variant_node._materialAdded(mocked_container)
    for key in unchanged_material_list:
        assert original_material_nodes[key] == variant_node.materials[key]
    for key in changed_material_list:
        assert original_material_nodes[key] != variant_node.materials[key]

def test_preferredMaterialExactMatch(empty_variant_node):
    if False:
        return 10
    'Tests the preferred material when the exact base file is available in the\n\n    materials list for this node.\n    '
    empty_variant_node.materials = {'some_different_material': MagicMock(getMetaDataEntry=lambda x: 3), 'preferred_material': MagicMock(getMetaDataEntry=lambda x: 3)}
    empty_variant_node.machine.preferred_material = 'preferred_material'
    assert empty_variant_node.preferredMaterial(approximate_diameter=3) == empty_variant_node.materials['preferred_material'], "It should match exactly on this one since it's the preferred material."

def test_preferredMaterialSubmaterial(empty_variant_node):
    if False:
        while True:
            i = 10
    'Tests the preferred material when a submaterial is available in the\n\n    materials list for this node.\n    '
    empty_variant_node.materials = {'some_different_material': MagicMock(getMetaDataEntry=lambda x: 3), 'preferred_material_base_file_aa04': MagicMock(getMetaDataEntry=lambda x: 3)}
    empty_variant_node.machine.preferred_material = 'preferred_material_base_file_aa04'
    assert empty_variant_node.preferredMaterial(approximate_diameter=3) == empty_variant_node.materials['preferred_material_base_file_aa04'], 'It should match on the submaterial just as well.'

def test_preferredMaterialDiameter(empty_variant_node):
    if False:
        return 10
    'Tests the preferred material matching on the approximate diameter of the filament.\n    '
    empty_variant_node.materials = {'some_different_material': MagicMock(getMetaDataEntry=lambda x: 3), 'preferred_material_wrong_diameter': MagicMock(getMetaDataEntry=lambda x: 2), 'preferred_material_correct_diameter': MagicMock(getMetaDataEntry=lambda x: 3)}
    empty_variant_node.machine.preferred_material = 'preferred_material_correct_diameter'
    assert empty_variant_node.preferredMaterial(approximate_diameter=3) == empty_variant_node.materials['preferred_material_correct_diameter'], 'It should match only on the material with correct diameter.'

def test_preferredMaterialDiameterNoMatch(empty_variant_node):
    if False:
        while True:
            i = 10
    'Tests the preferred material matching on a different material if the diameter is wrong.'
    empty_variant_node.materials = collections.OrderedDict()
    empty_variant_node.materials['some_different_material'] = MagicMock(getMetaDataEntry=lambda x: 3)
    empty_variant_node.materials['preferred_material'] = MagicMock(getMetaDataEntry=lambda x: 2)
    assert empty_variant_node.preferredMaterial(approximate_diameter=3) == empty_variant_node.materials['some_different_material'], 'It should match on another material with the correct diameter if the preferred one is unavailable.'

def test_preferredMaterialDiameterPreference(empty_variant_node):
    if False:
        return 10
    'Tests that the material diameter is considered more important to match than\n    the preferred diameter.\n    '
    empty_variant_node.materials = collections.OrderedDict()
    empty_variant_node.materials['some_different_material'] = MagicMock(getMetaDataEntry=lambda x: 2)
    empty_variant_node.materials['preferred_material'] = MagicMock(getMetaDataEntry=lambda x: 2)
    empty_variant_node.materials['not_preferred_but_correct_diameter'] = MagicMock(getMetaDataEntry=lambda x: 3)
    assert empty_variant_node.preferredMaterial(approximate_diameter=3) == empty_variant_node.materials['not_preferred_but_correct_diameter'], "It should match on the correct diameter even if it's not the preferred one."