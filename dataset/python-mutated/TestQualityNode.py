from unittest.mock import patch, MagicMock
import pytest
from cura.Machines.QualityNode import QualityNode
metadatas = [{'id': 'matching_intent', 'type': 'intent', 'definition': 'correct_definition', 'variant': 'correct_variant', 'material': 'correct_material', 'quality_type': 'correct_quality_type'}, {'id': 'matching_intent_2', 'type': 'intent', 'definition': 'correct_definition', 'variant': 'correct_variant', 'material': 'correct_material', 'quality_type': 'correct_quality_type'}, {'id': 'bad_type', 'type': 'quality', 'definition': 'correct_definition', 'variant': 'correct_variant', 'material': 'correct_material', 'quality_type': 'correct_quality_type'}, {'id': 'bad_definition', 'type': 'intent', 'definition': 'wrong_definition', 'variant': 'correct_variant', 'material': 'correct_material', 'quality_type': 'correct_quality_type'}, {'id': 'bad_variant', 'type': 'intent', 'definition': 'correct_definition', 'variant': 'wrong_variant', 'material': 'correct_material', 'quality_type': 'correct_quality_type'}, {'id': 'bad_material', 'type': 'intent', 'definition': 'correct_definition', 'variant': 'correct_variant', 'material': 'wrong_material', 'quality_type': 'correct_quality_type'}, {'id': 'bad_quality', 'type': 'intent', 'definition': 'correct_definition', 'variant': 'correct_variant', 'material': 'correct_material', 'quality_type': 'wrong_quality_type'}, {'id': 'quality_1', 'quality_type': 'correct_quality_type', 'material': 'correct_material'}]

@pytest.fixture
def container_registry():
    if False:
        for i in range(10):
            print('nop')
    result = MagicMock()

    def findContainersMetadata(**kwargs):
        if False:
            while True:
                i = 10
        return [metadata for metadata in metadatas if kwargs.items() <= metadata.items()]
    result.findContainersMetadata = findContainersMetadata
    result.findInstanceContainersMetadata = findContainersMetadata
    return result

def test_qualityNode_machine_1(container_registry):
    if False:
        print('Hello World!')
    material_node = MagicMock()
    material_node.variant.machine.quality_definition = 'correct_definition'
    material_node.variant.variant_name = 'correct_variant'
    with patch('cura.Machines.QualityNode.IntentNode'):
        with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
            node = QualityNode('quality_1', material_node)
    assert len(node.intents) == 3
    assert 'matching_intent' in node.intents
    assert 'matching_intent_2' in node.intents
    assert 'empty_intent' in node.intents