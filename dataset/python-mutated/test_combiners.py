import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
import torch
from ludwig.combiners.combiners import ComparatorCombiner, ConcatCombiner, ProjectAggregateCombiner, SequenceCombiner, SequenceConcatCombiner, TabNetCombiner, TabTransformerCombiner, TransformerCombiner
from ludwig.constants import CATEGORY, ENCODER_OUTPUT, ENCODER_OUTPUT_STATE, TYPE
from ludwig.encoders.registry import get_sequence_encoder_registry
from ludwig.schema.combiners.comparator import ComparatorCombinerConfig
from ludwig.schema.combiners.concat import ConcatCombinerConfig
from ludwig.schema.combiners.project_aggregate import ProjectAggregateCombinerConfig
from ludwig.schema.combiners.sequence import SequenceCombinerConfig
from ludwig.schema.combiners.sequence_concat import SequenceConcatCombinerConfig
from ludwig.schema.combiners.tab_transformer import TabTransformerCombinerConfig
from ludwig.schema.combiners.tabnet import TabNetCombinerConfig
from ludwig.schema.combiners.transformer import TransformerCombinerConfig
from ludwig.schema.utils import load_config
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('ludwig').setLevel(logging.INFO)
DEVICE = get_torch_device()
BATCH_SIZE = 16
SEQ_SIZE = 12
HIDDEN_SIZE = 24
OTHER_HIDDEN_SIZE = 32
OUTPUT_SIZE = 8
BASE_OUTPUT_SIZE = 16
NUM_FILTERS = 20
RANDOM_SEED = 1919

class PseudoInputFeature:

    def __init__(self, feature_name, output_shape, feature_type=None):
        if False:
            return 10
        self.name = feature_name
        self._output_shape = output_shape
        self.feature_type = feature_type

    def type(self):
        if False:
            return 10
        return self.feature_type

    @property
    def output_shape(self):
        if False:
            while True:
                i = 10
        return torch.Size(self._output_shape[1:])

def check_combiner_output(combiner, combiner_output, batch_size):
    if False:
        print('Hello World!')
    assert hasattr(combiner, 'input_dtype')
    assert hasattr(combiner, 'output_shape')
    assert isinstance(combiner_output, dict)
    assert 'combiner_output' in combiner_output
    assert combiner_output['combiner_output'].shape == (batch_size, *combiner.output_shape)

@pytest.fixture
def features_to_test(feature_list: List[Tuple[str, list]]) -> Tuple[dict, dict]:
    if False:
        return 10
    set_random_seed(RANDOM_SEED)
    encoder_outputs = {}
    input_features = {}
    for i in range(len(feature_list)):
        feature_name = f'feature_{i:02d}'
        encoder_outputs[feature_name] = {ENCODER_OUTPUT: torch.randn(feature_list[i][1], dtype=torch.float32, device=DEVICE)}
        input_features[feature_name] = PseudoInputFeature(feature_name, feature_list[i][1], feature_list[i][0])
    return (encoder_outputs, input_features)

@pytest.fixture
def encoder_outputs():
    if False:
        while True:
            i = 10
    set_random_seed(RANDOM_SEED)
    encoder_outputs = {}
    input_features = OrderedDict()
    shapes_list = [[BATCH_SIZE, HIDDEN_SIZE], [BATCH_SIZE, OTHER_HIDDEN_SIZE], [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE], [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE]]
    feature_names = ['feature_' + str(i + 1) for i in range(len(shapes_list))]
    for (feature_name, batch_shape) in zip(feature_names, shapes_list):
        encoder_outputs[feature_name] = {ENCODER_OUTPUT: torch.randn(batch_shape, dtype=torch.float32, device=DEVICE)}
        if len(batch_shape) > 2:
            encoder_outputs[feature_name][ENCODER_OUTPUT_STATE] = torch.randn([batch_shape[0], batch_shape[2]], dtype=torch.float32, device=DEVICE)
        input_features[feature_name] = PseudoInputFeature(feature_name, batch_shape)
    return (encoder_outputs, input_features)

@pytest.fixture
def encoder_comparator_outputs():
    if False:
        i = 10
        return i + 15
    encoder_outputs = {}
    input_features = {}
    shapes_list = [[BATCH_SIZE, HIDDEN_SIZE], [BATCH_SIZE, OTHER_HIDDEN_SIZE], [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE], [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE]]
    text_feature_names = ['text_feature_' + str(i + 1) for i in range(len(shapes_list))]
    image_feature_names = ['image_feature_' + str(i + 1) for i in range(len(shapes_list))]
    for (i, (feature_name, batch_shape)) in enumerate(zip(text_feature_names, shapes_list)):
        if i == 0 or i == 3:
            dot_product_shape = [batch_shape[0], BASE_OUTPUT_SIZE]
            encoder_outputs[feature_name] = {ENCODER_OUTPUT: torch.randn(dot_product_shape, dtype=torch.float32, device=DEVICE)}
            input_features[feature_name] = PseudoInputFeature(feature_name, dot_product_shape)
        else:
            encoder_outputs[feature_name] = {ENCODER_OUTPUT: torch.randn(batch_shape, dtype=torch.float32, device=DEVICE)}
            input_features[feature_name] = PseudoInputFeature(feature_name, batch_shape)
    for (i, (feature_name, batch_shape)) in enumerate(zip(image_feature_names, shapes_list)):
        if i == 0 or i == 3:
            dot_product_shape = [batch_shape[0], BASE_OUTPUT_SIZE]
            encoder_outputs[feature_name] = {ENCODER_OUTPUT: torch.randn(dot_product_shape, dtype=torch.float32, device=DEVICE)}
            input_features[feature_name] = PseudoInputFeature(feature_name, dot_product_shape)
        else:
            encoder_outputs[feature_name] = {ENCODER_OUTPUT: torch.randn(batch_shape, dtype=torch.float32, device=DEVICE)}
            input_features[feature_name] = PseudoInputFeature(feature_name, batch_shape)
    return (encoder_outputs, input_features)

@pytest.mark.parametrize('norm', [None, 'batch', 'layer', 'ghost'])
@pytest.mark.parametrize('number_inputs', [None, 1])
@pytest.mark.parametrize('flatten_inputs', [True, False])
@pytest.mark.parametrize('fc_layer', [None, [{'output_size': OUTPUT_SIZE}, {'output_size': OUTPUT_SIZE}]])
def test_concat_combiner(encoder_outputs: Tuple, fc_layer: Optional[List[Dict]], flatten_inputs: bool, number_inputs: Optional[int], norm: str) -> None:
    if False:
        print('Hello World!')
    set_random_seed(RANDOM_SEED)
    (encoder_outputs_dict, input_features_dict) = encoder_outputs
    if not flatten_inputs:
        for feature in ['feature_3', 'feature_4']:
            del encoder_outputs_dict[feature]
            del input_features_dict[feature]
        if number_inputs == 1:
            del encoder_outputs_dict['feature_2']
            del input_features_dict['feature_2']
    elif number_inputs == 1:
        for feature in ['feature_1', 'feature_2', 'feature_3']:
            del encoder_outputs_dict[feature]
            del input_features_dict[feature]
    combiner = ConcatCombiner(input_features_dict, config=load_config(ConcatCombinerConfig, fc_layers=fc_layer, flatten_inputs=flatten_inputs, norm=norm)).to(DEVICE)
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k][ENCODER_OUTPUT].shape[1:] == combiner.input_shape[k]
    combiner_output = combiner(encoder_outputs_dict)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
    if fc_layer is not None:
        target = torch.randn(combiner_output['combiner_output'].shape)
        (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_outputs_dict,), target)
        assert tpc == upc, f'Failed to update parameters. Parameters not updated: {not_updated}'

@pytest.mark.parametrize('reduce_output', [None, 'sum'])
@pytest.mark.parametrize('main_sequence_feature', [None, 'feature_3'])
def test_sequence_concat_combiner(encoder_outputs: Tuple, main_sequence_feature: Optional[str], reduce_output: Optional[str]) -> None:
    if False:
        return 10
    (encoder_outputs_dict, input_feature_dict) = encoder_outputs
    combiner = SequenceConcatCombiner(input_feature_dict, config=load_config(SequenceConcatCombinerConfig, main_sequence_feature=main_sequence_feature, reduce_output=reduce_output)).to(DEVICE)
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k][ENCODER_OUTPUT].shape[1:] == combiner.input_shape[k]
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += encoder_outputs_dict[k][ENCODER_OUTPUT].shape[-1]
    assert combiner.concatenated_shape[-1] == hidden_size
    combiner_output = combiner(encoder_outputs_dict)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)

@pytest.mark.parametrize('reduce_output', [None, 'sum'])
@pytest.mark.parametrize('encoder', get_sequence_encoder_registry())
@pytest.mark.parametrize('main_sequence_feature', [None, 'feature_3'])
def test_sequence_combiner(encoder_outputs: Tuple, main_sequence_feature: Optional[str], encoder: str, reduce_output: Optional[str]) -> None:
    if False:
        while True:
            i = 10
    set_random_seed(RANDOM_SEED)
    (encoder_outputs_dict, input_features_dict) = encoder_outputs
    combiner = SequenceCombiner(input_features_dict, config=load_config(SequenceCombinerConfig, main_sequence_feature=main_sequence_feature, encoder={TYPE: encoder}, reduce_output=reduce_output), output_size=OUTPUT_SIZE, num_fc_layers=3).to(DEVICE)
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k][ENCODER_OUTPUT].shape[1:] == combiner.input_shape[k]
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += encoder_outputs_dict[k][ENCODER_OUTPUT].shape[-1]
    assert combiner.concatenated_shape[-1] == hidden_size
    combiner_output = combiner(encoder_outputs_dict)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
    target = torch.randn(combiner_output['combiner_output'].shape)
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_outputs_dict,), target)
    assert tpc == upc, f'Failed to update parameters. Parameters not updated: {not_updated}'

@pytest.mark.parametrize('feature_list', [[('binary', [BATCH_SIZE, 1]), ('number', [BATCH_SIZE, 1])], [('binary', [BATCH_SIZE, 1]), ('number', [BATCH_SIZE, 1]), ('number', [BATCH_SIZE, 1])], [('binary', [BATCH_SIZE, 1]), ('number', [BATCH_SIZE, 12]), ('category', [BATCH_SIZE, 8])]])
@pytest.mark.parametrize('size', [4, 8])
@pytest.mark.parametrize('output_size', [6, 10])
def test_tabnet_combiner(features_to_test: Dict, size: int, output_size: int) -> None:
    if False:
        while True:
            i = 10
    set_random_seed(RANDOM_SEED)
    (encoder_outputs, input_features) = features_to_test
    combiner = TabNetCombiner(input_features, config=load_config(TabNetCombinerConfig, size=size, output_size=output_size, num_steps=3, num_total_blocks=4, num_shared_blocks=2, dropout=0.1)).to(DEVICE)
    combiner_output = combiner(encoder_outputs)
    assert 'combiner_output' in combiner_output
    assert 'attention_masks' in combiner_output
    assert 'aggregated_attention_masks' in combiner_output
    assert isinstance(combiner_output['combiner_output'], torch.Tensor)
    assert combiner_output['combiner_output'].shape == (BATCH_SIZE, output_size)
    target = torch.randn(combiner_output['combiner_output'].shape)
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_outputs,), target)
    assert tpc == upc, f'Failed to update parameters. Parameters not updated: {not_updated}'

@pytest.mark.parametrize('fc_layer', [None, [{'output_size': 64}, {'output_size': 32}]])
@pytest.mark.parametrize('entity_1', [['text_feature_1', 'text_feature_4']])
@pytest.mark.parametrize('entity_2', [['image_feature_1', 'image_feature_2']])
def test_comparator_combiner(encoder_comparator_outputs: Tuple, fc_layer: Optional[List[Dict]], entity_1: str, entity_2: str) -> None:
    if False:
        print('Hello World!')
    set_random_seed(RANDOM_SEED)
    (encoder_comparator_outputs_dict, input_features_dict) = encoder_comparator_outputs
    del encoder_comparator_outputs_dict['text_feature_2']
    del encoder_comparator_outputs_dict['image_feature_3']
    del encoder_comparator_outputs_dict['text_feature_3']
    del encoder_comparator_outputs_dict['image_feature_4']
    output_size = fc_layer[0]['output_size'] if fc_layer else 256
    combiner = ComparatorCombiner(input_features_dict, config=load_config(ComparatorCombinerConfig, entity_1=entity_1, entity_2=entity_2, fc_layers=fc_layer, output_size=output_size)).to(DEVICE)
    combiner_output = combiner(encoder_comparator_outputs_dict)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
    target = torch.randn(combiner_output['combiner_output'].shape)
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_comparator_outputs_dict,), target)
    assert tpc == upc, f'Failed to update parameters. Parameters not updated: {not_updated}'

@pytest.mark.parametrize('output_size', [8, 16])
@pytest.mark.parametrize('transformer_output_size', [4, 12])
def test_transformer_combiner(encoder_outputs: tuple, transformer_output_size: int, output_size: int) -> None:
    if False:
        while True:
            i = 10
    set_random_seed(RANDOM_SEED)
    (encoder_outputs_dict, input_feature_dict) = encoder_outputs
    combiner = TransformerCombiner(input_features=input_feature_dict, config=load_config(TransformerCombinerConfig)).to(DEVICE)
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k][ENCODER_OUTPUT].shape[1:] == combiner.input_shape[k]
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += np.prod(encoder_outputs_dict[k][ENCODER_OUTPUT].shape[1:])
    assert combiner.concatenated_shape[-1] == hidden_size
    combiner_output = combiner(encoder_outputs_dict)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
    target = torch.randn(combiner_output['combiner_output'].shape)
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_outputs_dict,), target)
    assert tpc == upc, f'Failed to update parameters. Parameters not updated: {not_updated}'

@pytest.mark.parametrize('projection_size', [8, 16])
@pytest.mark.parametrize('output_size', [8, 16])
def test_project_aggregate_combiner(encoder_outputs: tuple, projection_size: int, output_size: int) -> None:
    if False:
        i = 10
        return i + 15
    set_random_seed(RANDOM_SEED)
    (encoder_outputs_dict, input_feature_dict) = encoder_outputs
    combiner = ProjectAggregateCombiner(input_features=input_feature_dict, config=load_config(ProjectAggregateCombinerConfig, projection_size=projection_size, output_size=output_size)).to(DEVICE)
    assert isinstance(combiner.input_shape, dict)
    for k in encoder_outputs_dict:
        assert k in combiner.input_shape
        assert encoder_outputs_dict[k][ENCODER_OUTPUT].shape[1:] == combiner.input_shape[k]
    hidden_size = 0
    for k in encoder_outputs_dict:
        hidden_size += np.prod(encoder_outputs_dict[k][ENCODER_OUTPUT].shape[1:])
    assert combiner.concatenated_shape[-1] == hidden_size
    combiner_output = combiner(encoder_outputs_dict)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
    target = torch.randn(combiner_output['combiner_output'].shape)
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_outputs_dict,), target)
    assert tpc == upc, f'Failed to update parameters. Parameters not updated: {not_updated}'
PARAMETERS_IN_SELF_ATTENTION = 4
PARAMETERS_IN_TRANSFORMER_BLOCK = 16
UNEMBEDDABLE_LAYER_NORM_PARAMETERS = 2

@pytest.mark.parametrize('feature_list', [[('binary', [BATCH_SIZE, 1]), ('number', [BATCH_SIZE, 1])], [('number', [BATCH_SIZE, 1]), ('binary', [BATCH_SIZE, 1]), ('number', [BATCH_SIZE, 1])], [('binary', [BATCH_SIZE, 1]), ('number', [BATCH_SIZE, 1]), ('binary', [BATCH_SIZE, 1])]])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('reduce_output', ['concat', 'sum'])
@pytest.mark.parametrize('fc_layers', [None, [{'output_size': 256}]])
@pytest.mark.parametrize('embed_input_feature_name', [None, 64, 'add'])
def test_tabtransformer_combiner_binary_and_number_without_category(features_to_test: tuple, embed_input_feature_name: Optional[Union[int, str]], fc_layers: Optional[list], reduce_output: str, num_layers: int) -> None:
    if False:
        i = 10
        return i + 15
    set_random_seed(RANDOM_SEED)
    (encoder_outputs, input_features) = features_to_test
    combiner = TabTransformerCombiner(input_features=input_features, config=load_config(TabTransformerCombinerConfig, embed_input_feature_name=embed_input_feature_name, num_layers=num_layers, fc_layers=fc_layers, reduce_output=reduce_output)).to(DEVICE)
    combiner_output = combiner(encoder_outputs)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
    target = torch.randn(combiner_output['combiner_output'].shape)
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_outputs,), target)
    assert upc == tpc - num_layers * PARAMETERS_IN_TRANSFORMER_BLOCK - (1 if embed_input_feature_name is not None else 0), f'Failed to update parameters. Parameters not updated: {not_updated}'

@pytest.mark.parametrize('feature_list', [[('number', [BATCH_SIZE, 1]), ('category', [BATCH_SIZE, 64]), ('binary', [BATCH_SIZE, 1])], [('binary', [BATCH_SIZE, 1]), ('category', [BATCH_SIZE, 16]), ('number', [BATCH_SIZE, 1]), ('category', [BATCH_SIZE, 48]), ('number', [BATCH_SIZE, 32]), ('binary', [BATCH_SIZE, 1])]])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('reduce_output', ['concat', 'sum'])
@pytest.mark.parametrize('fc_layers', [None, [{'output_size': 256}]])
@pytest.mark.parametrize('embed_input_feature_name', [None, 64, 'add'])
def test_tabtransformer_combiner_number_and_binary_with_category(features_to_test: tuple, embed_input_feature_name: Optional[Union[int, str]], fc_layers: Optional[list], reduce_output: str, num_layers: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    set_random_seed(RANDOM_SEED)
    (encoder_outputs, input_features) = features_to_test
    combiner = TabTransformerCombiner(input_features=input_features, config=load_config(TabTransformerCombinerConfig, embed_input_feature_name=embed_input_feature_name, num_layers=num_layers, fc_layers=fc_layers, reduce_output=reduce_output)).to(DEVICE)
    combiner_output = combiner(encoder_outputs)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
    target = torch.randn(combiner_output['combiner_output'].shape)
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_outputs,), target)
    number_category_features = sum((input_features[i_f].type() == CATEGORY for i_f in input_features))
    adjustment_for_single_category = 1 if number_category_features == 1 else 0
    assert upc == tpc - adjustment_for_single_category * (num_layers * PARAMETERS_IN_SELF_ATTENTION), f'Failed to update parameters. Parameters not updated: {not_updated}'

@pytest.mark.parametrize('feature_list', [[('binary', [BATCH_SIZE, 1]), ('binary', [BATCH_SIZE, 1])], [('number', [BATCH_SIZE, 1]), ('number', [BATCH_SIZE, 1])], [('number', [BATCH_SIZE, 1]), ('binary', [BATCH_SIZE, 1])]])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('reduce_output', ['concat', 'sum'])
@pytest.mark.parametrize('fc_layers', [None, [{'output_size': 256}]])
@pytest.mark.parametrize('embed_input_feature_name', [None, 64, 'add'])
def test_tabtransformer_combiner_number_or_binary_without_category(features_to_test: tuple, embed_input_feature_name: Optional[Union[int, str]], fc_layers: Optional[list], reduce_output: str, num_layers: int) -> None:
    if False:
        i = 10
        return i + 15
    set_random_seed(RANDOM_SEED)
    (encoder_outputs, input_features) = features_to_test
    combiner = TabTransformerCombiner(input_features=input_features, config=load_config(TabTransformerCombinerConfig, embed_input_feature_name=embed_input_feature_name, num_layers=num_layers, fc_layers=fc_layers, reduce_output=reduce_output)).to(DEVICE)
    combiner_output = combiner(encoder_outputs)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
    target = torch.randn(combiner_output['combiner_output'].shape)
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_outputs,), target)
    assert upc == tpc - num_layers * PARAMETERS_IN_TRANSFORMER_BLOCK - (1 if embed_input_feature_name is not None else 0), f'Failed to update parameters. Parameters not updated: {not_updated}'

@pytest.mark.parametrize('feature_list', [[('binary', [BATCH_SIZE, 1]), ('category', [BATCH_SIZE, 16]), ('binary', [BATCH_SIZE, 1]), ('category', [BATCH_SIZE, 32])], [('number', [BATCH_SIZE, 1]), ('category', [BATCH_SIZE, 16]), ('number', [BATCH_SIZE, 1]), ('category', [BATCH_SIZE, 32])], [('number', [BATCH_SIZE, 1]), ('category', [BATCH_SIZE, 16]), ('binary', [BATCH_SIZE, 1]), ('category', [BATCH_SIZE, 32])]])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('reduce_output', ['concat', 'sum'])
@pytest.mark.parametrize('fc_layers', [None, [{'output_size': 256}]])
@pytest.mark.parametrize('embed_input_feature_name', [None, 64, 'add'])
def test_tabtransformer_combiner_number_or_binary_with_category(features_to_test: tuple, embed_input_feature_name: Optional[Union[int, str]], fc_layers: Optional[list], reduce_output: str, num_layers: int) -> None:
    if False:
        i = 10
        return i + 15
    set_random_seed(RANDOM_SEED)
    (encoder_outputs, input_features) = features_to_test
    combiner = TabTransformerCombiner(input_features=input_features, config=load_config(TabTransformerCombinerConfig, embed_input_feature_name=embed_input_feature_name, num_layers=num_layers, fc_layers=fc_layers, reduce_output=reduce_output)).to(DEVICE)
    combiner_output = combiner(encoder_outputs)
    check_combiner_output(combiner, combiner_output, BATCH_SIZE)
    target = torch.randn(combiner_output['combiner_output'].shape)
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(combiner, (encoder_outputs,), target)
    assert upc == tpc, f'Failed to update parameters. Parameters not updated: {not_updated}'